"""Métricas Prometheus: o que o /metrics expõe, o que é lido NO SCRAPE e a regra de cardinalidade.

Os valores que MUDAM depois de um replay de verdade (API, worker, Redis real) são verificados em
tests/integration/test_metrics.py."""

import sys
import time

import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

import app.main as main_module
from app import metrics as api_metrics
from app.config import RedisSettings
from app.hub import HubRegistry
from app.main import app
from app.redis_client import create_sync_redis
from replay.latency import LATENCY_BUCKETS
from replay.state import STREAMS_REGISTRY_KEY, WORKER_GROUP
from worker import metrics as worker_metrics

# Os ÚNICOS rótulos permitidos em qualquer métrica nossa. `le` é dos buckets do histograma e
# `generation` do coletor de lixo padrão do Python (3 valores fixos: 0, 1 e 2).
ALLOWED_LABELS = {"race", "group", "result", "le", "quantile", "generation"}
FORBIDDEN_LABELS = {"driver", "piloto", "id", "event_id", "entry_id", "consumer", "client", "t"}


class FakePipeline:
    def __init__(self, redis: "FakeSyncRedis") -> None:
        self._redis, self._commands = redis, []

    def __enter__(self) -> "FakePipeline":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def xlen(self, stream: str) -> None:
        self._commands.append(("xlen", stream))

    def xinfo_groups(self, stream: str) -> None:
        self._commands.append(("xinfo", stream))

    def execute(self, raise_on_error: bool = True) -> list:
        if self._redis.down:
            raise RedisConnectionError("redis fora")
        results = []
        for kind, stream in self._commands:
            if stream not in self._redis.streams:
                results.append(ResponseError("no such key"))  # ficou no registro, mas sumiu
            elif kind == "xlen":
                results.append(self._redis.streams[stream]["length"])
            else:
                results.append(self._redis.streams[stream]["groups"])
        return results


class FakeSyncRedis:
    """Um Redis síncrono mínimo, com o que o collector lê: o registro de streams, o tamanho e os
    grupos de cada um. Os testes MUDAM estes dados entre dois scrapes."""

    def __init__(self) -> None:
        self.registry: set[str] = set()
        self.streams: dict[str, dict] = {}
        self.down = False

    def smembers(self, key: str) -> set[str]:
        assert key == STREAMS_REGISTRY_KEY
        if self.down:
            raise RedisConnectionError("redis fora")
        return set(self.registry)

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        return FakePipeline(self)

    def close(self) -> None:
        return None

    def add_stream(self, stream: str, length: int, *, lag, pending: int) -> None:
        self.registry.add(stream)
        group = {"name": WORKER_GROUP, "lag": lag, "pending": pending}
        self.streams[stream] = {"length": length, "groups": [group]}


def samples(registry_or_collector) -> dict[tuple[str, tuple], float]:
    """{(nome da amostra, ((rótulo, valor), ...)): valor} de um registro ou collector."""
    families = (
        registry_or_collector.collect()
        if hasattr(registry_or_collector, "collect")
        else registry_or_collector
    )
    return {(s.name, tuple(sorted(s.labels.items()))): s.value for f in families for s in f.samples}


def value(found: dict, name: str, **labels: str) -> float | None:
    return found.get((name, tuple(sorted(labels.items()))))


# ---------------------------------------------------------------- collector do Redis, NO scrape


def test_lag_pending_and_length_are_read_from_redis_at_scrape_time():
    redis = FakeSyncRedis()
    collector = api_metrics.StreamCollector(redis)  # type: ignore[arg-type]
    redis.add_stream("race:2024:21", 1_000, lag=40, pending=7)

    first = samples(collector)
    redis.add_stream("race:2024:21", 5_000, lag=0, pending=0)  # o Redis MUDOU entre os scrapes
    second = samples(collector)

    race = {"race": "2024-21"}
    grp = {**race, "group": WORKER_GROUP}
    assert value(first, "f1_stream_length", **race) == 1_000
    assert value(first, "f1_consumer_group_lag", **grp) == 40
    assert value(first, "f1_consumer_group_pending", **grp) == 7
    assert value(first, "f1_redis_up") == 1
    # Nada em cache nem laço de fundo: cada scrape vê o valor DE AGORA.
    assert value(second, "f1_stream_length", **race) == 5_000
    assert value(second, "f1_consumer_group_lag", **grp) == 0


def test_a_redis_outage_turns_redis_up_to_zero_instead_of_failing_the_scrape():
    redis = FakeSyncRedis()
    redis.add_stream("race:2024:21", 10, lag=0, pending=0)
    collector = api_metrics.StreamCollector(redis)  # type: ignore[arg-type]
    redis.down = True

    found = samples(collector)  # não levanta

    assert value(found, "f1_redis_up") == 0
    assert not [name for name, _ in found if name != "f1_redis_up"]  # sem números velhos


def test_the_scrape_fails_fast_against_a_really_unreachable_redis():
    """Sem dublê: o cliente REAL apontado para uma porta fechada. O retry com backoff do redis-py
    faria este scrape levar ~3,4 s (medido); sem retry ele responde na hora."""
    redis = create_sync_redis(RedisSettings(_env_file=None, host="127.0.0.1", port=1))
    started = time.monotonic()

    found = samples(api_metrics.StreamCollector(redis))

    assert time.monotonic() - started < 1.0
    assert value(found, "f1_redis_up") == 0


def test_unknown_lag_a_vanished_stream_and_foreign_keys_are_skipped_not_faked():
    redis = FakeSyncRedis()
    redis.add_stream("race:2024:21", 10, lag=None, pending=3)  # o Redis não soube calcular o lag
    redis.registry |= {"race:2023:1", "algo-que-nao-e-corrida"}  # sumiu do Redis / chave estranha
    found = samples(api_metrics.StreamCollector(redis))  # type: ignore[arg-type]

    grp = {"race": "2024-21", "group": WORKER_GROUP}
    assert value(found, "f1_consumer_group_lag", **grp) is None  # ausente, não 0 nem NaN
    assert value(found, "f1_consumer_group_pending", **grp) == 3
    assert value(found, "f1_stream_length", race="2023-1") is None
    assert value(found, "f1_redis_up") == 1


class FakeRegistryHubs(HubRegistry):
    def __init__(self) -> None:
        self.count = 0

    @property
    def clients(self) -> int:
        return self.count


def test_connected_clients_are_read_from_the_hubs_at_scrape_time():
    hubs = FakeRegistryHubs()
    collector = api_metrics.ClientsCollector(hubs)

    hubs.count = 3
    first = value(samples(collector), "f1_ws_clients_connected")
    hubs.count = 0
    second = value(samples(collector), "f1_ws_clients_connected")

    assert (first, second) == (3, 0)  # sempre presente (0 quando não há clientes)


# ---------------------------------------------------------------- histograma e cardinalidade


def test_the_latency_histogram_covers_1_ms_to_1_s():
    assert LATENCY_BUCKETS[0] == 0.001 and LATENCY_BUCKETS[-1] == 1.0
    assert list(LATENCY_BUCKETS) == sorted(set(LATENCY_BUCKETS))  # crescentes, sem repetição
    assert api_metrics.DELIVERY_BUCKETS == LATENCY_BUCKETS  # API e worker usam a MESMA régua

    exposed = generate_latest(api_metrics.REGISTRY).decode()

    for bound in ("0.001", "0.005", "0.01", "0.1", "1.0", "+Inf"):
        assert f'f1_delivery_latency_seconds_bucket{{le="{bound}"}}' in exposed
    assert "f1_worker_apply_latency_seconds" in generate_latest(worker_metrics.REGISTRY).decode()


def test_worker_counters_exist_at_zero_from_startup_not_only_after_the_first_event():
    """Série que só aparece no 1º evento (ex.: `malformed`) quebra rate() e alertas de ausência."""
    exposed = samples(worker_metrics.REGISTRY.collect())

    for result in ("applied", "ignored", "malformed"):
        assert value(exposed, "f1_worker_events_total", result=result) is not None
    assert value(exposed, "f1_worker_claimed_total") is not None
    assert value(exposed, "f1_worker_batches_total") is not None


def test_no_metric_uses_a_high_cardinality_label():
    """Piloto, ID de evento/entrada, consumidor ou cliente como rótulo criaria milhares de séries
    (e uma nova a cada evento). Varre os DOIS registros (API e worker) com dados de verdade."""
    redis = FakeSyncRedis()
    redis.add_stream("race:2024:21", 10, lag=1, pending=1)
    api_metrics.REGISTRY.register(collector := api_metrics.StreamCollector(redis))  # type: ignore[arg-type]
    try:
        api_metrics.DELIVERY_LATENCY.observe(0.004)
        worker_metrics.EVENTS.labels("applied").inc()
        used: set[str] = set()
        for registry in (api_metrics.REGISTRY, worker_metrics.REGISTRY):
            for family in registry.collect():
                for sample in family.samples:
                    used |= set(sample.labels)
    finally:
        api_metrics.REGISTRY.unregister(collector)

    assert not used & FORBIDDEN_LABELS
    assert used <= ALLOWED_LABELS, f"rótulo inesperado: {used - ALLOWED_LABELS}"


# ---------------------------------------------------------------- o endpoint /metrics


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    """A API com um Redis síncrono FALSO no lifespan: não toca o Redis de desenvolvimento."""
    redis = FakeSyncRedis()
    redis.add_stream("race:2024:21", 123, lag=4, pending=2)
    monkeypatch.setattr(main_module, "create_sync_redis", lambda: redis)
    with TestClient(app) as client:
        yield client


def test_the_metrics_endpoint_serves_the_prometheus_text_format(api: TestClient):
    response = api.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    families = {f.name: f for f in text_string_to_metric_families(response.text)}
    for name in (
        "f1_delivery_latency_seconds",
        "f1_replay_events_published",  # o parser tira o sufixo _total do nome da família
        "f1_ws_connections",
        "f1_ws_slow_client_disconnects",
        "f1_ws_clients_connected",
        "f1_redis_up",
        "f1_stream_length",
        "f1_consumer_group_lag",
        "f1_consumer_group_pending",
        "python_gc_objects_collected",
    ):
        assert name in families, f"{name} não está no /metrics"
    if sys.platform.startswith("linux"):  # o ProcessCollector lê /proc: só existe no Linux
        assert "process_resident_memory_bytes" in families


def test_the_collectors_are_registered_by_the_lifespan_and_removed_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sem retirar os collectors ao desligar, o 2º lifespan (ou cada TestClient) falharia com
    "Duplicated timeseries"."""
    redis = FakeSyncRedis()
    monkeypatch.setattr(main_module, "create_sync_redis", lambda: redis)

    for _ in range(3):  # sobe e desce 3 vezes, sem erro de registro duplicado
        with TestClient(app) as client:
            assert "f1_ws_clients_connected" in client.get("/metrics").text
    assert "f1_ws_clients_connected" not in generate_latest(api_metrics.REGISTRY).decode()
