"""As métricas de ponta a ponta: o /metrics REAL da API (servidor uvicorn) e o do worker, raspados
ANTES e DEPOIS de um replay curto, com cliente WebSocket conectado, workers e Redis de verdade.

O critério não é só "a métrica existe": os valores têm de MUDAR e BATER ENTRE SI (o que a API diz
que publicou é o que o stream tem, é o que os workers dizem que processaram, é o que o grupo diz que
já leu)."""

import asyncio

import httpx
from prometheus_client import start_http_server
from prometheus_client.parser import text_string_to_metric_families
from redis.asyncio import Redis

from app import metrics as api_metrics
from app.main import app
from replay.manager import ReplayManager
from replay.state import WORKER_GROUP, RaceStateStore
from tests.integration.helpers import (
    RACE,
    STREAM,
    drained,
    final_truth,
    make_worker,
    run_client,
    running,
    wait_until,
)
from worker import metrics as worker_metrics

RACE_LABEL = {"race": f"{RACE[0]}-{RACE[1]}"}


async def scrape(url: str) -> dict[tuple[str, tuple], float]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    assert response.status_code == 200
    families = text_string_to_metric_families(response.text)
    return {(s.name, tuple(sorted(s.labels.items()))): s.value for f in families for s in f.samples}


def get(found: dict, name: str, **labels: str) -> float | None:
    return found.get((name, tuple(sorted(labels.items()))))


def delta(before: dict, after: dict, name: str, **labels: str) -> float:
    return (get(after, name, **labels) or 0.0) - (get(before, name, **labels) or 0.0)


def http_base(ws_url: str) -> str:
    return ws_url.split("/ws/")[0].replace("ws://", "http://")


async def test_metrics_appear_and_change_after_a_short_replay(
    url: str, redis_client: Redis, store: RaceStateStore, synthetic_race
):
    api_url = f"{http_base(url)}/metrics"
    worker_server, _ = start_http_server(0, addr="127.0.0.1", registry=worker_metrics.REGISTRY)
    worker_url = f"http://127.0.0.1:{worker_server.server_port}/metrics"
    manager = ReplayManager(
        synthetic_race,
        redis_client,
        page_size=100,
        on_events_published=api_metrics.EVENTS_PUBLISHED.inc,  # a mesma ligação do lifespan
    )
    try:
        api_before, worker_before = await scrape(api_url), await scrape(worker_url)
        # ANTES: o que já existe e o que ainda não (nenhum stream, nenhum cliente).
        assert get(api_before, "f1_redis_up") == 1
        assert get(api_before, "f1_ws_clients_connected") == 0
        assert get(api_before, "f1_stream_length", **RACE_LABEL) is None  # ainda não há stream

        workers = [make_worker(redis_client, consumer=f"w{i}", batch_size=20) for i in range(2)]
        async with running(workers[0]), running(workers[1]):
            await manager.start(*RACE, speed=16)
            watcher = asyncio.create_task(run_client(url, until=lambda view: view.finished))
            await asyncio.sleep(0.7)  # no MEIO do replay
            api_mid = await scrape(api_url)
            await watcher
            await final_truth(store, redis_client)  # os workers esgotaram o stream
        await manager.shutdown()
        await wait_until(_no_clients)  # o cliente já saiu e o hub foi liberado
        api_after, worker_after = await scrape(api_url), await scrape(worker_url)
        entries = await redis_client.xlen(STREAM)
    finally:
        worker_server.shutdown()
        worker_server.server_close()

    # ---- durante o replay (API)
    assert get(api_mid, "f1_ws_clients_connected") == 1  # o cliente conectado, no scrape
    mid_length = get(api_mid, "f1_stream_length", **RACE_LABEL)
    assert mid_length is not None and 0 < mid_length < entries  # o stream CRESCE durante o replay

    # ---- depois (API): cada número bate com a realidade
    assert get(api_after, "f1_ws_clients_connected") == 0
    assert delta(api_before, api_after, "f1_replay_events_published_total") == entries
    assert delta(api_before, api_after, "f1_ws_connections_total") == 1
    assert get(api_after, "f1_stream_length", **RACE_LABEL) == entries
    group = {**RACE_LABEL, "group": WORKER_GROUP}
    assert get(api_after, "f1_consumer_group_lag", **group) == 0  # os workers alcançaram
    assert get(api_after, "f1_consumer_group_pending", **group) == 0
    assert get(api_after, "f1_redis_up") == 1

    # ---- latência de entrega: um histograma de verdade, coerente
    observed = delta(api_before, api_after, "f1_delivery_latency_seconds_count")
    assert 0 < observed <= entries  # só eventos ao vivo; o catch-up não entra
    assert delta(api_before, api_after, "f1_delivery_latency_seconds_sum") > 0
    cumulative = [
        delta(api_before, api_after, "f1_delivery_latency_seconds_bucket", le=bound)
        for bound in ("0.001", "0.005", "0.01", "0.1", "1.0", "+Inf")
    ]
    assert cumulative == sorted(cumulative)  # os buckets são cumulativos
    assert cumulative[-1] == observed  # o +Inf vê tudo
    assert cumulative[-2] >= 0.99 * observed  # na mesma máquina, quase tudo entrega em < 1 s

    # ---- o worker (OUTRO /metrics): o que ele processou bate com o que a API publicou
    processed = delta(
        worker_before, worker_after, "f1_worker_events_total", result="applied"
    ) + delta(worker_before, worker_after, "f1_worker_events_total", result="ignored")
    assert processed == entries  # cada entrada foi processada UMA vez, aplicada ou ignorada
    assert delta(worker_before, worker_after, "f1_worker_events_total", result="applied") > 0
    assert delta(worker_before, worker_after, "f1_worker_batches_total") > 0
    assert delta(worker_before, worker_after, "f1_worker_claimed_total") == 0  # ninguém morreu
    assert delta(worker_before, worker_after, "f1_worker_apply_latency_seconds_count") == entries


async def _no_clients() -> bool:
    return app.state.hubs.clients == 0 and app.state.hubs.hubs == []


async def test_each_process_exposes_only_its_own_metrics(url: str):
    """A API e o worker têm registros separados: nenhum vaza para o /metrics do outro."""
    worker_server, _ = start_http_server(0, addr="127.0.0.1", registry=worker_metrics.REGISTRY)
    try:
        api = await scrape(f"{http_base(url)}/metrics")
        worker = await scrape(f"http://127.0.0.1:{worker_server.server_port}/metrics")
    finally:
        worker_server.shutdown()
        worker_server.server_close()

    api_names = {name for name, _ in api}
    worker_names = {name for name, _ in worker}
    assert any(n.startswith("f1_ws_") or n.startswith("f1_delivery") for n in api_names)
    assert not any(n.startswith("f1_worker_") for n in api_names)
    assert any(n.startswith("f1_worker_") for n in worker_names)
    assert not any(
        n.startswith(("f1_ws_", "f1_delivery", "f1_stream", "f1_consumer")) for n in worker_names
    )


def _worker_sample(name: str, **labels: str) -> float:
    return worker_metrics.REGISTRY.get_sample_value(name, labels) or 0.0


async def test_rescued_entries_count_as_claimed_but_never_pollute_the_freshness_histogram(
    redis_client: Redis, store: RaceStateStore
):
    """Um worker morreu com 30 entradas pendentes; outro as resgata (XAUTOCLAIM). Elas esperaram,
    por definição, muito: contá-las em `apply_latency` faria um worker morto por 30 s parecer um
    estado "defasado 30 s" para TODAS as entradas normais. Só o contador de resgates sobe."""
    from replay.events import PositionEvent
    from replay.publisher import StreamPublisher

    await store.register_stream(*RACE)
    await redis_client.xgroup_create(STREAM, WORKER_GROUP, id="0", mkstream=True)
    await StreamPublisher(redis_client, STREAM, epoch=1).publish_many(
        [PositionEvent("VER", t, -t, t) for t in range(1, 31)]
    )
    await redis_client.xreadgroup(WORKER_GROUP, "morto", {STREAM: ">"}, count=100)  # e some

    claimed_before = _worker_sample("f1_worker_claimed_total")
    latency_before = _worker_sample("f1_worker_apply_latency_seconds_count")
    applied_before = _worker_sample("f1_worker_events_total", result="applied")
    async with running(make_worker(redis_client, consumer="vivo", claim_min_idle_ms=100)):
        await wait_until(lambda: drained(redis_client))

    assert _worker_sample("f1_worker_claimed_total") - claimed_before == 30
    assert _worker_sample("f1_worker_events_total", result="applied") - applied_before == 30
    assert _worker_sample("f1_worker_apply_latency_seconds_count") == latency_before  # nenhuma
