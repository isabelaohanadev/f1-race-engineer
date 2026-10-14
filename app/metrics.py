"""Métricas Prometheus da API (`GET /metrics`).

CADA PROCESSO EXPÕE AS SUAS. A API e o worker são processos separados, cada um com seu registro e
seu /metrics (o do worker, numa porta própria: worker/metrics.py). Prometheus os raspa como alvos
distintos (o rótulo `instance` os diferencia) e agrega com `sum by (...)`. Não usamos o modo
multiprocess do prometheus-client: o motivo está no README.

O QUE É CALCULADO NA HORA DO SCRAPE (collectors) E O QUE É CONTADO (Counter/Histogram)
* Clientes WebSocket conectados e o estado do Redis (tamanho do stream, lag/pendentes do consumer
  group) são ESTADO ATUAL: um Collector os lê quando o Prometheus pede. Nada de laço de fundo que
  atualiza um Gauge a cada N segundos: ele ficaria desatualizado entre as atualizações, gastaria
  Redis mesmo sem ninguém raspando e seria mais uma tarefa para dar errado. No scrape o valor é,
  por construção, o de agora.
* Eventos publicados, conexões, desconexões de cliente lento e a latência de entrega são
  ACONTECIMENTOS: só dá para contá-los quando ocorrem (Counter/Histogram).

O LAG VEM DA API, NÃO DO WORKER. É a métrica que alerta "os workers não estão dando conta". Se ela
dependesse do worker, quando TODOS os workers morressem (o pior caso) ela sumiria junto, e um
alerta "lag > X" não dispararia. Lida pela API (que está viva), ela continua subindo.

CARDINALIDADE. Todo rótulo multiplica o número de séries que o Prometheus guarda. Por isso os
únicos rótulos são `race` (poucos valores: as corridas que já tiveram replay), `group` (1 valor) e
`le` (dos buckets). NUNCA piloto, ID de evento, ID de entrada, nome de consumidor ou cliente: seriam
milhares de séries, e uma nova a cada evento. Há um teste que garante isso.
"""

import logging
import re

from prometheus_client import (
    CollectorRegistry,
    Counter,
    GCCollector,
    Histogram,
    ProcessCollector,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from redis import Redis as SyncRedis
from redis.exceptions import RedisError

from app.hub import HubRegistry
from replay.latency import LATENCY_BUCKETS
from replay.state import STREAMS_REGISTRY_KEY, WORKER_GROUP

logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()
ProcessCollector(
    registry=REGISTRY
)  # memória/CPU/fds do processo (útil para provar "sem vazamento")
GCCollector(registry=REGISTRY)

# De 1 ms a 1 s (a régua é compartilhada com o worker: replay/latency.py).
DELIVERY_BUCKETS = LATENCY_BUCKETS

DELIVERY_LATENCY = Histogram(
    "f1_delivery_latency_seconds",
    "Latência entre a PUBLICAÇÃO do evento no stream e o fim do envio ao cliente WebSocket "
    "(só eventos ao vivo; o catch-up não entra). O instante da publicação é a parte de tempo do ID "
    "do stream, do relógio do Redis: com Redis e API em máquinas diferentes, inclui o desvio "
    "entre os relógios.",
    buckets=DELIVERY_BUCKETS,
    registry=REGISTRY,
)
EVENTS_PUBLISHED = Counter(
    "f1_replay_events_published_total",
    "Eventos que o motor de replay desta instância publicou no stream. Use rate() para eventos/s.",
    registry=REGISTRY,
)
WS_CONNECTIONS = Counter(
    "f1_ws_connections_total", "Conexões WebSocket aceitas.", registry=REGISTRY
)
WS_SLOW_DISCONNECTS = Counter(
    "f1_ws_slow_client_disconnects_total",
    "Clientes desconectados (código 1013) por não acompanharem a velocidade dos eventos.",
    registry=REGISTRY,
)

_STREAM_RE = re.compile(r"^race:(\d+):(\d+)$")


class ClientsCollector(Collector):
    """Clientes WebSocket conectados a ESTA instância, lidos dos hubs no instante do scrape."""

    def __init__(self, hubs: HubRegistry) -> None:
        self._hubs = hubs

    def collect(self):
        yield GaugeMetricFamily(
            "f1_ws_clients_connected",
            "Clientes WebSocket conectados a esta instância da API.",
            value=self._hubs.clients,
        )


class StreamCollector(Collector):
    """Tamanho dos streams e lag/pendentes do consumer group, lidos do Redis NO SCRAPE.

    Se o Redis não responder, o scrape NÃO falha: `f1_redis_up` vai a 0 e as demais métricas desta
    família ficam de fora. Falhar o scrape inteiro apagaria também as métricas que não dependem do
    Redis (e esconderia o problema atrás de um alvo "down").
    """

    def __init__(self, redis: SyncRedis) -> None:
        self._redis = redis

    def collect(self):
        up = GaugeMetricFamily("f1_redis_up", "1 se o scrape conseguiu ler o Redis, 0 se não.")
        length = GaugeMetricFamily(
            "f1_stream_length", "Entradas retidas no stream da corrida.", labels=["race"]
        )
        lag = GaugeMetricFamily(
            "f1_consumer_group_lag",
            "Entradas do stream que o consumer group AINDA NÃO LEU (os workers estão atrasados "
            "quando isto sobe e não desce).",
            labels=["race", "group"],
        )
        pending = GaugeMetricFamily(
            "f1_consumer_group_pending",
            "Entradas entregues ao grupo e ainda sem XACK (em processamento, ou de um worker que "
            "morreu e espera o XAUTOCLAIM).",
            labels=["race", "group"],
        )
        try:
            streams = sorted(self._redis.smembers(STREAMS_REGISTRY_KEY))  # type: ignore[misc]
            with self._redis.pipeline(transaction=False) as pipe:
                for stream in streams:
                    pipe.xlen(stream)
                    pipe.xinfo_groups(stream)
                results = pipe.execute(raise_on_error=False)  # um stream sumido não derruba o resto
            for index, stream in enumerate(streams):
                match = _STREAM_RE.match(stream)
                stream_length, groups = results[2 * index], results[2 * index + 1]
                if match is None or isinstance(stream_length, Exception):
                    continue
                race = f"{match.group(1)}-{match.group(2)}"
                length.add_metric([race], stream_length)
                for group in [] if isinstance(groups, Exception) else groups:
                    if group["name"] != WORKER_GROUP:
                        continue
                    pending.add_metric([race, WORKER_GROUP], group["pending"])
                    if group.get("lag") is not None:  # o Redis não sabe calcular após XDEL/trim
                        lag.add_metric([race, WORKER_GROUP], group["lag"])
            up.add_metric([], 1)
        except (RedisError, OSError) as exc:
            # Sem traceback: o Prometheus raspa a cada ~15 s; um por scrape inundaria o log.
            logger.warning("Scrape: Redis indisponível (%s)", exc)
            up = GaugeMetricFamily("f1_redis_up", "1 se o scrape conseguiu ler o Redis, 0 se não.")
            up.add_metric([], 0)
        yield up
        yield length
        yield lag
        yield pending
