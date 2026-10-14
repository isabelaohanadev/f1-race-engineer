"""Métricas Prometheus do WORKER (`/metrics` numa porta própria: WORKER_METRICS_PORT, padrão 9108).

O worker é um processo à parte da API, com registro e porta próprios. Rótulos: só `result` (3
valores fixos). Nunca piloto, ID de evento, nome de consumidor (muda a cada reinício) nem stream:
ver a regra de cardinalidade em app/metrics.py.
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    GCCollector,
    Histogram,
    ProcessCollector,
)

from replay.latency import LATENCY_BUCKETS

REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)

EVENTS = Counter(
    "f1_worker_events_total",
    "Entradas do stream processadas por este worker, por resultado: `applied` gravou no estado; "
    "`ignored` era velha, repetida ou de replay antigo (a trava de ordem em ação); `malformed` "
    "era ilegível (confirmada e descartada).",
    ["result"],
    registry=REGISTRY,
)
# Série que só nasce no 1º evento quebra rate() e alertas de ausência: nasce com 0, na partida.
for _result in ("applied", "ignored", "malformed"):
    EVENTS.labels(_result)

CLAIMED = Counter(
    "f1_worker_claimed_total",
    "Entradas resgatadas de um worker que sumiu (XAUTOCLAIM). Sobe quando algum worker morre.",
    registry=REGISTRY,
)
BATCHES = Counter(
    "f1_worker_batches_total", "Lotes lidos do stream e aplicados.", registry=REGISTRY
)
APPLY_LATENCY = Histogram(
    "f1_worker_apply_latency_seconds",
    "Latência entre a PUBLICAÇÃO da entrada no stream e o estado já gravado no Redis: o quão "
    "fresco está o snapshot. Não inclui entradas resgatadas (a espera delas é longa, por "
    "definição).",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
