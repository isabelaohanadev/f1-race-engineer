"""Métricas Prometheus do CONSUMIDOR DE ESTRATÉGIA (registro próprio, como o do worker).

O consumidor de estratégia é um processo à parte da API: tem o seu registro (`REGISTRY`) e, no
passo das métricas, a sua porta. Rótulos: só `result`, com 4 valores fixos. NUNCA piloto, corrida,
volta nem ID de evento: a regra de cardinalidade é a de app/metrics.py (um rótulo que cresce com os
dados derruba o Prometheus).

`result`:
* `completed`: a simulação terminou e o resultado foi entregue;
* `cancelled`: chegou a volta seguinte e a simulação parou no meio (o resultado antigo é
descartado);
* `stale`: terminou, mas uma volta mais nova já tinha chegado: não é entregue;
* `failed`: a simulação levantou uma exceção (o consumidor segue com a próxima volta).
"""

import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    ProcessCollector,
)

REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)

RESULTS = ("completed", "cancelled", "stale", "failed")
# Uma volta dura ~90 s e uma simulação de 5.000 trajetórias leva de 0,05 s (fim de corrida) a ~1,3 s
# (corrida inteira, 20 carros), medido; os buckets vão de 25 ms a 30 s para pegar tanto o normal
# quanto uma máquina lenta ou um travamento.
DURATION_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

SIMULATION_DURATION = Histogram(
    "f1_strategy_simulation_duration_seconds",
    "Duração de uma simulação Monte Carlo (5.000 trajetórias da volta atual até o fim), por "
    "resultado. As canceladas medem só até o ponto em que pararam.",
    ["result"],
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)
SIMULATIONS = Counter(
    "f1_strategy_simulations_total",
    "Simulações por resultado: `completed` entregue; `cancelled` parada pela volta seguinte; "
    "`stale` terminou depois de uma volta mais nova ter chegado; `failed` com exceção.",
    ["result"],
    registry=REGISTRY,
)
# Séries que só nascem no 1º evento quebram rate() e alertas de ausência: nascem com 0, na partida.
for _result in RESULTS:
    SIMULATIONS.labels(_result)
    SIMULATION_DURATION.labels(_result)

_last_delivered_at: float | None = None
_clock = time.time


def record(duration_s: float, result: str) -> None:
    """Registra uma simulação (duração e resultado); `completed` também marca o instante do último
    resultado entregue, de onde sai a idade."""
    global _last_delivered_at
    if result not in RESULTS:
        raise ValueError(f"resultado desconhecido: {result!r}")
    SIMULATION_DURATION.labels(result).observe(duration_s)
    SIMULATIONS.labels(result).inc()
    if result == "completed":
        _last_delivered_at = _clock()


def last_result_age() -> float:
    """Segundos desde o último resultado entregue (NaN enquanto não houve nenhum: zero pareceria
    "fresco", e uma idade enorme pareceria "parado")."""
    return float("nan") if _last_delivered_at is None else max(_clock() - _last_delivered_at, 0.0)


LAST_RESULT_AGE = Gauge(
    "f1_strategy_last_result_age_seconds",
    "Idade do último resultado de estratégia entregue, lida no scrape. Sobe a ~1 s por segundo "
    "enquanto não chega uma volta nova ou a simulação trava; NaN antes do primeiro.",
    registry=REGISTRY,
)
LAST_RESULT_AGE.set_function(last_result_age)
