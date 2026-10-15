"""Testes do executor "só a última" (strategy/forecast_runner.py) e das métricas dele
(strategy/metrics.py).

Os trabalhos de teste ficam presos numa porta (`threading.Event`) e conferem o sinal de
cancelamento,
então a ordem dos eventos é controlada pelo teste e nenhum depende de quanto tempo algo demora.
"""

import asyncio
import logging
import math
import threading

import pytest

from strategy import metrics
from strategy.forecast_runner import Job, LatestOnlyRunner, forecast_job
from strategy.history import (
    DEFAULT_COST_RATIO,
    CompoundPrior,
    RetirementPrior,
    SafetyCarPrior,
    SimPriors,
)
from strategy.montecarlo import SimConfig, SimulationCancelled
from tests.strategy.synthetic import World, chase_field


class Harness:
    """Um executor com os resultados entregues, as durações observadas e as threads usadas."""

    def __init__(self, *, fail_delivery_for: set | None = None) -> None:
        self.delivered: list[tuple] = []
        self.observed: list[tuple[float, str]] = []
        self.threads: dict = {}
        self.ran: list = []
        self._fail_delivery_for = fail_delivery_for or set()
        self.runner: LatestOnlyRunner = LatestOnlyRunner(self._on_result, observe=self._observe)

    async def _on_result(self, tag, result) -> None:
        if tag in self._fail_delivery_for:
            raise RuntimeError("a entrega falhou")
        self.delivered.append((tag, result))

    def _observe(self, duration: float, result: str) -> None:
        self.observed.append((duration, result))

    def job(
        self,
        tag,
        gate: threading.Event | None = None,
        *,
        honors_cancel: bool = True,
        raises: Exception | None = None,
    ) -> Job:
        """Um trabalho que, enquanto `gate` não abrir, confere o cancelamento a cada 5 ms."""
        started = threading.Event()
        self.threads[tag] = (started, [])

        def run(should_stop):
            self.ran.append(tag)
            self.threads[tag][1].append(threading.get_ident())
            started.set()
            while gate is not None and not gate.wait(0.005):
                if honors_cancel and should_stop():
                    raise SimulationCancelled
            if raises is not None:
                raise raises
            return f"resultado de {tag}"

        return Job(tag, run)

    async def started(self, tag) -> None:
        flag = self.threads[tag][0]
        for _ in range(600):
            if flag.is_set():
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"o trabalho {tag!r} não começou")


@pytest.fixture
async def harness():
    h = Harness()
    yield h
    await h.runner.close()


# ------------------------------------------------------------------ a política


async def test_a_job_runs_in_a_worker_thread_and_its_result_is_delivered(harness):
    job = harness.job("v30")

    await harness.runner.submit(job)
    await harness.runner.wait_idle()

    assert harness.delivered == [("v30", "resultado de v30")]
    assert harness.threads["v30"][1][0] != threading.get_ident()  # fora do event loop
    assert harness.runner.stats.completed == 1
    assert [result for _, result in harness.observed] == ["completed"]
    assert harness.observed[0][0] >= 0


async def test_the_event_loop_keeps_running_while_a_job_computes(harness):
    """O ponto de rodar numa thread: com o trabalho preso, o loop continua girando."""
    gate = threading.Event()
    await harness.runner.submit(harness.job("v30", gate))
    await harness.started("v30")

    ticks = 0
    for _ in range(200):
        await asyncio.sleep(0)
        ticks += 1
    gate.set()
    await harness.runner.wait_idle()

    assert ticks == 200 and harness.delivered


async def test_a_newer_lap_cancels_the_running_job_and_replaces_the_one_waiting(harness):
    """A, B, C: A roda, B espera e C chega. A é cancelada, B é substituída (nunca roda: sem fila) e
    só C é entregue."""
    stuck, free = threading.Event(), threading.Event()
    free.set()

    await harness.runner.submit(harness.job("A", stuck))
    await harness.started("A")
    await harness.runner.submit(harness.job("B", free))
    await harness.runner.submit(harness.job("C", free))
    await harness.runner.wait_idle()

    assert harness.ran == ["A", "C"]  # B nunca rodou
    assert [tag for tag, _ in harness.delivered] == ["C"]
    stats = harness.runner.stats
    assert (stats.cancelled, stats.superseded, stats.completed, stats.stale) == (1, 1, 1, 0)
    assert [result for _, result in harness.observed] == ["cancelled", "completed"]


async def test_only_one_job_runs_at_a_time(harness):
    concurrent, peak = 0, 0
    lock = threading.Lock()

    def run(should_stop):
        nonlocal concurrent, peak
        with lock:
            concurrent += 1
            peak = max(peak, concurrent)
        threading.Event().wait(0.02)
        with lock:
            concurrent -= 1
        return "ok"

    for lap in range(5):
        await harness.runner.submit(Job(lap, run))
        await asyncio.sleep(0.005)
    await harness.runner.wait_idle()

    assert peak == 1


async def test_a_result_that_finishes_after_a_newer_lap_arrived_is_stale_and_not_delivered(harness):
    """A trabalho que ignora o cancelamento termina mesmo assim: como já há uma volta mais nova,
    o resultado antigo não é entregue (o cliente não vê a probabilidade andar para trás)."""
    gate, free = threading.Event(), threading.Event()
    free.set()
    await harness.runner.submit(harness.job("A", gate, honors_cancel=False))
    await harness.started("A")
    await harness.runner.submit(harness.job("B", free))
    gate.set()
    await harness.runner.wait_idle()

    assert [tag for tag, _ in harness.delivered] == ["B"]
    assert harness.runner.stats.stale == 1 and harness.runner.stats.completed == 1
    assert [result for _, result in harness.observed] == ["stale", "completed"]


async def test_a_failing_job_does_not_stop_the_consumer(harness, caplog):
    free = threading.Event()
    free.set()
    with caplog.at_level(logging.ERROR, logger="strategy.forecast_runner"):
        await harness.runner.submit(harness.job("A", free, raises=ValueError("deu ruim")))
        await harness.runner.wait_idle()
        await harness.runner.submit(harness.job("B", free))
        await harness.runner.wait_idle()

    assert [tag for tag, _ in harness.delivered] == ["B"]
    assert harness.runner.stats.failed == 1 and harness.runner.stats.completed == 1
    assert [result for _, result in harness.observed] == ["failed", "completed"]
    assert "simulação de A falhou" in caplog.text


async def test_a_failing_delivery_is_counted_and_the_next_lap_still_runs():
    h = Harness(fail_delivery_for={"A"})
    free = threading.Event()
    free.set()
    try:
        await h.runner.submit(h.job("A", free))
        await h.runner.wait_idle()
        await h.runner.submit(h.job("B", free))
        await h.runner.wait_idle()

        assert [tag for tag, _ in h.delivered] == ["B"]
        assert h.runner.stats.failed == 1 and h.runner.stats.completed == 1
    finally:
        await h.runner.close()


async def test_the_runner_is_idle_only_when_nothing_runs_or_waits(harness):
    gate = threading.Event()
    assert not harness.runner.busy

    await harness.runner.submit(harness.job("A", gate))
    await harness.started("A")
    assert harness.runner.busy

    gate.set()
    await harness.runner.wait_idle()
    assert not harness.runner.busy


async def test_closing_cancels_the_running_job_even_when_no_newer_lap_arrived():
    """O fechamento é quem cancela aqui: não há volta nova, e o trabalho preso só sai se `close`
    mandar parar (senão o desligamento esperaria para sempre)."""
    h = Harness()
    stuck = threading.Event()
    await h.runner.submit(h.job("A", stuck))
    await h.started("A")

    await asyncio.wait_for(h.runner.close(), timeout=5)

    assert h.delivered == [] and h.runner.stats.cancelled == 1
    with pytest.raises(RuntimeError, match="fechado"):
        await h.runner.submit(h.job("C"))


async def test_closing_drops_the_waiting_job_and_delivers_nothing():
    h = Harness()
    stuck = threading.Event()
    await h.runner.submit(h.job("A", stuck))
    await h.started("A")
    await h.runner.submit(h.job("B", stuck))  # esta já cancela a A; a B espera

    await asyncio.wait_for(h.runner.close(), timeout=5)

    assert h.delivered == [] and h.ran == ["A"]  # a B nunca rodou


# ------------------------------------------------------------------ a previsão como trabalho


def quiet_priors() -> SimPriors:
    sc = SafetyCarPrior(0, 20, (), (4,), (1.46,), (1.75,), DEFAULT_COST_RATIO, 15, 100)
    return SimPriors(sc, RetirementPrior(0, 400, ()), CompoundPrior({("MEDIUM", "HARD"): 10}), "t")


def real_job(lap: int = 30):
    rows = chase_field(gap=5.0, world=World(total_laps=60, noise=0.1, seed=2))
    return forecast_job(
        rows,
        year=2024,
        round=6,
        upto_lap=lap,
        total_laps=60,
        sim_priors=quiet_priors(),
        config=SimConfig(n_sims=400),
    )


async def test_a_forecast_job_returns_the_forecast_tagged_with_race_and_lap(harness):
    job = real_job()

    await harness.runner.submit(job)
    await harness.runner.wait_idle()

    ((tag, forecast),) = harness.delivered
    assert tag == (2024, 6, 30) == job.tag
    assert forecast.lap == 30 and forecast.n_sims == 400
    assert sum(d.win for d in forecast.drivers) == pytest.approx(1.0)


def test_a_forecast_job_already_cancelled_stops_before_doing_any_work():
    with pytest.raises(SimulationCancelled):
        real_job().run(lambda: True)


async def test_a_real_forecast_is_cancelled_between_laps_when_asked_to_stop(harness):
    calls = []

    def stop_on_third_check():
        calls.append(1)
        return len(calls) >= 3

    with pytest.raises(SimulationCancelled):
        await asyncio.to_thread(real_job().run, stop_on_third_check)

    assert (
        len(calls) == 3
    )  # parou na terceira conferência: uma por volta simulada (a 1ª é a de entrada)


# ------------------------------------------------------------------ as métricas


@pytest.fixture
def clean_metrics(monkeypatch):
    monkeypatch.setattr(metrics, "_last_delivered_at", None)
    return monkeypatch


def sample(name: str, labels: dict) -> float:
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


def test_a_recorded_simulation_lands_in_the_histogram_and_the_counter(clean_metrics):
    before = sample("f1_strategy_simulation_duration_seconds_count", {"result": "completed"})
    counted = sample("f1_strategy_simulations_total", {"result": "completed"})

    metrics.record(0.3, "completed")

    assert (
        sample("f1_strategy_simulation_duration_seconds_count", {"result": "completed"})
        == before + 1
    )
    assert sample("f1_strategy_simulations_total", {"result": "completed"}) == counted + 1
    # 0,3 s cai no bucket de 0,5 s e não no de 0,25 s
    assert (
        sample(
            "f1_strategy_simulation_duration_seconds_bucket", {"result": "completed", "le": "0.5"}
        )
        >= 1
    )


def test_every_result_series_exists_from_the_start_and_an_unknown_one_is_refused():
    for result in metrics.RESULTS:
        assert (
            metrics.REGISTRY.get_sample_value("f1_strategy_simulations_total", {"result": result})
            is not None
        )
    with pytest.raises(ValueError, match="resultado desconhecido"):
        metrics.record(1.0, "pelo-piloto-VER")


def test_the_only_label_is_the_result_never_a_driver_race_or_lap():
    labels = {
        key
        for family in metrics.REGISTRY.collect()
        if family.name.startswith("f1_strategy")
        for s in family.samples
        for key in s.labels
    }

    assert labels <= {"result", "le"}  # `le` é o bucket do histograma


def test_the_age_of_the_last_result_is_nan_before_the_first_and_grows_after(clean_metrics):
    now = [1000.0]
    clean_metrics.setattr(metrics, "_clock", lambda: now[0])
    assert math.isnan(metrics.last_result_age())
    assert math.isnan(sample_gauge())

    metrics.record(0.1, "completed")
    now[0] += 42.0

    assert metrics.last_result_age() == pytest.approx(42.0)
    assert sample_gauge() == pytest.approx(42.0)


def test_only_a_delivered_result_resets_the_age(clean_metrics):
    now = [10.0]
    clean_metrics.setattr(metrics, "_clock", lambda: now[0])
    metrics.record(0.1, "completed")
    now[0] += 30.0

    for other in ("cancelled", "stale", "failed"):
        metrics.record(0.1, other)

    assert metrics.last_result_age() == pytest.approx(30.0)


def sample_gauge() -> float:
    value = metrics.REGISTRY.get_sample_value("f1_strategy_last_result_age_seconds")
    assert value is not None
    return value
