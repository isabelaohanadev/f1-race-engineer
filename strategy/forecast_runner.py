"""Executor "só a última": roda a simulação FORA do event loop e descarta a antiga, sem fila.

Por que existe. A simulação de uma volta leva de 0,05 s a ~1,3 s de CPU de numpy (medido, 5.000
trajetórias, 20 carros). Dentro do event loop ela travaria tudo o que roda nele (a leitura do Redis,
a
publicação, as métricas); por isso ela roda numa thread (o numpy solta o GIL nas contas grandes) do
processo do consumidor de estratégia, que já é um processo à parte da API.

A POLÍTICA, a pedido: "se uma simulação ainda estiver rodando quando chegar a volta seguinte,
descarte
a antiga e não acumule fila".
* No máximo UMA simulação roda e UMA espera. Chegou uma volta nova: a em curso recebe o sinal de
  cancelamento (confere a cada volta simulada, em milissegundos) e a que esperava é substituída.
* Um resultado que termina depois de uma volta mais nova ter chegado é `stale` e NÃO é entregue: o
  mais novo vem logo atrás, e entregar o antigo faria o cliente ver a probabilidade andar para trás.
* Uma simulação que falha não derruba o consumidor: conta `failed` e a próxima volta segue.
* Sob sobrecarga (cada simulação mais longa que o intervalo entre voltas) a última ainda roda
inteira
  se nenhuma outra chegar enquanto ela roda: o sistema entrega o que consegue em vez de nada.

Tudo o que mexe no estado (`submit`, o início e o fim de uma simulação) roda no event loop, então
não
há trava; só o sinal de cancelamento (`threading.Event`) atravessa para a thread.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Hashable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field

from strategy import metrics
from strategy.fuel import FuelVariant
from strategy.history import SimPriors
from strategy.laps import LapRow
from strategy.montecarlo import RaceForecast, SimConfig, SimulationCancelled, prepare, simulate
from strategy.priors import DEFAULT_PRIORS, Priors

log = logging.getLogger(__name__)

ShouldStop = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class Job[T]:
    tag: Hashable  # (ano, etapa, volta): quem recebe o resultado sabe de que volta ele é
    run: Callable[[ShouldStop], T]


@dataclass(slots=True)
class Stats:
    completed: int = 0
    cancelled: int = 0
    stale: int = 0
    failed: int = 0
    superseded: int = 0  # esperavam e foram substituídas por uma volta mais nova: nunca rodaram


@dataclass(slots=True)
class _Active:
    job: Job
    cancel: threading.Event = field(default_factory=threading.Event)


class LatestOnlyRunner[T]:
    def __init__(
        self,
        on_result: Callable[[Hashable, T], Awaitable[None]],
        *,
        executor: Executor | None = None,
        observe: Callable[[float, str], None] = metrics.record,
    ) -> None:
        self._on_result = on_result
        self._observe = observe
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="strategy-sim"
        )
        self._running: _Active | None = None
        self._pending: Job | None = None
        self._task: asyncio.Task | None = None
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self.stats = Stats()

    @property
    def busy(self) -> bool:
        return self._running is not None or self._pending is not None

    async def submit(self, job: Job[T]) -> None:
        """Entrega uma volta nova. Não espera a simulação: só troca o que estava esperando e manda a
        em curso parar."""
        if self._closed:
            raise RuntimeError("o executor foi fechado")
        if self._running is not None:
            self._running.cancel.set()
        if self._pending is not None:
            self.stats.superseded += 1
        self._pending = job
        self._idle.clear()
        if self._running is None:
            self._task = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        """Roda o que espera, uma por vez, até não sobrar nada."""
        while self._pending is not None and not self._closed:
            job, self._pending = self._pending, None
            self._running = _Active(job)
            try:
                await self._run_one(self._running)
            finally:
                self._running = None
        if self._pending is None:
            self._idle.set()

    async def _run_one(self, active: _Active) -> None:
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            result = await loop.run_in_executor(
                self._executor, active.job.run, active.cancel.is_set
            )
        except SimulationCancelled:
            self.stats.cancelled += 1
            self._observe(time.perf_counter() - started, "cancelled")
            return
        except Exception:
            log.exception("a simulação de %s falhou", active.job.tag)
            self.stats.failed += 1
            self._observe(time.perf_counter() - started, "failed")
            return
        duration = time.perf_counter() - started
        if active.cancel.is_set() or self._pending is not None or self._closed:
            self.stats.stale += 1  # terminou, mas há uma volta mais nova: não se entrega a antiga
            self._observe(duration, "stale")
            return
        try:
            await self._on_result(active.job.tag, result)
        except Exception:
            log.exception("a entrega do resultado de %s falhou", active.job.tag)
            self.stats.failed += 1
            self._observe(duration, "failed")
            return
        self.stats.completed += 1
        self._observe(duration, "completed")

    async def wait_idle(self) -> None:
        """Espera não haver nada rodando nem esperando (testes e desligamento ordenado)."""
        await self._idle.wait()

    async def close(self) -> None:
        """Descarta o que espera, manda o que roda parar e espera ele sair; depois disso `submit`
        levanta. Nada é entregue depois do fechamento."""
        self._closed = True
        self._pending = None
        if self._running is not None:
            self._running.cancel.set()
        if self._task is not None and not self._task.done():
            await self._task
        self._idle.set()
        if self._owns_executor:
            self._executor.shutdown(wait=True)


def forecast_job(
    rows: Sequence[LapRow],
    *,
    year: int,
    round: int,
    upto_lap: int,
    total_laps: int,
    sim_priors: SimPriors,
    priors: Priors = DEFAULT_PRIORS,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    config: SimConfig | None = None,
) -> Job[RaceForecast]:
    """A previsão da volta `upto_lap` como um trabalho do executor. Tudo (ajustar o modelo e
    simular) acontece na thread; a etiqueta (ano, etapa, volta) volta junto com o resultado."""

    def run(should_stop: ShouldStop) -> RaceForecast:
        if should_stop():
            raise SimulationCancelled
        inp = prepare(
            rows,
            year=year,
            round=round,
            upto_lap=upto_lap,
            total_laps=total_laps,
            sim_priors=sim_priors,
            priors=priors,
            variant=variant,
        )
        return simulate(inp, config, should_stop=should_stop)

    return Job((year, round, upto_lap), run)
