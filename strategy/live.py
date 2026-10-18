"""O estado ao vivo de UMA corrida e o cálculo de cada volta: a parte PURA do consumidor.

Aqui não há Redis, asyncio nem relógio de parede: o "agora" entra como argumento. É o que deixa
testar com números o que mais importa do consumidor: dedupe, janela de espera, voltas atrasadas e
reprodutibilidade (a mesma corrida, com eventos embaralhados, dá o mesmo resultado).

O QUE É "O LÍDER COMPLETA A VOLTA N"
O consumidor dispara o cálculo da volta N na chegada da PRIMEIRA entrada de volta N (a do líder, no
stream em ordem), mas só o executa depois de uma JANELA de espera (`wait_s`, configurável). A janela
serve a uma coisa: voltas que pertencem ao instante do líder (as dos carros de trás, N-1 ou antes,
que cruzaram a linha ANTES dele) e chegaram atrasadas, por um produtor fora de ordem ou uma
reentrega. As voltas que acontecem DEPOIS desse instante não importam: `state_until` as deixa de
fora de propósito (é o que o sistema ao vivo saberia). No stream do replay, que já sai em ordem de
tempo, a janela não muda o resultado; ela protege o dia em que a ordem não for garantida.

  * Dedupe por (piloto, volta): a primeira entrada vence; a repetição é contada e ignorada (o
    consumer group entrega "pelo menos uma vez").
  * Volta atrasada: cruzou a linha ANTES de um instante cujo resultado já foi calculado. Não reabre
    o cálculo (o resultado de uma volta é calculado uma vez), mas entra no estado e vale para as
    voltas seguintes. Fica contada (`late_laps`) e vai no resultado.
  * O disparo só AVANÇA: a volta N dispara se N é maior que a maior já disparada. Uma volta velha de
    um retardatário (N-5 chegando quando o líder já está na N) nunca dispara um resultado velho.

A MESMA FUNÇÃO DO BACKTEST. `compute` chama `prepare(snapshot="tempo")` e
`build_context(as_of_ms=)`, as mesmas do backtest, que usam `strategy.snapshot.state_until`. As
voltas entram ordenadas por (piloto, volta), como `load_race_laps` as devolve, então o resultado é
numericamente o do backtest (`tests/strategy/test_live.py` compara os dois).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from replay.events import LapCompletedEvent, RaceStatusEvent, StreamEntry
from strategy.forecast_eval import MIN_DRIVERS, current_leader
from strategy.forecast_runner import Job, ShouldStop
from strategy.fuel import FuelVariant
from strategy.history import SimPriors
from strategy.laps import LapRow
from strategy.montecarlo import (
    RaceForecast,
    SimConfig,
    SimInput,
    SimulationCancelled,
    prepare,
    simulate,
)
from strategy.payload import track_warnings
from strategy.priors import Priors
from strategy.protocol import FROZEN, frozen_sim_config, frozen_undercut_config
from strategy.snapshot import crossing_ms
from strategy.undercut import (
    LapContext,
    UndercutAssessment,
    UndercutConfig,
    assess_all,
    build_context,
)

FIRST_LAP = FROZEN.first_lap  # o backtest começa na volta 5: antes não há dados para o modelo
DEFAULT_WAIT_S = 0.3


class Outcome(StrEnum):
    """O que aconteceu com uma entrada (os valores são os rótulos da métrica: poucos e fixos)."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    LATE = "late"  # entrou no estado, mas depois de um resultado do instante dela
    STALE_EPOCH = "stale_epoch"  # de um replay antigo
    STATUS = "status"  # início ou fim do replay
    IGNORED = "ignored"  # tipo que a estratégia não lê


@dataclass(frozen=True, slots=True)
class Applied:
    outcome: Outcome
    new_epoch: bool = False  # esta entrada abriu um replay novo: o estado foi zerado
    scheduled_lap: int | None = None  # o cálculo desta volta foi agendado


@dataclass(frozen=True, slots=True)
class Trigger:
    """Um cálculo pronto para rodar: as voltas que se sabem agora, congeladas."""

    year: int
    round: int
    epoch: int
    lap: int
    total_laps: int
    rows: tuple[LapRow, ...]  # ordenadas por (piloto, volta), como o backtest as recebe
    cars_known: int  # pilotos que apareceram no stream até agora
    late_laps: int  # voltas que chegaram depois de um resultado do instante delas
    duplicates: int
    wait_s: float
    leader_t_ms: int  # o instante em que o líder completou a volta


def lap_row(event: LapCompletedEvent) -> LapRow:
    return LapRow(
        event.driver,
        event.lap,
        event.lap_time_ms,
        event.compound,
        event.tyre_life,
        event.stint,
        event.position,
        event.session_time_ms,
        event.pit_in,
        event.pit_out,
        event.track_status,
    )


@dataclass(slots=True)
class RaceTracker:
    """O que o consumidor sabe de uma corrida, e quando calcular cada volta."""

    year: int
    round: int
    wait_s: float = DEFAULT_WAIT_S
    first_lap: int = FIRST_LAP
    epoch: int = 0
    total_laps: int | None = None
    finished: bool = False
    rows: dict[tuple[str, int], LapRow] = field(default_factory=dict)
    duplicates: int = 0
    late_laps: int = 0
    last_trigger_lap: int = 0
    _pending: dict[int, float] = field(default_factory=dict)  # volta -> quando disparar
    _latest_fired_t: int | None = None  # o instante do resultado mais novo já disparado

    def _reset(self, epoch: int) -> None:
        self.epoch = epoch
        self.total_laps = None
        self.finished = False
        self.rows.clear()
        self.duplicates = self.late_laps = 0
        self.last_trigger_lap = 0
        self._pending.clear()
        self._latest_fired_t = None

    # ------------------------------------------------------------ entradas

    def apply(self, entry: StreamEntry, now: float, *, schedule: bool = True) -> Applied:
        """Aplica uma entrada do stream de voltas. `schedule=False` na reconstrução (a volta de um
        estado antigo não agenda cálculo nenhum)."""
        if entry.epoch < self.epoch:
            return Applied(Outcome.STALE_EPOCH)
        new_epoch = entry.epoch > self.epoch
        if new_epoch:
            self._reset(entry.epoch)
        event = entry.event
        if isinstance(event, RaceStatusEvent):
            if event.state == "started" and event.total_laps is not None:
                self.total_laps = event.total_laps
            elif event.state == "finished":
                self.finished = True
            return Applied(Outcome.STATUS, new_epoch)
        if not isinstance(event, LapCompletedEvent):
            return Applied(Outcome.IGNORED, new_epoch)

        key = (event.driver, event.lap)
        if key in self.rows:
            self.duplicates += 1
            return Applied(Outcome.DUPLICATE, new_epoch)
        row = lap_row(event)
        self.rows[key] = row
        late = (
            row.session_time_ms is not None
            and self._latest_fired_t is not None
            and row.session_time_ms <= self._latest_fired_t
        )
        if late:
            self.late_laps += 1
        scheduled = None
        if schedule and event.lap > self.last_trigger_lap and event.lap >= self.first_lap:
            self.last_trigger_lap = event.lap
            self._pending[event.lap] = now + self.wait_s
            scheduled = event.lap
        return Applied(Outcome.LATE if late else Outcome.APPLIED, new_epoch, scheduled)

    # ------------------------------------------------------------ disparos

    @property
    def next_deadline(self) -> float | None:
        """Quando a próxima janela fecha (para o consumidor saber por quanto tempo dormir)."""
        return min(self._pending.values(), default=None)

    def _trigger(self, lap: int) -> Trigger | None:
        if self.total_laps is None or not self.first_lap <= lap < self.total_laps:
            return None  # sem o total não há o que simular; a última volta já é o fim da corrida
        rows = tuple(sorted(self.rows.values(), key=lambda r: (r.driver, r.lap_number)))
        t = crossing_ms(rows, lap)
        if t is None:
            return None
        self._latest_fired_t = t if self._latest_fired_t is None else max(self._latest_fired_t, t)
        return Trigger(
            self.year,
            self.round,
            self.epoch,
            lap,
            self.total_laps,
            rows,
            len({r.driver for r in rows}),
            self.late_laps,
            self.duplicates,
            self.wait_s,
            t,
        )

    def due(self, now: float) -> list[Trigger]:
        """Os cálculos cuja janela fechou, em ordem de volta. Sem o total de voltas a volta espera
        (o `started` pode chegar depois da primeira volta, num produtor fora de ordem)."""
        ready = sorted(lap for lap, deadline in self._pending.items() if deadline <= now)
        triggers = []
        for lap in ready:
            if self.total_laps is None:
                continue  # volta a tentar no próximo ciclo
            del self._pending[lap]
            trigger = self._trigger(lap)
            if trigger is not None:
                triggers.append(trigger)
        return triggers

    def latest_lap(self) -> int | None:
        """A maior volta já calculável (a da retomada: o estado foi reconstruído do stream)."""
        if self.total_laps is None:
            return None
        laps = {r.lap_number for r in self.rows.values() if r.session_time_ms is not None}
        eligible = [n for n in laps if self.first_lap <= n < self.total_laps]
        return max(eligible, default=None)

    def resume(self) -> Trigger | None:
        """Depois de reconstruir o estado: o cálculo da última volta, SEM janela (o que tinha de
        chegar já chegou), e o disparo passa a avançar dali."""
        lap = self.latest_lap()
        if lap is None:
            return None
        self.last_trigger_lap = max(self.last_trigger_lap, lap)
        return self._trigger(lap)


# ---------------------------------------------------------------- o cálculo (roda numa thread)


@dataclass(frozen=True, slots=True)
class Constants:
    """As constantes que vêm dos dados, sem a corrida reproduzida (ver strategy/constants.py)."""

    priors: Priors
    sim_priors: SimPriors


@dataclass(frozen=True, slots=True)
class Computation:
    trigger: Trigger
    forecast: RaceForecast
    sim_input: SimInput
    assessments: list[UndercutAssessment]
    leader: str | None
    warnings: list[dict[str, str]]


def leader_lap_assessments(
    rows: Sequence[LapRow],
    *,
    lap: int,
    total_laps: int,
    t_ms: int,
    priors: Priors,
    config: UndercutConfig | None = None,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
) -> tuple[LapContext, list[UndercutAssessment]]:
    """Os alertas de undercut de TODOS os pilotos no instante em que o líder completa a volta `lap`:
    um contexto só (`build_context(as_of_ms=)`) e cada piloto contra o carro à frente dele no estado
    daquele instante. É o que o consumidor entrega, e o que `strategy.undercut_live` avalia: uma
    função só, para a avaliação nunca divergir do que roda ao vivo."""
    context = build_context(
        rows,
        upto_lap=lap,
        total_laps=total_laps,
        variant=variant,
        priors=priors,
        config=config or frozen_undercut_config(),
        as_of_ms=t_ms,
    )
    return context, assess_all(context)


def compute(
    trigger: Trigger,
    constants: Constants,
    *,
    sim_config: SimConfig | None = None,
    undercut_config: UndercutConfig | None = None,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    should_stop: ShouldStop = lambda: False,
) -> Computation | None:
    """A previsão e os alertas da volta do gatilho, com a MESMA função do backtest. Devolve None
    quando há pilotos de menos para prever um pódio (o backtest também os pula)."""
    if should_stop():
        raise SimulationCancelled
    inp = prepare(
        trigger.rows,
        year=trigger.year,
        round=trigger.round,
        upto_lap=trigger.lap,
        total_laps=trigger.total_laps,
        sim_priors=constants.sim_priors,
        priors=constants.priors,
        variant=variant,
        snapshot="tempo",
    )
    if len(inp.drivers) < MIN_DRIVERS:
        return None
    forecast = simulate(inp, sim_config or frozen_sim_config(), should_stop=should_stop)
    if should_stop():
        raise SimulationCancelled
    _, assessments = leader_lap_assessments(
        trigger.rows,
        lap=trigger.lap,
        total_laps=trigger.total_laps,
        t_ms=inp.t_ms,  # type: ignore[arg-type]
        priors=constants.priors,
        config=undercut_config,
        variant=variant,
    )
    visible = [
        r for r in trigger.rows if r.session_time_ms is not None and r.session_time_ms <= inp.t_ms
    ]  # type: ignore[operator]
    return Computation(
        trigger,
        forecast,
        inp,
        assessments,
        current_leader(visible, trigger.lap),
        track_warnings(visible, trigger.lap, inp.sc_elapsed),
    )


def computation_job(trigger: Trigger, constants: Constants, **options) -> Job[Computation | None]:
    """O cálculo como trabalho do executor "só a última" (strategy/forecast_runner.py). A etiqueta
    inclui o epoch: um resultado de um replay que já acabou nunca se confunde com o do novo."""

    def run(should_stop: ShouldStop) -> Computation | None:
        return compute(trigger, constants, should_stop=should_stop, **options)

    return Job((trigger.year, trigger.round, trigger.epoch, trigger.lap), run)
