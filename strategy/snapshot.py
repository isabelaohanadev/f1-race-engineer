"""O estado da corrida ATÉ O INSTANTE t: o que um sistema ao vivo saberia, e nada mais.

POR QUE EXISTE. O primeiro desenho tomava "o estado na volta N" como a volta N de CADA piloto. Só
que o líder completa a volta N antes de todos: o segundo a completa alguns segundos depois e um
retardatário muito depois (um carro uma volta atrás completa a "volta N" quando o líder já está na
N+1). Usar as voltas N de todos dá ao modelo, em t, informação de depois de t: o tempo da volta N de
quem vem atrás, se parou nela, com que composto. O backtest seria otimista e, pior, ninguém saberia
quanto.

A REGRA. `state_until(rows, t_ms)` devolve só as voltas cujo relógio da sessão (`session_time_ms`, o
instante em que o piloto CRUZOU a linha ao completá-las) é menor ou igual a `t_ms`. De cada piloto
fica a sua última volta completa: o líder, na volta N; os demais, em geral, na N-1 (um retardatário,
na
N-2 ou antes). O que acontece na volta seguinte deles (tempo, parada, composto) é futuro e não está
no
estado.

QUEM SAIU DA CORRIDA. Um piloto que não cruza a linha há mais de `max_silence_laps` voltas típicas
dele (a mediana das voltas dele) saiu: não entra no estado. Um carro lapado ainda cruza a linha uma
vez
por volta dele, então não é confundido com um abandono. É uma heurística (uma parada de bandeira
vermelha o excluiria), e a corrida de bandeira vermelha já não entra no backtest principal.

FUNÇÃO PURA. Não lê banco nem relógio: recebe as voltas e o instante. A etapa de consumo ao vivo a
reusa com o instante do evento `lap_completed` do líder (ou do piloto, para o undercut), sobre as
voltas que já chegaram do stream.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from strategy.laps import LapRow

MAX_SILENCE_LAPS = 2.0


@dataclass(frozen=True, slots=True)
class CarState:
    """Um piloto na SUA última volta completa."""

    driver: str
    clock_s: float  # relógio da sessão ao fim dessa volta
    compound: str | None
    age: int | None
    stint: int | None
    status: str | None
    pit_in: bool
    pit_out: bool
    laps_done: int = 0  # a volta que ele completou (no estado por instante, varia entre pilotos)


@dataclass(frozen=True, slots=True)
class RaceState:
    t_ms: int
    rows: tuple[LapRow, ...]  # as voltas completadas até t_ms
    cars: dict[str, CarState]  # a última volta de cada piloto que ainda está na corrida
    leader_lap: int  # a volta em que o líder está, completa

    @property
    def laps_down(self) -> dict[str, int]:
        return {d: self.leader_lap - c.laps_done for d, c in self.cars.items()}


def car_state(row: LapRow) -> CarState:
    assert row.session_time_ms is not None
    return CarState(
        row.driver,
        row.session_time_ms / 1000,
        row.compound,
        row.tyre_life,
        row.stint,
        row.track_status,
        bool(row.pit_in),
        bool(row.pit_out),
        row.lap_number,
    )


def car_states(rows: Sequence[LapRow], upto_lap: int) -> dict[str, CarState]:
    """O estado POR CONTAGEM DE VOLTAS: cada piloto na volta `upto_lap` dele. É o desenho antigo e
    dá
    informação de depois do instante em que o líder completa a volta (ver o docstring do módulo);
    fica para os testes de mecânica e para o diagnóstico do vazamento no backtest."""
    return {
        r.driver: car_state(r)
        for r in rows
        if r.lap_number == upto_lap and r.session_time_ms is not None
    }


def crossing_ms(rows: Sequence[LapRow], lap: int) -> int | None:
    """O instante em que o primeiro carro completou a volta `lap` (o líder cruza a linha)."""
    times = [
        r.session_time_ms for r in rows if r.lap_number == lap and r.session_time_ms is not None
    ]
    return min(times) if times else None


def driver_crossing_ms(rows: Sequence[LapRow], driver: str, lap: int) -> int | None:
    for r in rows:
        if r.driver == driver and r.lap_number == lap:
            return r.session_time_ms
    return None


def rows_until(rows: Sequence[LapRow], t_ms: int) -> list[LapRow]:
    """As voltas completadas até o instante `t_ms` (inclusive). Voltas sem relógio não entram: não
    há
    como saber quando foram completadas."""
    return [r for r in rows if r.session_time_ms is not None and r.session_time_ms <= t_ms]


def _typical_lap_s(rows: Sequence[LapRow]) -> tuple[dict[str, float], float | None]:
    per: dict[str, list[float]] = {}
    for r in rows:
        if r.lap_time_ms is not None:
            per.setdefault(r.driver, []).append(r.lap_time_ms / 1000)
    typical = {d: float(median(v)) for d, v in per.items()}
    field = float(median(typical.values())) if typical else None
    return typical, field


def state_until(
    rows: Sequence[LapRow], t_ms: int, *, max_silence_laps: float = MAX_SILENCE_LAPS
) -> RaceState:
    """O estado da corrida no instante `t_ms`, só com o que já tinha acontecido."""
    visible = rows_until(rows, t_ms)
    typical, field_lap = _typical_lap_s(visible)
    last: dict[str, LapRow] = {}
    for r in visible:
        current = last.get(r.driver)
        if current is None or r.lap_number > current.lap_number:
            last[r.driver] = r
    cars: dict[str, CarState] = {}
    for driver, row in last.items():
        lap_s = typical.get(driver, field_lap)
        silence = (t_ms - row.session_time_ms) / 1000  # type: ignore[operator]
        if lap_s is not None and silence > max_silence_laps * lap_s:
            continue  # saiu da corrida
        cars[driver] = car_state(row)
    leader_lap = max((c.laps_done for c in cars.values()), default=0)
    return RaceState(t_ms, tuple(visible), cars, leader_lap)


def state_at_leader_lap(
    rows: Sequence[LapRow], lap: int, *, max_silence_laps: float = MAX_SILENCE_LAPS
) -> RaceState | None:
    """O estado no instante em que o líder completa a volta `lap` (None se ninguém a completou)."""
    t = crossing_ms(rows, lap)
    return None if t is None else state_until(rows, t, max_silence_laps=max_silence_laps)


def state_at_driver_lap(
    rows: Sequence[LapRow],
    driver: str,
    lap: int,
    *,
    max_silence_laps: float = MAX_SILENCE_LAPS,
) -> RaceState | None:
    """O estado no instante em que `driver` completa a volta `lap`: o que se sabia quando ele (o
    atacante, no undercut) decide parar na volta seguinte."""
    t = driver_crossing_ms(rows, driver, lap)
    return None if t is None else state_until(rows, t, max_silence_laps=max_silence_laps)
