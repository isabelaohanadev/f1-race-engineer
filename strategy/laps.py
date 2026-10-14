"""As voltas de uma corrida como a estratégia as lê: linhas leves e imutáveis, sem ORM.

Inspeção de qualidade, limpeza, combustível e o modelo de pneu consomem a MESMA estrutura, então
um campo novo entra num lugar só. O resto do pacote `strategy` nunca fala com o banco: recebe
`LapRow`, o que mantém os cálculos testáveis com voltas sintéticas.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Driver, Lap, Race, RaceEntry

DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})


@dataclass(frozen=True, slots=True)
class LapRow:
    """Uma volta, com só o que a estratégia lê."""

    driver: str
    lap_number: int
    lap_time_ms: int | None
    compound: str | None
    tyre_life: int | None
    stint: int | None
    position: int | None
    session_time_ms: int | None
    pit_in: bool | None
    pit_out: bool | None
    track_status: str | None


@dataclass(frozen=True, slots=True)
class EntryRow:
    """O resultado OFICIAL de um piloto (race_entries): o que a FIA classificou, com penalidades e
    desqualificações. Não é a ordem na pista (ver strategy/results.py)."""

    driver: str
    finish_position: int | None
    status: str | None = None


def load_race_entries(session: Session, race_id: int) -> list[EntryRow]:
    rows = session.execute(
        select(Driver.code, RaceEntry.finish_position, RaceEntry.status)
        .join(Driver, Driver.id == RaceEntry.driver_id)
        .where(RaceEntry.race_id == race_id)
        .order_by(Driver.code)
    ).all()
    return [EntryRow(*row) for row in rows]


def load_race_laps(session: Session, race_id: int) -> list[LapRow]:
    rows = session.execute(
        select(
            Driver.code,
            Lap.lap_number,
            Lap.lap_time_ms,
            Lap.compound,
            Lap.tyre_life,
            Lap.stint,
            Lap.position,
            Lap.session_time_ms,
            Lap.pit_in,
            Lap.pit_out,
            Lap.track_status,
        )
        .join(Driver, Driver.id == Lap.driver_id)
        .where(Lap.race_id == race_id)
        .order_by(Driver.code, Lap.lap_number)
    ).all()
    return [LapRow(*row) for row in rows]


def list_races(session: Session, year: int | None = None) -> Sequence:
    """(id, year, round, name, total_laps) das corridas, em ordem de calendário."""
    stmt = select(Race.id, Race.year, Race.round, Race.name, Race.total_laps)
    if year is not None:
        stmt = stmt.where(Race.year == year)
    return session.execute(stmt.order_by(Race.year, Race.round)).all()


def race_label(year: int, round_number: int, name: str) -> str:
    """'2024 R10 Spanish': o rótulo curto das tabelas."""
    return f"{year} R{round_number:02d} {name.removesuffix(' Grand Prix')}"
