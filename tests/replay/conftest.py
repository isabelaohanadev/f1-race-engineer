"""Fixtures do replay: um banco SQLite semeado com uma corrida pequena e determinística.

O `engine` (SQLite em memória) vem do conftest raiz. O que SQLite não cobre (o ON CONFLICT, a
hypertable, o plano de execução no índice real) fica em tests/integration/test_replay_timescale.py.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Driver, Lap, PositionSample, Race, RaceEntry, Team

BASE = datetime(2024, 11, 3, 15, 50, tzinfo=UTC)
VER, NOR, ALB = 1, 2, 3  # ids dos pilotos semeados


def at(ms: int) -> datetime:
    """Instante `ms` milissegundos depois do início da corrida de teste."""
    return BASE + timedelta(milliseconds=ms)


PositionRow = tuple[int, datetime, int, int]  # (driver_id, time, x, y)
LapRow = tuple[int, int, datetime | None, int | None, str | None, int | None]
# (driver_id, lap_number, completed_at, lap_time_ms, compound, position)


def seed_race(
    engine: Engine,
    *,
    positions: Sequence[PositionRow] = (),
    laps: Sequence[LapRow] = (),
    total_laps: int | None = 3,
    year: int = 2024,
    round_number: int = 21,
    race_id: int = 1,
    lap_extras: dict[tuple[int, int], dict] | None = None,  # (driver_id, volta) -> colunas a mais
) -> None:
    with Session(engine) as s:
        # Pilotos/equipes só na 1ª corrida semeada (o teste pode semear mais de uma).
        if s.get(Team, 1) is None:
            s.add_all(
                [
                    Team(id=1, name="Red Bull Racing"),
                    Driver(id=VER, code="VER", full_name="Max Verstappen"),
                    Driver(id=NOR, code="NOR", full_name="Lando Norris"),
                    Driver(id=ALB, code="ALB", full_name="Alexander Albon"),
                ]
            )
        s.add(Race(id=race_id, year=year, round=round_number, name="GP", total_laps=total_laps))
        s.flush()
        s.add_all(
            RaceEntry(race_id=race_id, driver_id=d, team_id=1, driver_number=d)
            for d in (VER, NOR, ALB)
        )
        s.flush()
        s.add_all(
            PositionSample(race_id=race_id, driver_id=d, time=t, x=x, y=y)
            for d, t, x, y in positions
        )
        s.add_all(
            Lap(
                race_id=race_id,
                driver_id=d,
                lap_number=n,
                completed_at=done,
                lap_time_ms=ms,
                compound=compound,
                position=pos,
                **(lap_extras or {}).get((d, n), {}),
            )
            for d, n, done, ms, compound, pos in laps
        )
        s.commit()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def seed(engine: Engine) -> Callable[..., None]:
    return lambda **kwargs: seed_race(engine, **kwargs)
