"""Fixtures dos testes UNITÁRIOS da API.

Estratégia: SQLite em memória + dados semeados à mão (mockados), sem nenhuma infraestrutura.
Por que SQLite e não um Session falso: os endpoints rodam SQL de verdade (joins, filtros,
ordenação), então testamos a consulta e não a implementação de um mock. Os models só usam tipos
portáveis, então `create_all` funciona. O que NÃO é portável (ON CONFLICT do Postgres, hypertable,
timestamptz) é coberto pelos testes de integração, contra o Timescale real (tests/integration).
"""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.deps import get_db
from app.main import app
from db.models import Base, Driver, Lap, Race, RaceEntry, Team

T0 = datetime(2024, 11, 3, 15, 50, tzinfo=UTC)


@pytest.fixture
def engine() -> Iterator[Engine]:
    # StaticPool + check_same_thread=False: o TestClient executa endpoints `def` em outra thread,
    # e um SQLite em memória só existe dentro de UMA conexão; ela precisa ser compartilhada.
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def seeded(engine: Engine) -> Engine:
    """Banco com: GP do Brasil 2024 (VER 3 voltas, NOR 2, ALB não largou) e uma 2ª corrida
    (2023/1) em que o VER também correu, para provar que o filtro não vaza entre corridas.
    HAM existe na tabela de pilotos, mas NÃO correu em 2024/21."""
    with Session(engine) as s:
        s.add_all(
            [
                Team(id=1, name="Red Bull Racing", color_hex="3671C6"),
                Team(id=2, name="McLaren", color_hex="FF8000"),
                Team(id=3, name="Williams", color_hex="64C4FF"),
                Driver(id=1, code="VER", full_name="Max Verstappen"),
                Driver(id=2, code="NOR", full_name="Lando Norris"),
                Driver(id=3, code="ALB", full_name="Alexander Albon"),
                Driver(id=4, code="HAM", full_name="Lewis Hamilton"),
                Race(
                    id=1,
                    year=2024,
                    round=21,
                    name="São Paulo Grand Prix",
                    location="São Paulo",
                    country="Brazil",
                    starts_at=T0,
                    total_laps=69,
                ),
                Race(id=2, year=2023, round=1, name="Bahrain Grand Prix", total_laps=57),
            ]
        )
        s.flush()
        s.add_all(
            [
                RaceEntry(race_id=1, driver_id=1, team_id=1, driver_number=1, finish_position=1),
                RaceEntry(race_id=1, driver_id=2, team_id=2, driver_number=4, finish_position=6),
                RaceEntry(
                    race_id=1, driver_id=3, team_id=3, driver_number=23, status="Did not start"
                ),
                RaceEntry(race_id=2, driver_id=1, team_id=1, driver_number=1),
            ]
        )
        s.flush()
        s.add_all(
            [
                # lap_time_ms nulo: volta sem tempo válido (largada com bandeira vermelha, p. ex.)
                Lap(
                    race_id=1,
                    driver_id=1,
                    lap_number=1,
                    lap_time_ms=None,
                    compound="INTERMEDIATE",
                    tyre_life=1,
                    stint=1,
                    position=11,
                    completed_at=T0,
                ),
                Lap(
                    race_id=1,
                    driver_id=1,
                    lap_number=2,
                    lap_time_ms=87134,
                    compound="INTERMEDIATE",
                    tyre_life=2,
                    stint=1,
                    position=10,
                    completed_at=T0,
                ),
                Lap(
                    race_id=1,
                    driver_id=1,
                    lap_number=3,
                    lap_time_ms=86240,
                    compound="WET",
                    tyre_life=1,
                    stint=2,
                    position=10,
                    completed_at=T0,
                ),
                Lap(
                    race_id=1,
                    driver_id=2,
                    lap_number=1,
                    lap_time_ms=95000,
                    compound="INTERMEDIATE",
                    tyre_life=1,
                    stint=1,
                    position=1,
                    completed_at=T0,
                ),
                Lap(
                    race_id=1,
                    driver_id=2,
                    lap_number=2,
                    lap_time_ms=88000,
                    compound="INTERMEDIATE",
                    tyre_life=2,
                    stint=1,
                    position=1,
                    completed_at=T0,
                ),
                # Volta do VER na OUTRA corrida: não pode aparecer em 2024/21.
                Lap(
                    race_id=2,
                    driver_id=1,
                    lap_number=1,
                    lap_time_ms=99999,
                    compound="SOFT",
                    tyre_life=1,
                    stint=1,
                    position=1,
                    completed_at=T0,
                ),
            ]
        )
        s.commit()
    return engine


def _client_for(engine: Engine) -> Iterator[TestClient]:
    factory = sessionmaker(engine, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()  # não vaza o override para outros testes


@pytest.fixture
def client(seeded: Engine) -> Iterator[TestClient]:
    yield from _client_for(seeded)


@pytest.fixture
def empty_client(engine: Engine) -> Iterator[TestClient]:
    yield from _client_for(engine)
