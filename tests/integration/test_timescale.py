"""Testes contra o TimescaleDB real. Cobrem o que SQLite não consegue: a migration de verdade, a
hypertable, o ON CONFLICT do Postgres e o formato real de timestamptz."""

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.deps import get_db
from app.main import app
from ingestion.dto import EntryInfo, LapInfo, PositionTrack, RaceData, RaceInfo
from ingestion.writer import write_race
from tests.integration.conftest import _alembic

T0 = datetime(2024, 11, 3, 15, 50, tzinfo=UTC)
SESSION0 = 3_600_000  # relógio da sessão (ms) no instante T0 da corrida de teste


def make_race_data() -> RaceData:
    """Uma corrida pequena e determinística: VER com 3 voltas e 5 posições; ALB não largou."""
    return RaceData(
        race=RaceInfo(2024, 21, "São Paulo Grand Prix", "São Paulo", "Brazil", T0, 69),
        entries=[
            EntryInfo("VER", 1, "Max Verstappen", "Red Bull Racing", "3671C6", 17, 1, "Finished"),
            EntryInfo("ALB", 23, "Alexander Albon", "Williams", "64C4FF", 7, 18, "Did not start"),
        ],
        laps=[
            LapInfo(
                "VER",
                1,
                99161,
                "INTERMEDIATE",
                1,
                1,
                11,
                T0 + timedelta(seconds=99),
                pit_in=False,
                pit_out=False,
                track_status="1",
                session_time_ms=SESSION0 + 99_000,
            ),
            LapInfo(
                "VER",
                2,
                87134,
                "INTERMEDIATE",
                2,
                1,
                10,
                T0 + timedelta(seconds=186),
                pit_in=True,
                pit_out=False,
                track_status="12",
                session_time_ms=SESSION0 + 186_000,
            ),
            LapInfo(
                "VER",
                3,
                None,
                "WET",
                1,
                2,
                None,
                T0 + timedelta(seconds=270),
                pit_in=False,
                pit_out=True,
                track_status="124",
                session_time_ms=SESSION0 + 270_000,
            ),
        ],
        tracks=[
            PositionTrack(
                "VER",
                [T0 + timedelta(milliseconds=220 * i) for i in range(5)],
                xs=[10, 20, 30, 40, 50],
                ys=[-5, -6, -7, -8, -9],
            )
        ],
    )


def count(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


def ingest(engine: Engine, data: RaceData, chunk_size: int = 2):
    # chunk_size=2 com 5 posições força 3 INSERTs em lote (2+2+1): exercita o chunking.
    with Session(engine) as session:
        report = write_race(session, data, chunk_size=chunk_size)
        session.commit()
    return report


# ---------------------------------------------------------------- schema / migration


def test_position_samples_is_a_hypertable_with_the_designed_layout(pg_engine: Engine):
    with pg_engine.connect() as conn:
        hypertables = (
            conn.execute(text("SELECT hypertable_name FROM timescaledb_information.hypertables"))
            .scalars()
            .all()
        )
        interval = conn.execute(
            text(
                "SELECT time_interval FROM timescaledb_information.dimensions "
                "WHERE hypertable_name = 'position_samples'"
            )
        ).scalar_one()
        indexes = set(
            conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'position_samples'")
            ).scalars()
        )

    assert hypertables == ["position_samples"]  # e só ela: as demais são tabelas normais
    assert interval == timedelta(days=7)
    # Só a PK e o índice de leitura da corrida: o índice automático em `time` foi desligado.
    assert indexes == {"pk_position_samples", "ix_position_samples_race_id_time"}


def test_models_and_migrations_are_in_sync(migrated_db_name: str):
    """Guarda contra deriva: se alguém alterar db/models.py sem criar a migration, quebra aqui."""
    result = _alembic(migrated_db_name, "check")

    assert result.returncode == 0, f"models != migrations:\n{result.stderr}"


# ---------------------------------------------------------------- ingestão (ON CONFLICT real)


def test_ingestion_writes_everything_the_first_time(clean_pg_engine: Engine):
    report = ingest(clean_pg_engine, make_race_data())

    assert (report.entries_changed, report.laps_changed, report.positions_inserted) == (2, 3, 5)
    assert report.positions_offered == 5
    assert count(clean_pg_engine, "teams") == 2
    assert count(clean_pg_engine, "drivers") == 2
    assert count(clean_pg_engine, "position_samples") == 5


def test_ingestion_is_idempotent_on_real_timescale(clean_pg_engine: Engine):
    data = make_race_data()
    ingest(clean_pg_engine, data)
    before = {t: count(clean_pg_engine, t) for t in ("races", "race_entries", "laps")}
    before["position_samples"] = count(clean_pg_engine, "position_samples")

    report = ingest(clean_pg_engine, data)  # 2ª execução, dados idênticos

    assert (report.entries_changed, report.laps_changed, report.positions_inserted) == (0, 0, 0)
    after = {t: count(clean_pg_engine, t) for t in before}
    assert after == before


def test_reingestion_updates_changed_laps_and_restores_missing_positions(clean_pg_engine: Engine):
    data = make_race_data()
    ingest(clean_pg_engine, data)
    with clean_pg_engine.begin() as conn:
        conn.execute(text("UPDATE laps SET lap_time_ms = 1 WHERE lap_number = 1"))
        conn.execute(text("DELETE FROM position_samples WHERE x IN (10, 20)"))

    report = ingest(clean_pg_engine, data)

    assert report.laps_changed == 1  # só a adulterada foi reescrita
    assert report.positions_inserted == 2  # só as apagadas voltaram
    with clean_pg_engine.connect() as conn:
        lap_time = conn.execute(text("SELECT lap_time_ms FROM laps WHERE lap_number = 1"))
        assert lap_time.scalar_one() == 99161


def strategy_columns(engine: Engine) -> list[tuple]:
    """As colunas que a Fase 3 acrescentou a `laps`, volta a volta."""
    query = "SELECT lap_number, pit_in, pit_out, track_status, session_time_ms FROM laps ORDER BY 1"
    with engine.connect() as conn:
        return [tuple(row) for row in conn.execute(text(query))]


def test_ingestion_round_trips_pit_flags_track_status_and_session_clock(clean_pg_engine: Engine):
    ingest(clean_pg_engine, make_race_data())

    assert strategy_columns(clean_pg_engine) == [
        (1, False, False, "1", SESSION0 + 99_000),
        (2, True, False, "12", SESSION0 + 186_000),
        (3, False, True, "124", SESSION0 + 270_000),
    ]


def test_reingesting_fills_the_new_columns_of_laps_saved_before_they_existed(
    clean_pg_engine: Engine,
):
    """O caso real do GP do Brasil: as 1.134 voltas já estavam no banco, sem as colunas novas."""
    data = make_race_data()
    ingest(clean_pg_engine, data)
    with clean_pg_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE laps SET pit_in = NULL, pit_out = NULL, track_status = NULL, "
                "session_time_ms = NULL"
            )
        )

    report = ingest(clean_pg_engine, data)
    second = ingest(clean_pg_engine, data)

    assert report.laps_changed == 3  # todas foram completadas...
    assert second.laps_changed == 0  # ...e a seguinte volta a ser idempotente
    assert strategy_columns(clean_pg_engine)[1] == (2, True, False, "12", SESSION0 + 186_000)


def laps_only(data: RaceData, overrides: dict[int, dict] | None = None) -> RaceData:
    """A mesma corrida como o `--laps-only` a entrega: sem posições e sem horário absoluto.
    `overrides` troca campos de voltas específicas (por número) para simular um dado corrigido."""
    overrides = overrides or {}
    laps = [
        replace(lap, completed_at=None, **overrides.get(lap.lap_number, {})) for lap in data.laps
    ]
    return replace(data, laps=laps, tracks=[], laps_only=True)


def test_laps_only_ingestion_updates_laps_without_erasing_completed_at_or_positions(
    clean_pg_engine: Engine,
):
    """O Brasil já tem posições e horários de uma ingestão completa: reingerir só as voltas para
    ganhar as colunas novas NÃO pode apagar nenhum dos dois (o replay depende deles)."""
    data = make_race_data()
    ingest(clean_pg_engine, data)

    report = ingest(clean_pg_engine, laps_only(data))

    assert report.laps_changed == 0  # nada que a ingestão completa já gravara mudou
    assert report.positions_offered == 0  # nenhuma posição oferecida, logo nenhuma tocada
    with clean_pg_engine.connect() as conn:
        missing = conn.execute(text("SELECT count(*) FROM laps WHERE completed_at IS NULL"))
        assert missing.scalar_one() == 0
    assert count(clean_pg_engine, "position_samples") == 5


def test_laps_only_ingestion_still_updates_the_other_lap_columns(clean_pg_engine: Engine):
    data = make_race_data()
    ingest(clean_pg_engine, data)
    corrected = laps_only(data, {2: {"pit_in": False, "track_status": "1"}})

    report = ingest(clean_pg_engine, corrected)

    assert report.laps_changed == 1
    assert strategy_columns(clean_pg_engine)[1] == (2, False, False, "1", SESSION0 + 186_000)
    with clean_pg_engine.connect() as conn:  # e o horário da volta 2 continua lá
        done = conn.execute(text("SELECT completed_at FROM laps WHERE lap_number = 2"))
        assert done.scalar_one() == T0 + timedelta(seconds=186)


def test_laps_only_ingestion_of_a_new_race_leaves_completed_at_null_and_writes_no_positions(
    clean_pg_engine: Engine,
):
    report = ingest(clean_pg_engine, laps_only(make_race_data()))
    again = ingest(clean_pg_engine, laps_only(make_race_data()))

    assert (report.laps_changed, report.positions_offered, report.positions_inserted) == (3, 0, 0)
    assert again.laps_changed == 0  # idempotente também neste modo
    assert count(clean_pg_engine, "position_samples") == 0
    with clean_pg_engine.connect() as conn:
        nulls = conn.execute(text("SELECT count(*) FROM laps WHERE completed_at IS NULL"))
        assert nulls.scalar_one() == 3
    assert strategy_columns(clean_pg_engine)[0] == (1, False, False, "1", SESSION0 + 99_000)


def test_failed_ingestion_leaves_no_partial_race(clean_pg_engine: Engine):
    """Atomicidade: se qualquer etapa falha, NADA fica gravado (rollback da transação única)."""
    data = make_race_data()
    poisoned = PositionTrack("VER", [T0 + timedelta(hours=1)], xs=[None], ys=[1])  # x NULL
    bad = RaceData(data.race, data.entries, data.laps, [*data.tracks, poisoned])

    with pytest.raises(Exception, match="null value"), Session(clean_pg_engine) as session:
        write_race(session, bad)
        session.commit()

    assert [count(clean_pg_engine, t) for t in ("races", "race_entries", "laps")] == [0, 0, 0]
    assert count(clean_pg_engine, "position_samples") == 0


# ---------------------------------------------------------------- API sobre o banco real


@pytest.fixture
def real_client(clean_pg_engine: Engine) -> Iterator[TestClient]:
    ingest(clean_pg_engine, make_race_data())
    factory = sessionmaker(clean_pg_engine, expire_on_commit=False)

    def override() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_api_serves_laps_ingested_into_real_timescale(real_client: TestClient):
    response = real_client.get("/races/2024/21/laps", params={"driver": "ver"})

    assert response.status_code == 200
    body = response.json()
    assert body["race"]["name"] == "São Paulo Grand Prix"
    first, _, third = body["laps"]
    assert (first["lap_time_ms"], first["compound"], first["position"]) == (
        99161,
        "INTERMEDIATE",
        11,
    )
    assert third["lap_time_ms"] is None and third["position"] is None
    # timestamptz real: o instante exato sobrevive ao round-trip pelo Postgres, em UTC.
    assert datetime.fromisoformat(first["completed_at"]) == T0 + timedelta(seconds=99)


def test_api_driver_rules_on_real_timescale(real_client: TestClient):
    url = "/races/2024/21/laps"

    assert real_client.get(url, params={"driver": "ALB"}).json()["laps"] == []  # DNS: 200 vazio
    missing = real_client.get(url, params={"driver": "HAM"})
    assert missing.status_code == 404
    assert "ALB, VER" in missing.json()["detail"]
    assert real_client.get("/races/2024/22/laps").status_code == 404


def test_health_reports_database_ok_against_real_timescale(real_client: TestClient):
    response = real_client.get("/health")

    assert response.status_code == 200
    # O Redis aqui é o stub saudável do conftest raiz: o /health contra o Redis REAL está em
    # test_redis.py. Este teste é sobre o banco.
    assert response.json()["checks"]["database"] == "ok"
