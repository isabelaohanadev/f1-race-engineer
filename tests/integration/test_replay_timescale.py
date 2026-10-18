"""O replay inteiro sobre a infraestrutura REAL: TimescaleDB (hypertable) -> gerenciador -> Redis.

O relógio é falso (FakeClock) de propósito: o que está sendo provado aqui é a leitura paginada na
hypertable e a publicação no Redis, não o ritmo (que os testes unitários já cobrem). Assim o teste
roda em milissegundos e é determinístico mesmo com I/O de rede de verdade.
"""

import asyncio
import re
from datetime import UTC, datetime, timedelta

import pytest
from redis.asyncio import Redis
from sqlalchemy import Engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker

from ingestion.dto import EntryInfo, LapInfo, PositionTrack, RaceData, RaceInfo
from ingestion.writer import write_race
from replay.engine import ReplayState
from replay.errors import ReplayAlreadyRunning
from replay.events import (
    LapCompletedEvent,
    PositionEvent,
    RaceStatusEvent,
    laps_key,
    parse_event,
    stream_key,
)
from replay.lock import lock_key
from replay.manager import ReplayManager
from replay.source import RaceSource, position_page_query
from tests.replay.fakes import FakeClock

T0 = datetime(2024, 11, 3, 15, 50, tzinfo=UTC)
RACE = (2024, 21)


def race_data(year: int = 2024, round_number: int = 21, start: datetime = T0) -> RaceData:
    """3 pilotos com amostras nos MESMOS 40 instantes (empate em `time` em todas as linhas: o caso
    que quebra uma paginação feita só por horário) e 3 voltas entre elas."""
    times = [start + timedelta(milliseconds=220 * i) for i in range(40)]
    codes = ("VER", "NOR", "ALB")
    return RaceData(
        race=RaceInfo(year, round_number, "GP", "Cidade", "País", start, 3),
        entries=[
            EntryInfo(code, n, f"Piloto {code}", "Equipe", "FFFFFF", n, n, "Finished")
            for n, code in enumerate(codes, start=1)
        ],
        laps=[
            LapInfo("VER", 1, 90_000, "SOFT", 1, 1, 1, start + timedelta(seconds=2)),
            LapInfo("NOR", 1, 91_000, "HARD", 1, 1, 2, start + timedelta(seconds=3)),
            LapInfo("VER", 2, 89_000, "SOFT", 2, 1, 1, start + timedelta(seconds=6)),
        ],
        tracks=[
            PositionTrack(code, times, xs=[i * (k + 1) for i in range(40)], ys=[k] * 40)
            for k, code in enumerate(codes)
        ],
    )


def ingest(engine: Engine, data: RaceData) -> None:
    with Session(engine) as session:
        write_race(session, data)
        session.commit()


@pytest.fixture
def session_factory(clean_pg_engine: Engine) -> sessionmaker[Session]:
    ingest(clean_pg_engine, race_data())
    return sessionmaker(clean_pg_engine, expire_on_commit=False)


async def wait_for_state(manager: ReplayManager, state: ReplayState, timeout_s: float = 10) -> None:
    async with asyncio.timeout(timeout_s):
        while (status := manager.status(*RACE)) is None or status.state is not state:
            await asyncio.sleep(0.005)  # polling de verdade: o I/O de rede não é do relógio falso


# ---------------------------------------------------------------- leitura na hypertable


@pytest.mark.parametrize("page_size", [1, 7, 119, 120, 121])
async def test_keyset_pages_over_the_real_hypertable_match_a_single_big_read(
    session_factory, page_size
):
    async def collect(size: int):
        source = await RaceSource.open(session_factory, *RACE, page_size=size)
        return [e async for e in source.events()]

    paged, whole = await collect(page_size), await collect(5000)

    assert paged == whole
    positions = [e for e in paged if isinstance(e, PositionEvent)]
    assert len(positions) == 120 and len({(p.driver, p.t) for p in positions}) == 120
    assert [p.t for p in positions] == sorted(p.t for p in positions)


def test_the_page_query_excludes_the_chunks_before_the_cursor(clean_pg_engine: Engine):
    """Regressão do que foi MEDIDO: a condição `time >=` explícita faz o TimescaleDB descartar os
    chunks anteriores ao cursor no planejamento (6 chunks -> 3 num teste maior). Cada corrida cai
    num chunk de 7 dias; as 4 abaixo ficam em semanas diferentes."""
    for k in range(4):
        ingest(clean_pg_engine, race_data(2024, k + 1, T0 + timedelta(days=7 * k)))
    with clean_pg_engine.connect() as conn:
        total_chunks = conn.execute(
            text("SELECT count(*) FROM timescaledb_information.chunks")
        ).scalar_one()
        race_id, driver_id = conn.execute(
            text(
                "SELECT r.id, e.driver_id FROM races r JOIN race_entries e ON e.race_id = r.id "
                "WHERE r.round = 3 LIMIT 1"
            )
        ).one()
        cursor = (T0 + timedelta(days=14, seconds=1), driver_id)  # no meio da 3ª corrida

        query = position_page_query(race_id, cursor, 100)
        sql = str(
            query.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        )
        plan = "\n".join(conn.execute(text("EXPLAIN (COSTS OFF) " + sql)).scalars())

    touched = set(re.findall(r"_hyper_\d+_\d+_chunk", plan))
    assert total_chunks == 4
    assert 0 < len(touched) < total_chunks  # as semanas 1 e 2 (antes do cursor) ficaram de fora


# ---------------------------------------------------------------- replay completo


async def test_the_whole_chain_delivers_every_event_once_and_in_order(
    session_factory, redis_client: Redis
):
    manager = ReplayManager(session_factory, redis_client, clock=FakeClock(), page_size=7)

    await manager.start(*RACE, speed=16)
    await wait_for_state(manager, ReplayState.FINISHED)

    stream = [parse_event(f) for _, f in await redis_client.xrange(stream_key(*RACE))]
    reference = await RaceSource.open(session_factory, *RACE, page_size=5000)
    expected = [e async for e in reference.events()]
    # o stream principal entrega as voltas SEM os campos da estratégia (o contrato é o de sempre)
    assert stream == [e.slim() if isinstance(e, LapCompletedEvent) else e for e in expected]
    # o stream de voltas leva as voltas inteiras, o início e o fim, na mesma ordem
    laps_stream = [parse_event(f) for _, f in await redis_client.xrange(laps_key(*RACE))]
    assert laps_stream == [
        e
        for e in expected
        if isinstance(e, LapCompletedEvent)
        or (isinstance(e, RaceStatusEvent) and e.state in ("started", "finished"))
    ]
    assert isinstance(stream[0], RaceStatusEvent) and stream[0].state == "started"
    assert isinstance(stream[-1], RaceStatusEvent) and stream[-1].state == "finished"
    assert sum(isinstance(e, PositionEvent) for e in stream) == 120
    assert sum(isinstance(e, LapCompletedEvent) for e in stream) == 3
    final = manager.status(*RACE)
    assert final is not None and final.events_published == len(stream)
    assert await redis_client.exists(lock_key(*RACE)) == 0  # lock liberado ao terminar


async def test_a_second_replay_starts_from_a_clean_stream(session_factory, redis_client: Redis):
    manager = ReplayManager(session_factory, redis_client, clock=FakeClock())

    await manager.start(*RACE, speed=16)
    await wait_for_state(manager, ReplayState.FINISHED)
    await _until_lock_is_free(redis_client)
    first_length = await redis_client.xlen(stream_key(*RACE))

    await manager.start(*RACE, speed=16)
    await wait_for_state(manager, ReplayState.FINISHED)

    # Sem o DEL no início, o stream teria o dobro de entradas (2 replays empilhados).
    assert await redis_client.xlen(stream_key(*RACE)) == first_length


async def test_two_managers_cannot_replay_the_same_race_through_the_real_lock(
    session_factory, redis_client: Redis
):
    """Dois gerenciadores = dois processos da API, ambos falando com o MESMO Redis."""
    one = ReplayManager(session_factory, redis_client, clock=FakeClock())
    other = ReplayManager(session_factory, redis_client, clock=FakeClock())
    await one.start(*RACE)
    one.pause(*RACE)
    await wait_for_state(one, ReplayState.PAUSED)
    entries_before = await redis_client.xlen(stream_key(*RACE))

    with pytest.raises(ReplayAlreadyRunning):
        await other.start(*RACE)

    assert await redis_client.xlen(stream_key(*RACE)) == entries_before  # stream intocado
    assert await redis_client.pttl(lock_key(*RACE)) > 0  # o lock real, com TTL, está lá
    await one.stop(*RACE)
    assert await redis_client.exists(lock_key(*RACE)) == 0


async def _until_lock_is_free(redis: Redis, timeout_s: float = 5) -> None:
    async with asyncio.timeout(timeout_s):
        while await redis.exists(lock_key(*RACE)):
            await asyncio.sleep(0.005)
