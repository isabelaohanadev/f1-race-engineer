"""A cadeia inteira, de ponta a ponta: banco -> motor de replay -> stream de voltas -> consumidor
de estratégia -> armazém -> hub -> WebSocket (e o GET). Tudo de verdade (Timescale e Redis reais, o
uvicorn, o publisher, o consumer group, a thread do Monte Carlo); só o relógio do replay é falso,
para a corrida de 40 voltas passar em segundos."""

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta

import httpx
import websockets
from redis.asyncio import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Driver, Lap, Race, RaceEntry, Team
from replay.engine import ReplayState
from replay.manager import ReplayManager
from strategy.service.results import ResultStore
from tests.integration.helpers import RACE, wait_until
from tests.integration.test_strategy_consumer import TOTAL, make_consumer, running
from tests.replay.fakes import FakeClock
from tests.strategy.synthetic import World, chase_field

BASE = datetime(2024, 6, 23, 13, 0, tzinfo=UTC)


def seed_strategy_race(engine: Engine) -> sessionmaker:
    """Uma corrida sintética de 14 carros e 40 voltas, com os campos que a estratégia lê."""
    rows = chase_field(world=World(total_laps=TOTAL))
    codes = sorted({r.driver for r in rows})
    with Session(engine) as s:
        s.add(Team(id=1, name="Equipe"))
        s.add_all(Driver(id=i + 1, code=c, full_name=c) for i, c in enumerate(codes))
        s.add(Race(id=1, year=RACE[0], round=RACE[1], name="Sintético", total_laps=TOTAL))
        s.flush()
        ids = {c: i + 1 for i, c in enumerate(codes)}
        s.add_all(
            RaceEntry(race_id=1, driver_id=ids[c], team_id=1, driver_number=ids[c]) for c in codes
        )
        s.flush()
        s.add_all(
            Lap(
                race_id=1,
                driver_id=ids[r.driver],
                lap_number=r.lap_number,
                lap_time_ms=r.lap_time_ms,
                compound=r.compound,
                tyre_life=r.tyre_life,
                stint=r.stint,
                position=r.position,
                completed_at=BASE + timedelta(milliseconds=r.session_time_ms),
                session_time_ms=r.session_time_ms,
                pit_in=r.pit_in,
                pit_out=r.pit_out,
                track_status=r.track_status,
            )
            for r in rows
        )
        s.commit()
    return sessionmaker(engine, expire_on_commit=False)


async def test_a_replay_ends_as_strategy_events_on_the_websocket_and_in_the_get(
    clean_pg_engine: Engine, redis_client: Redis, server: str, url: str
):
    factory = seed_strategy_race(clean_pg_engine)
    manager = ReplayManager(factory, redis_client, clock=FakeClock(), page_size=500)
    received: list[dict] = []
    connected = asyncio.Event()  # o cliente recebeu o snapshot: o hub já o inscreveu

    async def client() -> None:
        async with websockets.connect(url, max_size=None) as ws:
            async for raw in ws:
                message = json.loads(raw)
                connected.set()
                if message["type"] == "strategy":
                    received.append(message)

    async with running(make_consumer(redis_client, wait_s=0.1)):
        watcher = asyncio.create_task(client())
        await asyncio.wait_for(connected.wait(), 5)
        await manager.start(*RACE, speed=16)
        await _wait_finished(manager)
        last = TOTAL - 1  # a última volta não se calcula: é o fim da corrida

        async def last_result_stored() -> bool:
            return await ResultStore(redis_client).has(*RACE, epoch=1, lap=last)

        await wait_until(last_result_stored, timeout_s=30)
        await wait_until(_received(received, last), timeout_s=5)  # e o WebSocket a entregou
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        async with httpx.AsyncClient(base_url=server.replace("ws://", "http://")) as http:
            rest = (await http.get(f"/races/{RACE[0]}/{RACE[1]}/strategy")).json()
    await manager.shutdown()

    pushed = next(m for m in received if m["lap"] == last)
    assert {k: v for k, v in pushed.items() if k != "id"} == rest  # o mesmo documento nos dois
    assert rest["epoch"] == 1 and rest["laps_left"] == 1 and rest["reliability"] == "experimental"
    assert rest["baseline"]["leader"] in {d["driver"] for d in rest["forecast"]["drivers"]}
    assert abs(sum(d["win"] for d in rest["forecast"]["drivers"]) - 1.0) < 1e-6
    assert [m["lap"] for m in received] == sorted({m["lap"] for m in received})  # nunca volta atrás
    assert rest["inputs"]["cars_reported"] == 14 and rest["inputs"]["late_laps"] == 0


def _received(received: list[dict], lap: int):
    async def done() -> bool:
        return any(m["lap"] == lap for m in received)

    return done


async def _wait_finished(manager: ReplayManager) -> None:
    async def finished() -> bool:
        status = manager.status(*RACE)
        return status is not None and status.state is ReplayState.FINISHED

    await wait_until(finished, timeout_s=30)
