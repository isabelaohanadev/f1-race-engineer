"""O que os testes de integração do consumidor de estratégia compartilham: o consumidor com
tempos curtos, o produtor de voltas sintéticas (pelo publisher de verdade) e as consultas ao
armazém."""

import asyncio
import contextlib
import json
import random
from collections.abc import AsyncIterator

from redis.asyncio import Redis

from replay.events import LapCompletedEvent, RaceStatusEvent, laps_key
from replay.publisher import StreamPublisher
from replay.state import RaceStateStore
from strategy.live import Constants
from strategy.priors import DEFAULT_PRIORS
from strategy.protocol import frozen_sim_config
from strategy.service.consumer import StrategyConsumer
from strategy.service.results import ResultStore
from tests.integration.helpers import wait_until
from tests.strategy.synthetic import World, chase_field, lap_event
from tests.strategy.test_montecarlo import sim_priors

Y, R = 2024, 10
TOTAL = 40
CONFIG = frozen_sim_config(n_sims=300)
CONSTANTS = Constants(DEFAULT_PRIORS, sim_priors())


class FakeConstants:
    async def for_race(self, year: int, round_number: int) -> Constants:
        return CONSTANTS


def make_consumer(redis: Redis, **kwargs) -> StrategyConsumer:
    defaults = dict(
        wait_s=0.05, lock_ttl_ms=600, standby_poll_s=0.05, block_ms=50, discover_interval_s=0.05,
        sim_config=CONFIG,
    )  # fmt: skip
    return StrategyConsumer(redis, FakeConstants(), **(defaults | kwargs))


@contextlib.asynccontextmanager
async def running(consumer: StrategyConsumer) -> AsyncIterator[StrategyConsumer]:
    stop = asyncio.Event()
    task = asyncio.create_task(consumer.run(stop))
    try:
        yield consumer
    finally:
        stop.set()
        try:
            await asyncio.wait_for(task, 10)
        except TimeoutError:  # pragma: no cover
            task.cancel()


def make_rows():
    return chase_field(world=World(total_laps=TOTAL))


class Producer:
    """Publica as voltas de uma corrida sintética no Redis, pelo publisher de verdade."""

    def __init__(self, redis: Redis, rows, year: int = Y, round_number: int = R, epoch: int = 1):
        self.redis, self.rows, self.epoch = redis, rows, epoch
        self.year, self.round = year, round_number
        self.publisher = StreamPublisher(
            redis, f"race:{year}:{round_number}", epoch, laps_stream=laps_key(year, round_number)
        )
        self.state = RaceStateStore(redis)
        self.leader_t = {
            n: min(r.session_time_ms for r in rows if r.lap_number == n)
            for n in range(1, TOTAL + 1)
        }

    async def start(self) -> None:
        await self.state.register_stream(self.year, self.round)
        await self.publisher.publish_many([RaceStatusEvent("started", 1, 1_700_000_000_000, TOTAL)])
        for lap in range(1, 5):  # as voltas de antes da primeira volta avaliada (a 5)
            await self.publish_lap(lap)

    def lap_events(self, lap: int) -> list[LapCompletedEvent]:
        """Tudo o que cruza a linha depois do líder da volta anterior, até o líder desta volta."""
        low, high = self.leader_t.get(lap - 1, -1), self.leader_t[lap]
        rows = [r for r in self.rows if low < r.session_time_ms <= high]
        return [lap_event(r) for r in sorted(rows, key=lambda r: (r.session_time_ms, r.driver))]

    async def publish_lap(self, lap: int, *, shuffle: random.Random | None = None) -> None:
        events = self.lap_events(lap)
        if shuffle is not None:
            shuffle.shuffle(events)
        await self.publisher.publish_many(events)

    async def finish(self) -> None:
        await self.publisher.publish_many(
            [RaceStatusEvent("finished", TOTAL, 1_700_009_000_000, TOTAL)]
        )


def store_of(redis: Redis) -> ResultStore:
    return ResultStore(redis)


async def has_result(redis: Redis, lap: int, epoch: int = 1, year: int = Y, rnd: int = R) -> bool:
    return await store_of(redis).has(year, rnd, epoch=epoch, lap=lap)


async def wait_result(redis: Redis, lap: int, **kwargs) -> None:
    async def done() -> bool:
        return await has_result(redis, lap, **kwargs)

    await wait_until(done, timeout_s=20)


async def stored(redis: Redis, lap: int, year: int = Y, rnd: int = R) -> dict:
    result = await store_of(redis).get(year, rnd, lap)
    assert result is not None
    return json.loads(result.payload)


async def active_count(*consumers: StrategyConsumer) -> int:
    return sum(c.active for c in consumers)
