"""Utilitários compartilhados pelos testes de integração do streaming (worker, WebSocket)."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from replay.events import stream_key
from replay.state import WORKER_GROUP
from worker.consumer import StateWorker

RACE = (2024, 21)
STREAM = stream_key(*RACE)


async def wait_until(predicate: Callable[[], Awaitable[bool]], timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not await predicate():
            await asyncio.sleep(0.01)


async def drained(redis: Redis, stream: str = STREAM) -> bool:
    """O grupo leu o stream inteiro e confirmou tudo (nada pendente). Antes de o worker criar o
    grupo (ou de o stream existir) a resposta é simplesmente "ainda não"."""
    try:
        info = await redis.xinfo_stream(stream)
        groups = await redis.xinfo_groups(stream)
    except ResponseError:
        return False
    group = next((g for g in groups if g["name"] == WORKER_GROUP), None)
    if group is None:
        return False
    return group["pending"] == 0 and group["last-delivered-id"] == info["last-generated-id"]


@contextlib.asynccontextmanager
async def running(worker: StateWorker) -> AsyncIterator[StateWorker]:
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        yield worker
    finally:
        stop.set()
        try:
            await asyncio.wait_for(task, 5)
        except TimeoutError:  # pragma: no cover
            task.cancel()


def make_worker(redis: Redis, **kwargs) -> StateWorker:
    defaults = {
        "block_ms": 100,
        "discover_interval_s": 0.05,
        "claim_interval_s": 0.05,
        "claim_min_idle_ms": 60_000,
    }
    return StateWorker(redis, **(defaults | kwargs))
