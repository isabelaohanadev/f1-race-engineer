"""Utilitários compartilhados pelos testes de integração do streaming (worker, WebSocket)."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import websockets
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from replay.events import stream_key
from replay.state import WORKER_GROUP, RaceStateStore
from scripts.watch_race import RaceView
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


# ---------------------------------------------------------------- o "cliente"

ClientView = RaceView  # o MESMO cliente de referência que o README ensina a copiar


@dataclass
class ClientRun:
    view: ClientView
    as_of: str
    received_ids: list[str]


async def run_client(url: str, *, until, timeout_s: float = 25.0) -> ClientRun:
    """Conecta, aplica o snapshot e os eventos até `until(view)`; devolve o que viu."""
    view, ids = ClientView(), []
    async with asyncio.timeout(timeout_s), websockets.connect(url) as ws:
        snapshot = json.loads(await ws.recv())
        assert snapshot["type"] == "snapshot"
        view.apply_snapshot(snapshot)
        while not until(view):
            message = json.loads(await ws.recv())
            ids.append(message["id"])
            view.apply_event(message)
    return ClientRun(view, snapshot["as_of"], ids)


async def final_truth(store: RaceStateStore, redis: Redis):
    await wait_until(lambda: drained(redis), timeout_s=10)
    return await store.snapshot(*RACE)


async def stream_ids(redis: Redis) -> list[str]:
    return [entry_id for entry_id, _ in await redis.xrange(STREAM)]
