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
from replay.state import WORKER_GROUP, RaceStateStore, classify
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

RANK = {"started": 0, "running": 1, "finished": 2}


# ---------------------------------------------------------------- o "cliente": reconstrói o estado


class ClientView:
    """O que um cliente faz com o que recebe. As regras são as DOCUMENTADAS no protocolo:
    epoch maior = replay novo (descarta tudo); por piloto, uma posição só vale se `t` > `pos_t`, e
    uma volta se `t` > `lap_t`; o status da corrida só avança em (volta, estado). Com isso,
    repetições e atrasos (o catch-up pode repetir o que o snapshot já tinha) são inofensivos."""

    def __init__(self) -> None:
        self.epoch = 0
        self.race: dict = {}
        self.cars: dict[str, dict] = {}

    def apply_snapshot(self, message: dict) -> None:
        self.epoch = message["epoch"]
        self.race = dict(message["race"])
        self.cars = {code: dict(car) for code, car in message["cars"].items()}

    def apply_event(self, message: dict) -> None:
        if message["epoch"] < self.epoch:
            return  # evento de um replay antigo
        if message["epoch"] > self.epoch:
            self.epoch, self.race, self.cars = message["epoch"], {}, {}  # replay novo
        match message["type"]:
            case "position":
                car = self.cars.setdefault(message["driver"], {})
                if car.get("pos_t", -1) < message["t"]:
                    car |= {"x": message["x"], "y": message["y"], "pos_t": message["t"]}
            case "lap_completed":
                car = self.cars.setdefault(message["driver"], {})
                if car.get("lap_t", -1) < message["t"]:
                    car |= {"lap": message["lap"], "lap_t": message["t"]}
                    for name in ("lap_time_ms", "compound", "position"):
                        if message[name] is None:
                            car.pop(name, None)
                        else:
                            car[name] = message[name]
            case "race_status":
                current = (self.race.get("lap", 0), RANK.get(self.race.get("state", ""), -1))
                if (message["lap"], RANK[message["state"]]) > current:
                    self.race = {"state": message["state"], "lap": message["lap"]}
                    if message["total_laps"] is not None:
                        self.race["total_laps"] = message["total_laps"]

    @property
    def finished(self) -> bool:
        return self.race.get("state") == "finished"

    @property
    def classification(self) -> list[str]:
        return classify(self.cars)


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
