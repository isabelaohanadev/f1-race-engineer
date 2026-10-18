"""O GET e o evento `strategy` do WebSocket, de ponta a ponta contra o Redis REAL: servidor uvicorn,
hub, cliente WebSocket e armazém de verdade. Os resultados são gravados direto no armazém (o
consumidor tem os testes dele em test_strategy_consumer.py)."""

import asyncio
import json

import httpx
import pytest
import websockets
from redis.asyncio import Redis

from replay.state import epoch_key
from scripts.watch_race import RaceView
from strategy.service.results import ResultStore
from tests.integration.helpers import RACE
from tests.test_strategy_api import document

Y, R = RACE


def doc(lap: int, epoch: int = 1) -> dict:
    return document(lap, year=Y, round=R, epoch=epoch)


async def put(store: ResultStore, lap: int, epoch: int = 1) -> None:
    await store.put(Y, R, epoch=epoch, lap=lap, payload=json.dumps(doc(lap, epoch)), version=1)


async def next_strategy(ws, timeout_s: float = 5.0) -> dict:
    """A próxima mensagem `strategy` do WebSocket (ignora o resto)."""
    async with asyncio.timeout(timeout_s):
        while True:
            message = json.loads(await ws.recv())
            if message["type"] == "strategy":
                return message


@pytest.fixture
def http(server: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=server.replace("ws://", "http://"), timeout=5)


@pytest.fixture
def results(redis_client: Redis) -> ResultStore:
    return ResultStore(redis_client)


# ================================================================ GET


async def test_the_get_returns_what_the_consumer_stored(http: httpx.AsyncClient, results):
    await put(results, 29)
    await put(results, 30)

    async with http:
        latest = await http.get(f"/races/{Y}/{R}/strategy")
        by_lap = await http.get(f"/races/{Y}/{R}/strategy", params={"lap": 29})
        missing = await http.get(f"/races/{Y}/{R}/strategy", params={"lap": 12})

    assert latest.status_code == 200 and latest.json() == doc(30)
    assert by_lap.json() == doc(29)
    assert missing.status_code == 404 and "a mais recente é a 30" in missing.json()["detail"]


async def test_the_get_hides_the_results_of_a_replay_that_already_passed(
    http: httpx.AsyncClient, results, redis_client: Redis
):
    await put(results, 30, epoch=1)
    await redis_client.set(epoch_key(Y, R), 2)  # um replay novo começou e ainda não calculou nada

    async with http:
        response = await http.get(f"/races/{Y}/{R}/strategy")

    assert response.status_code == 404


async def test_the_get_never_writes_or_calculates(
    http: httpx.AsyncClient, results, redis_client: Redis
):
    await put(results, 30)
    before = {key: await redis_client.type(key) async for key in redis_client.scan_iter("*")}

    async with http:
        for _ in range(5):
            await http.get(f"/races/{Y}/{R}/strategy")
            await http.get(f"/races/{Y}/{R}/strategy", params={"lap": 12})

    after = {key: await redis_client.type(key) async for key in redis_client.scan_iter("*")}
    assert after == before  # nenhuma chave nova: só leitura
    assert await redis_client.xlen(f"strategy:events:{Y}:{R}") == 1


# ================================================================ WebSocket


async def test_a_client_gets_the_latest_result_after_the_snapshot_and_then_the_live_ones(
    url: str, results: ResultStore
):
    await put(results, 10)
    view = RaceView()

    async with websockets.connect(url) as ws:
        snapshot = json.loads(await ws.recv())
        assert snapshot["type"] == "snapshot"
        view.apply_snapshot(snapshot)
        first = await next_strategy(ws)  # o resultado de agora, para quem entra no meio
        view.apply_event(first)
        assert first["lap"] == 10 and first == doc(10)  # o mesmo documento do armazém

        await put(results, 11)
        live = await next_strategy(ws)  # e o novo, empurrado pelo hub
        view.apply_event(live)

    assert live["lap"] == 11 and live["id"]  # o ID da entrada do stream, como nos outros eventos
    assert {k: v for k, v in live.items() if k != "id"} == doc(11)
    assert view.strategy is not None and view.strategy["lap"] == 11


async def test_a_result_published_while_joining_is_not_lost_nor_applied_twice(
    url: str, results: ResultStore
):
    """O último do armazém e o do stream podem ser o mesmo resultado: o cliente ignora a repetição
    (vale o maior `lap` do epoch)."""
    await put(results, 10)
    view = RaceView()

    async with websockets.connect(url) as ws:
        view.apply_snapshot(json.loads(await ws.recv()))
        for lap in (11, 12, 13):
            await put(results, lap)
        seen = []
        async with asyncio.timeout(5):
            while not seen or seen[-1] != 13:
                message = await next_strategy(ws)
                view.apply_event(message)
                seen.append(message["lap"])

    assert view.strategy is not None and view.strategy["lap"] == 13
    assert seen == sorted(seen) and set(seen) >= {13}  # nunca volta atrás


async def test_every_client_receives_the_same_strategy_events(url: str, results: ResultStore):
    async with websockets.connect(url) as one, websockets.connect(url) as two:
        await one.recv()
        await two.recv()
        await put(results, 20)

        a, b = await next_strategy(one), await next_strategy(two)

    assert a == b and a["lap"] == 20


async def test_a_client_never_sees_a_result_of_a_replay_that_already_passed(
    url: str, results: ResultStore, redis_client: Redis
):
    await put(results, 40, epoch=1)
    await redis_client.set(epoch_key(Y, R), 2)  # o replay 2 começou; o 1 não vale mais

    async with websockets.connect(url) as ws:
        await ws.recv()  # snapshot
        with pytest.raises(TimeoutError):  # o resultado velho não vem
            await next_strategy(ws, timeout_s=0.7)

        await put(results, 5, epoch=2)
        message = await next_strategy(ws)

    assert message["epoch"] == 2 and message["lap"] == 5
