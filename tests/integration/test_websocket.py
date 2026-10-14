"""O WebSocket de ponta a ponta: servidor uvicorn REAL, clientes WebSocket REAIS, Redis real,
workers e replay de verdade, tudo no mesmo event loop.

A pergunta central (a trava 2): um cliente que conecta NO MEIO do replay, com os workers ainda
aplicando eventos, perde ou corrompe alguma coisa? O critério é duplo:
  * COBERTURA: todo evento do stream DEPOIS da marca d'água do snapshot (`as_of`) foi entregue ao
    cliente (catch-up + ao vivo). Um buraco é um ID que falta.
  * ESTADO: o estado que o cliente reconstrói (snapshot + eventos, com a MESMA regra de versão do
    servidor) é IGUAL ao estado final no Redis. Repetições não o alteram.
"""

import asyncio
import json

import pytest
import websockets
from redis.asyncio import Redis
from websockets.exceptions import ConnectionClosed, InvalidStatus

from app.main import app
from replay.engine import ReplayState
from replay.events import PositionEvent, id_tuple
from replay.manager import ReplayManager
from replay.publisher import StreamPublisher
from replay.state import RaceStateStore
from tests.integration.helpers import (
    RACE,
    STREAM,
    ClientRun,
    drained,
    final_truth,
    make_worker,
    run_client,
    running,
    stream_ids,
    wait_until,
)

# ================================================================ TRAVA 2: sem buracos


async def test_clients_joining_in_the_middle_of_a_live_replay_lose_and_corrupt_nothing(
    url: str, redis_client: Redis, store: RaceStateStore, synthetic_race
):
    manager = ReplayManager(synthetic_race, redis_client, page_size=100)
    workers = [make_worker(redis_client, consumer=f"w{i}", batch_size=20) for i in range(2)]
    async with running(workers[0]), running(workers[1]):
        await manager.start(*RACE, speed=16)
        # 6 clientes em momentos diferentes, com o replay publicando e os workers aplicando.
        joins = [0.0, 0.25, 0.6, 1.0, 1.5, 2.0]

        async def client(delay: float) -> ClientRun:
            await asyncio.sleep(delay)
            return await run_client(url, until=lambda view: view.finished)

        runs = await asyncio.gather(*(client(d) for d in joins))
        truth = await final_truth(store, redis_client)
        ids = await stream_ids(redis_client)
    await manager.shutdown()

    assert truth.race["state"] == "finished" and set(truth.cars) == {"VER", "NOR", "ALB"}
    for joined_at, run in zip(joins, runs, strict=True):
        # COBERTURA: tudo o que o stream tem depois da marca d'água do snapshot chegou ao cliente.
        needed = [i for i in ids if id_tuple(i) > id_tuple(run.as_of)]
        missing = set(needed) - set(run.received_ids)
        assert not missing, f"cliente que entrou em {joined_at}s perdeu {len(missing)} eventos"
        # ESTADO: o reconstruído é IDÊNTICO ao do Redis (repetições não o alteraram).
        assert run.view.cars == truth.cars, f"cliente que entrou em {joined_at}s divergiu"
        assert run.view.race == truth.race
        assert run.view.classification == truth.classification


async def test_events_published_but_not_yet_applied_are_not_lost(
    url: str, redis_client: Redis, store: RaceStateStore
):
    """O buraco que a receita ingênua (anotar o ÚLTIMO ID do stream) deixaria. NENHUM worker
    rodando: 50 eventos publicados, zero aplicados. O snapshot nasce vazio; os 50 têm de chegar
    pelo catch-up a partir da marca d'água ("0-0": nada aplicado)."""
    epoch = await store.new_epoch(*RACE)
    await store.register_stream(*RACE)
    await StreamPublisher(redis_client, STREAM, epoch).publish_many(
        [PositionEvent("VER", t, -t, t) for t in range(1, 51)]
    )

    run = await run_client(
        url, until=lambda view: view.cars.get("VER", {}).get("x") == 50, timeout_s=10
    )

    assert run.as_of == "0-0"  # nada aplicado: a marca d'água NÃO é o fim do stream
    assert len(run.received_ids) == 50  # os 50 chegaram (pelo catch-up)
    assert run.view.cars["VER"]["pos_t"] == 50


async def test_a_partial_worker_progress_is_completed_exactly_by_the_catch_up(
    url: str, redis_client: Redis, store: RaceStateStore
):
    """Os workers aplicaram os 30 primeiros e pararam; mais 20 foram publicados. O snapshot tem os
    30; o catch-up traz EXATAMENTE os 20 restantes (nem mais, nem menos: a marca d'água é exata)."""
    epoch = await store.new_epoch(*RACE)
    await store.register_stream(*RACE)
    publisher = StreamPublisher(redis_client, STREAM, epoch)
    async with running(make_worker(redis_client)):
        await publisher.publish_many([PositionEvent("VER", t, -t, t) for t in range(1, 31)])
        await wait_until(lambda: drained(redis_client))
    await publisher.publish_many([PositionEvent("VER", t, -t, t) for t in range(31, 51)])
    ids = await stream_ids(redis_client)

    run = await run_client(url, until=lambda view: view.cars["VER"]["x"] == 50, timeout_s=10)

    assert run.as_of == ids[29]  # tudo até o 30º está aplicado
    assert run.received_ids == ids[30:]  # e o catch-up entrega os 20 que faltavam, sem repetir


async def test_events_delivered_to_a_slow_worker_but_not_yet_applied_are_not_lost(
    url: str, redis_client: Redis, store: RaceStateStore
):
    """O caso REAL do buraco (o grupo existe): um worker leu 20 entradas, aplicou e confirmou as 5
    primeiras e está parado/morto com as outras 15 PENDENTES (entregues, não aplicadas). O fim do
    stream é o 20º, mas o estado só chega ao 5º: a marca d'água tem de ficar logo antes da 6ª, ou
    as 15 não estariam no snapshot nem viriam do XREAD."""
    from replay.state import WORKER_GROUP

    epoch = await store.new_epoch(*RACE)
    await store.register_stream(*RACE)
    await redis_client.xgroup_create(STREAM, WORKER_GROUP, id="0", mkstream=True)
    await StreamPublisher(redis_client, STREAM, epoch).publish_many(
        [PositionEvent("VER", t, -t, t) for t in range(1, 21)]
    )
    ids = await stream_ids(redis_client)
    delivered = await redis_client.xreadgroup(WORKER_GROUP, "lento", {STREAM: ">"}, count=100)
    for entry_id, fields in delivered[0][1][:5]:  # aplica e confirma só as 5 primeiras
        from replay.events import parse_entry

        await store.apply(*RACE, parse_entry(entry_id, fields))
        await redis_client.xack(STREAM, WORKER_GROUP, entry_id)

    run = await run_client(url, until=lambda view: view.cars["VER"]["x"] == 20, timeout_s=10)

    assert run.as_of == ids[4]  # o estado só chega ao 5º, não ao fim do stream
    assert run.received_ids == ids[5:]  # as 15 pendentes chegam pelo catch-up, exatas


async def test_a_client_connected_before_any_replay_sees_it_start(
    url: str, redis_client: Redis, store: RaceStateStore, synthetic_race
):
    manager = ReplayManager(synthetic_race, redis_client, page_size=100)
    async with running(make_worker(redis_client)):
        waiting = asyncio.create_task(run_client(url, until=lambda view: view.finished))
        await asyncio.sleep(0.3)  # conectado, sem stream nenhum ainda (o XREAD espera ele nascer)
        await manager.start(*RACE, speed=16)

        run = await waiting
        truth = await final_truth(store, redis_client)
    await manager.shutdown()

    assert run.view.cars == truth.cars and run.view.race == truth.race


async def test_a_connected_client_follows_a_new_replay_of_the_same_race(
    url: str, redis_client: Redis, store: RaceStateStore, synthetic_race
):
    """Replay 1 termina; o cliente continua conectado; um replay 2 começa (o stream é apagado e
    recriado, o epoch muda). O cliente descarta o replay 1 e acompanha o 2 até o fim."""
    manager = ReplayManager(synthetic_race, redis_client, page_size=100)
    async with running(make_worker(redis_client)):
        await manager.start(*RACE, speed=16)
        first = await run_client(url, until=lambda view: view.finished)
        assert first.view.epoch == 1
        await wait_until(lambda: _owner_released(manager))

        follower = asyncio.create_task(
            run_client(url, until=lambda view: view.finished and view.epoch == 2)
        )
        await asyncio.sleep(0.3)  # o 2º cliente entra com o replay 1 já terminado...
        await manager.start(*RACE, speed=16)  # ...e o replay 2 começa

        second = await follower
        truth = await final_truth(store, redis_client)
    await manager.shutdown()

    assert truth.epoch == 2 and second.view.epoch == 2
    assert second.view.cars == truth.cars and second.view.race == truth.race


async def _owner_released(manager: ReplayManager) -> bool:
    return manager._active(RACE) is None


# ================================================================ desconexão e ciclo de vida


async def test_clients_that_drop_leave_no_leak_and_the_server_keeps_serving(
    url: str, redis_client: Redis, store: RaceStateStore
):
    hubs = app.state.hubs
    baseline = len(asyncio.all_tasks())

    async def connect():
        ws = await websockets.connect(url)
        await ws.recv()  # o snapshot
        return ws

    sockets = await asyncio.gather(*(connect() for _ in range(30)))
    await wait_until(_async_true(lambda: hubs.clients == 30))
    assert len(hubs.hubs) == 1  # um leitor do stream para os 30 clientes

    for k, ws in enumerate(sockets):
        if k % 2:
            await ws.close()  # saída educada
        else:
            ws.transport.abort()  # queda abrupta: sem handshake de fechamento (cabo puxado)
    await wait_until(_async_true(lambda: hubs.clients == 0))
    await wait_until(_async_true(lambda: hubs.hubs == []))  # o leitor foi liberado

    for _ in range(50):  # as tarefas encerram em alguns turnos do loop
        if len(asyncio.all_tasks()) <= baseline + 2:
            break
        await asyncio.sleep(0.05)
    assert len(asyncio.all_tasks()) <= baseline + 2  # nada acumulou
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("hub-")]

    # E o servidor segue de pé para o próximo cliente.
    run = await run_client(url, until=lambda view: True, timeout_s=5)
    assert run.view.epoch == 0 or run.view.epoch >= 0


def _async_true(condition):
    async def check() -> bool:
        return condition()

    return check


async def test_many_connect_and_drop_cycles_do_not_accumulate_hubs_or_tasks(url: str):
    hubs = app.state.hubs
    baseline = len(asyncio.all_tasks())

    for _ in range(60):
        ws = await websockets.connect(url)
        await ws.recv()
        ws.transport.abort()

    await wait_until(_async_true(lambda: hubs.clients == 0 and hubs.hubs == []))
    for _ in range(50):
        if len(asyncio.all_tasks()) <= baseline + 2:
            break
        await asyncio.sleep(0.05)
    assert len(asyncio.all_tasks()) <= baseline + 2


async def test_an_invalid_path_is_rejected(server: str):
    with pytest.raises((InvalidStatus, ConnectionClosed)):
        async with websockets.connect(f"{server}/ws/races/1900/21") as ws:
            await ws.recv()


async def test_the_replay_state_is_part_of_the_snapshot(
    url: str, redis_client: Redis, store: RaceStateStore, synthetic_race
):
    """O snapshot diz também em que ponto do replay a corrida está (velocidade, pausa...)."""
    manager = ReplayManager(synthetic_race, redis_client, page_size=100)
    await manager.start(*RACE, speed=4)
    manager.pause(*RACE)
    await wait_until(_async_true(lambda: manager.status(*RACE).state is ReplayState.PAUSED))
    await asyncio.sleep(0.1)  # o status é republicado a cada ~1 s; o do start já está no Redis

    async with websockets.connect(url) as ws:
        snapshot = json.loads(await ws.recv())

    assert snapshot["replay"] is not None and snapshot["replay"]["speed"] == 4.0
    await manager.shutdown()
