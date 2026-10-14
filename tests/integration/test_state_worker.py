"""Estado compartilhado + worker contra o Redis REAL.

Prova as garantias que dependem do servidor: o script Lua (trava 1), o consumer group, o XAUTOCLAIM
de um worker morto, o NOGROUP quando um replay novo apaga o stream e a marca d'água dos eventos
aplicados (usada pela trava 2, no WebSocket).
"""

import asyncio
import random

import pytest
from redis.asyncio import Redis

from replay.events import (
    LapCompletedEvent,
    PositionEvent,
    RaceStatusEvent,
    ReplayEvent,
    StreamEntry,
    id_before,
)
from replay.publisher import StreamPublisher
from replay.state import STREAMS_REGISTRY_KEY, WORKER_GROUP, RaceStateStore, state_key
from tests.integration.helpers import RACE, STREAM, drained, make_worker, running, wait_until
from worker.consumer import StateWorker


def pos(driver: str, t: int) -> PositionEvent:
    return PositionEvent(driver, x=t, y=-t, t=t)  # x == t: o valor revela de qual evento veio


def entry(event: ReplayEvent, epoch: int = 1, entry_id: str = "1-0") -> StreamEntry:
    return StreamEntry(entry_id, epoch, event)


@pytest.fixture
def store(redis_client: Redis) -> RaceStateStore:
    return RaceStateStore(redis_client)


# ================================================================ TRAVA 1: o script Lua


async def test_an_older_event_never_overwrites_a_newer_one(store: RaceStateStore):
    assert await store.apply(*RACE, entry(pos("VER", 5))) is True  # chegou primeiro...
    assert await store.apply(*RACE, entry(pos("VER", 3))) is False  # ...o velho é ignorado

    car = (await store.snapshot(*RACE)).cars["VER"]
    assert (car["x"], car["y"], car["pos_t"]) == (5, -5, 5)  # o estado NÃO voltou no tempo


async def test_applying_the_same_event_twice_is_the_same_as_once(store: RaceStateStore):
    event = entry(LapCompletedEvent("VER", 3, 9_000, 87_134, "SOFT", 2))

    assert await store.apply(*RACE, event) is True
    before = await store.snapshot(*RACE)
    assert await store.apply(*RACE, event) is False  # versão igual não é "maior": ignorada

    assert await store.snapshot(*RACE) == before  # idempotente: reentregar é inofensivo


async def test_a_lap_without_time_removes_the_previous_laps_time(store: RaceStateStore):
    await store.apply(*RACE, entry(LapCompletedEvent("VER", 1, 1_000, 90_000, "SOFT", 1)))
    await store.apply(*RACE, entry(LapCompletedEvent("VER", 2, 2_000)))  # volta sem tempo válido

    car = (await store.snapshot(*RACE)).cars["VER"]
    assert car["lap"] == 2
    assert "lap_time_ms" not in car and "compound" not in car  # nada herdado da volta anterior


async def test_position_and_lap_versions_are_independent(store: RaceStateStore):
    await store.apply(*RACE, entry(LapCompletedEvent("VER", 1, 9_000)))  # volta "nova" primeiro

    # Uma posição com tempo MENOR que o da volta, mas a primeira posição do piloto: tem de entrar.
    assert await store.apply(*RACE, entry(pos("VER", 5_000))) is True

    assert (await store.snapshot(*RACE)).cars["VER"]["x"] == 5_000


def _random_events(rng: random.Random) -> tuple[list[ReplayEvent], dict]:
    """~190 eventos de 3 pilotos + status, e o estado final ESPERADO (o do maior tempo de cada)."""
    events: list[ReplayEvent] = []
    for driver in ("VER", "NOR", "HAM"):
        times = rng.sample(range(1_000, 100_000), 40)
        events += [pos(driver, t) for t in times]
        events += [
            LapCompletedEvent(
                driver, lap, 10_000 * lap + (3 if driver == "NOR" else 0), 80_000 + lap
            )
            for lap in range(1, 6)
        ]
    events += [
        RaceStatusEvent("started", 1, 1_000, 5),
        RaceStatusEvent("running", 2, 10_000, 5),
        RaceStatusEvent("running", 3, 20_000, 5),
        RaceStatusEvent("finished", 5, 50_000, 5),
    ]
    expected = {
        "x": {
            d: max(e.t for e in events if isinstance(e, PositionEvent) and e.driver == d)
            for d in ("VER", "NOR", "HAM")
        },
        "lap": 5,
        "state": "finished",
    }
    return events, expected


def _check(snapshot, expected) -> None:
    for driver, x in expected["x"].items():
        assert snapshot.cars[driver]["x"] == x  # a posição do evento de MAIOR tempo
        assert snapshot.cars[driver]["lap"] == expected["lap"]
        assert snapshot.cars[driver]["lap_time_ms"] == 80_005
    assert snapshot.race["state"] == "finished"  # o `finished` venceu o resto, em qualquer ordem
    # VER e HAM empatam em voltas e no instante (desempate por código); o NOR cruzou 3 ms depois.
    assert snapshot.classification == ["HAM", "VER", "NOR"]


@pytest.mark.parametrize("seed", range(25))
async def test_the_final_state_is_the_same_for_any_processing_order(
    store: RaceStateStore, redis_client: Redis, seed: int
):
    rng = random.Random(seed)
    events, expected = _random_events(rng)
    rng.shuffle(events)  # ordem de processamento arbitrária

    for event in events:
        await store.apply(*RACE, entry(event))

    _check(await store.snapshot(*RACE), expected)


@pytest.mark.parametrize("seed", range(10))
async def test_concurrent_writers_converge_to_the_same_state(store: RaceStateStore, seed: int):
    """Dois "workers" gravando AO MESMO TEMPO (cada um com metade dos eventos, em ordem aleatória).
    A atomicidade do Lua evita a corrida ler-comparar-gravar entre eles."""
    rng = random.Random(1_000 + seed)
    events, expected = _random_events(rng)
    rng.shuffle(events)
    half = len(events) // 2

    async def writer(chunk: list[ReplayEvent]) -> None:
        for event in chunk:
            await store.apply(*RACE, entry(event))

    await asyncio.gather(writer(events[:half]), writer(events[half:]))

    _check(await store.snapshot(*RACE), expected)


# ================================================================ EPOCH: um número por replay


async def test_a_new_replay_overwrites_the_old_state_even_with_smaller_times(store: RaceStateStore):
    await store.apply(*RACE, entry(pos("VER", 90_000), epoch=1))  # o replay 1 foi até o fim
    await store.apply(*RACE, entry(pos("NOR", 80_000), epoch=1))
    await store.new_epoch(*RACE)  # epoch 1
    epoch = await store.new_epoch(*RACE)  # epoch 2: o replay novo começou

    assert (
        await store.apply(*RACE, entry(pos("VER", 5), epoch=epoch)) is True
    )  # t MENOR que o velho

    snapshot = await store.snapshot(*RACE)
    assert snapshot.epoch == 2
    assert snapshot.cars == {"VER": {"x": 5, "y": -5, "pos_t": 5}}  # o NOR do replay 1 sumiu


async def test_a_late_event_from_the_old_replay_is_ignored(store: RaceStateStore):
    await store.apply(*RACE, entry(pos("VER", 5), epoch=2))

    assert (
        await store.apply(*RACE, entry(pos("VER", 999_999), epoch=1)) is False
    )  # t gigante, replay velho

    assert (await store.snapshot(*RACE)).cars["VER"]["x"] == 5


async def test_the_snapshot_is_empty_until_the_new_replay_writes_something(store: RaceStateStore):
    await store.apply(*RACE, entry(pos("VER", 90_000), epoch=1))
    await store.new_epoch(*RACE)
    await store.new_epoch(*RACE)  # replay 2 anunciado, ainda sem nenhum evento aplicado

    snapshot = await store.snapshot(*RACE)

    assert snapshot.epoch == 2 and snapshot.cars == {}  # não mostra a corrida velha como a atual


# ================================================================ o worker (consumer group)


async def publish(redis: Redis, events: list[ReplayEvent], epoch: int = 1) -> None:
    for start in range(0, len(events), 100):
        await StreamPublisher(redis, STREAM, epoch).publish_many(events[start : start + 100])


async def test_two_workers_in_one_group_split_the_events_and_out_of_order_apply_is_harmless(
    redis_client: Redis, store: RaceStateStore
):
    """O teste que prova a trava 1 de ponta a ponta. 200 eventos do MESMO piloto no stream; dois
    workers no mesmo grupo. O segundo é LENTO: lê um lote e só o aplica 300 ms depois, quando o
    rápido já aplicou todos os eventos mais novos. Sem a trava, esse lote velho sobrescreveria o
    estado final com posições antigas."""

    class SlowWorker(StateWorker):
        async def _process(self, stream, messages):
            await asyncio.sleep(0.3)  # o lote já foi LIDO (entregue a este worker); aplica atrasado
            await super()._process(stream, messages)

    await store.register_stream(*RACE)
    await publish(redis_client, [pos("VER", t) for t in range(1, 201)])
    fast = make_worker(redis_client, consumer="fast", batch_size=5)
    slow = SlowWorker(
        redis_client,
        consumer="slow",
        batch_size=5,
        block_ms=100,
        discover_interval_s=0.05,
        claim_interval_s=0.05,
        claim_min_idle_ms=60_000,
    )

    async with running(slow), running(fast):
        await wait_until(lambda: drained(redis_client))

    car = (await store.snapshot(*RACE)).cars["VER"]
    assert (car["x"], car["pos_t"]) == (200, 200)  # o evento de MAIOR tempo, não o último aplicado
    # Cada entrada foi processada UMA vez, por UM worker (o grupo reparte, não duplica)...
    total = sum(w.stats.applied + w.stats.ignored for w in (fast, slow))
    assert total == 200
    # ...e a inversão de ordem ACONTECEU de verdade: o lote atrasado do lento foi descartado.
    assert slow.stats.ignored > 0
    assert (await redis_client.xpending(STREAM, WORKER_GROUP))["pending"] == 0


async def test_a_worker_that_dies_with_pending_messages_is_rescued_by_xautoclaim(
    redis_client: Redis, store: RaceStateStore
):
    await store.register_stream(*RACE)
    await redis_client.xgroup_create(STREAM, WORKER_GROUP, id="0", mkstream=True)
    await publish(redis_client, [pos("VER", t) for t in range(1, 51)])
    # O worker "dead" lê as 50 entradas e MORRE antes de aplicar/confirmar: ficam pendentes.
    delivered = await redis_client.xreadgroup(WORKER_GROUP, "dead", {STREAM: ">"}, count=100)
    assert len(delivered[0][1]) == 50
    assert (await redis_client.xpending(STREAM, WORKER_GROUP))["pending"] == 50
    assert (await store.snapshot(*RACE)).cars == {}  # nada aplicado ainda

    # Um worker saudável, com ociosidade mínima de 100 ms, resgata e aplica tudo.
    rescuer = make_worker(redis_client, consumer="alive", claim_min_idle_ms=100)
    async with running(rescuer):
        await wait_until(lambda: drained(redis_client))

    assert rescuer.stats.claimed == 50
    assert (await store.snapshot(*RACE)).cars["VER"]["x"] == 50
    assert (await redis_client.xpending(STREAM, WORKER_GROUP))["pending"] == 0


async def test_a_worker_does_not_steal_messages_that_are_not_idle_long_enough(
    redis_client: Redis, store: RaceStateStore
):
    """Outro worker pode estar só no meio do lote: antes de `claim_min_idle_ms` não se rouba."""
    await store.register_stream(*RACE)
    await redis_client.xgroup_create(STREAM, WORKER_GROUP, id="0", mkstream=True)
    await publish(redis_client, [pos("VER", t) for t in range(1, 11)])
    await redis_client.xreadgroup(WORKER_GROUP, "busy", {STREAM: ">"}, count=100)

    worker = make_worker(redis_client, consumer="alive", claim_min_idle_ms=60_000)
    async with running(worker):
        await asyncio.sleep(0.4)  # vários ciclos de claim

    assert worker.stats.claimed == 0
    assert (await redis_client.xpending(STREAM, WORKER_GROUP))["pending"] == 10  # intactas


async def test_a_new_replay_deleting_the_stream_does_not_break_the_workers(
    redis_client: Redis, store: RaceStateStore
):
    """O início de um replay faz DEL do stream, que apaga o consumer group (NOGROUP no worker)."""
    await store.register_stream(*RACE)
    worker = make_worker(redis_client)
    async with running(worker):
        epoch1 = await store.new_epoch(*RACE)
        await publish(redis_client, [pos("VER", t) for t in range(1, 6)], epoch1)
        await wait_until(lambda: drained(redis_client))

        publisher = StreamPublisher(redis_client, STREAM, epoch=2)
        await store.new_epoch(*RACE)
        await publisher.reset()  # DEL: o grupo some, com o worker bloqueado no XREADGROUP
        await publisher.publish_many([pos("NOR", 7)])

        async def second_replay_applied() -> bool:
            return (await store.snapshot(*RACE)).epoch == 2 and "NOR" in (
                await store.snapshot(*RACE)
            ).cars

        await wait_until(second_replay_applied)

    snapshot = await store.snapshot(*RACE)
    assert snapshot.epoch == 2 and set(snapshot.cars) == {"NOR"}  # o VER do replay 1 já não existe


async def test_an_unreadable_entry_is_acked_and_skipped_not_retried_forever(
    redis_client: Redis, store: RaceStateStore
):
    await store.register_stream(*RACE)
    await redis_client.xadd(STREAM, {"type": "position", "driver": "VER"})  # sem x, y, t, epoch
    await publish(redis_client, [pos("VER", 5)])

    worker = make_worker(redis_client)
    async with running(worker):
        await wait_until(lambda: drained(redis_client))

    assert worker.stats.malformed == 1 and worker.stats.applied == 1
    assert (await redis_client.xpending(STREAM, WORKER_GROUP))["pending"] == 0  # não ficou "veneno"


async def test_the_worker_discovers_streams_registered_after_it_started(
    redis_client: Redis, store: RaceStateStore
):
    worker = make_worker(redis_client)
    async with running(worker):
        await asyncio.sleep(0.15)  # rodando, sem nenhum stream conhecido
        await store.register_stream(*RACE)  # o POST /start de algum lugar anunciou uma corrida
        await publish(redis_client, [pos("VER", 9)])
        await wait_until(lambda: drained(redis_client))

    assert (await store.snapshot(*RACE)).cars["VER"]["x"] == 9
    assert STREAM in await redis_client.smembers(STREAMS_REGISTRY_KEY)


async def test_the_state_hash_has_one_key_per_race(redis_client: Redis, store: RaceStateStore):
    await store.apply(*RACE, entry(pos("VER", 1)))
    await store.apply(*RACE, entry(pos("NOR", 2)))

    assert await redis_client.exists(state_key(*RACE)) == 1
    assert {"VER:x", "NOR:x", "epoch"} <= set(await redis_client.hkeys(state_key(*RACE)))


# ================================================================ a marca d'água (base da trava 2)


async def test_watermark_is_zero_when_no_worker_has_ever_read_the_stream(
    redis_client: Redis, store: RaceStateStore
):
    assert await store.applied_watermark(*RACE) == "0-0"  # nem existe o stream
    await publish(redis_client, [pos("VER", 1)])
    assert await store.applied_watermark(*RACE) == "0-0"  # existe, mas nenhum grupo leu


async def test_watermark_is_the_last_delivered_id_when_nothing_is_pending(
    redis_client: Redis, store: RaceStateStore
):
    await store.register_stream(*RACE)
    await publish(redis_client, [pos("VER", t) for t in range(1, 6)])
    async with running(make_worker(redis_client)):
        await wait_until(lambda: drained(redis_client))

    last = (await redis_client.xinfo_stream(STREAM))["last-generated-id"]
    assert await store.applied_watermark(*RACE) == last


async def test_watermark_stops_just_before_the_oldest_pending_entry(
    redis_client: Redis, store: RaceStateStore
):
    """Entregues mas NÃO confirmadas podem não estar no estado: a marca fica antes delas."""
    await redis_client.xgroup_create(STREAM, WORKER_GROUP, id="0", mkstream=True)
    await publish(redis_client, [pos("VER", t) for t in range(1, 11)])
    ids = [i for i, _ in await redis_client.xrange(STREAM)]
    delivered = await redis_client.xreadgroup(WORKER_GROUP, "w", {STREAM: ">"}, count=10)
    # Confirma as 4 primeiras (aplicadas); as 6 últimas ficam pendentes.
    await redis_client.xack(STREAM, WORKER_GROUP, *[i for i, _ in delivered[0][1][:4]])

    assert await store.applied_watermark(*RACE) == id_before(ids[4])  # logo antes da 5ª
