"""O armazém dos resultados de estratégia contra o Redis REAL: o script Lua (as travas de epoch e de
"primeiro vence", e a atomicidade hash + stream), a expiração e o filtro do replay atual."""

import asyncio

from redis.asyncio import Redis

from replay.state import epoch_key
from strategy.service.results import (
    PutOutcome,
    ResultStore,
    events_key,
    result_key,
)

Y, R = 2024, 10


def doc(lap: int, tag: str = "a") -> str:
    return f'{{"lap":{lap},"tag":"{tag}"}}'


async def put(store: ResultStore, lap: int, *, epoch: int = 1, tag: str = "a") -> PutOutcome:
    return await store.put(Y, R, epoch=epoch, lap=lap, payload=doc(lap, tag), version=1)


async def test_a_result_is_readable_by_lap_and_as_the_latest(redis_client: Redis):
    store = ResultStore(redis_client)

    assert await put(store, 10) is PutOutcome.WRITTEN
    assert await put(store, 11) is PutOutcome.WRITTEN

    latest = await store.get(Y, R)
    assert latest is not None and (latest.lap, latest.latest, latest.epoch) == (11, 11, 1)
    assert latest.payload == doc(11)
    older = await store.get(Y, R, 10)
    assert older is not None and older.payload == doc(10) and older.latest == 11
    assert await store.get(Y, R, 12) is None  # uma volta que não foi calculada: nada


async def test_the_latest_lap_only_moves_forward(redis_client: Redis):
    store = ResultStore(redis_client)
    await put(store, 10)
    await put(store, 8)  # um cálculo atrasado, de uma volta anterior

    result = await store.get(Y, R)
    assert result is not None and result.latest == 10
    assert (await store.get(Y, R, 8)) is not None  # mas fica consultável


async def test_each_result_is_published_once_to_the_events_stream_with_the_same_document(
    redis_client: Redis,
):
    store = ResultStore(redis_client)

    await put(store, 10)

    ((_, fields),) = await redis_client.xrange(events_key(Y, R))
    assert fields == {"v": "1", "epoch": "1", "lap": "10", "data": doc(10)}


async def test_the_first_result_of_a_lap_wins_and_is_not_published_twice(redis_client: Redis):
    store = ResultStore(redis_client)
    await put(store, 10, tag="primeiro")

    assert await put(store, 10, tag="segundo") is PutOutcome.EXISTS

    result = await store.get(Y, R, 10)
    assert result is not None and "primeiro" in result.payload
    assert await redis_client.xlen(events_key(Y, R)) == 1  # o cliente não vê a volta duas vezes


async def test_concurrent_writers_of_the_same_lap_produce_exactly_one_result(redis_client: Redis):
    """Dois consumidores (o zumbi e o novo líder) calculando a mesma volta ao mesmo tempo."""
    stores = [ResultStore(redis_client) for _ in range(10)]

    outcomes = await asyncio.gather(*(put(s, 10, tag=str(i)) for i, s in enumerate(stores)))

    assert outcomes.count(PutOutcome.WRITTEN) == 1 and outcomes.count(PutOutcome.EXISTS) == 9
    assert await redis_client.xlen(events_key(Y, R)) == 1


async def test_a_result_from_an_older_replay_is_refused(redis_client: Redis):
    store = ResultStore(redis_client)
    await put(store, 10, epoch=3)

    assert await put(store, 11, epoch=2) is PutOutcome.FENCED  # um zumbi do replay anterior

    assert await store.get(Y, R, 11) is None
    assert await redis_client.xlen(events_key(Y, R)) == 1


async def test_a_newer_replay_starts_the_results_over(redis_client: Redis):
    store = ResultStore(redis_client)
    await put(store, 10, epoch=1)
    await put(store, 11, epoch=1)

    assert await put(store, 5, epoch=2) is PutOutcome.WRITTEN

    assert await store.get(Y, R, 10) is None and await store.get(Y, R, 11) is None
    result = await store.get(Y, R)
    assert result is not None and (result.lap, result.latest, result.epoch) == (5, 5, 2)


async def test_get_hides_the_results_of_a_replay_that_already_passed(redis_client: Redis):
    """Um replay novo começou (o epoch subiu) e ainda não calculou nada: a API não mostra a
    estratégia da corrida passada como se fosse a de agora."""
    store = ResultStore(redis_client)
    await put(store, 10, epoch=1)
    await redis_client.set(epoch_key(Y, R), 2)  # o `start` do replay novo

    assert await store.get(Y, R) is None and await store.get(Y, R, 10) is None

    await put(store, 5, epoch=2)
    result = await store.get(Y, R)
    assert result is not None and result.epoch == 2


async def test_has_is_per_epoch_and_per_lap(redis_client: Redis):
    store = ResultStore(redis_client)
    await put(store, 10, epoch=2)

    assert await store.has(Y, R, epoch=2, lap=10)
    assert not await store.has(Y, R, epoch=2, lap=11)
    assert not await store.has(Y, R, epoch=3, lap=10)  # outro replay: a volta 10 dele não existe
    assert not await store.has(Y, R, epoch=1, lap=10)


async def test_the_hash_expires_and_the_events_stream_is_bounded(redis_client: Redis):
    store = ResultStore(redis_client, ttl_s=3600, events_maxlen=100)

    for lap in range(1, 400):
        await put(store, lap)

    ttl = await redis_client.ttl(result_key(Y, R))
    assert 0 < ttl <= 3600
    assert 100 <= await redis_client.xlen(events_key(Y, R)) <= 200  # `~`: aparado em blocos
    assert (await store.get(Y, R, 1)) is not None  # o hash é o histórico: nada some dele
