"""Publisher, lock e /health contra o Redis REAL (o que fakeredis não garante: MULTI/EXEC, o corte
aproximado do MAXLEN, os scripts Lua e a expiração por TTL são comportamentos do servidor)."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.config import RedisSettings
from app.deps import get_redis
from app.main import app
from replay.events import LapCompletedEvent, PositionEvent, RaceStatusEvent, parse_event
from replay.lock import ReplayLock
from replay.publisher import STREAM_MAXLEN, StreamPublisher

STREAM = "race:9999:1"


# ---------------------------------------------------------------- publisher


async def test_published_events_are_readable_by_a_consumer_and_parse_back(redis_client: Redis):
    events = [
        RaceStatusEvent("started", 1, 1_000, 71),
        PositionEvent("VER", -120, 3400, 1_000),
        LapCompletedEvent("NOR", 1, 1_500, 87_134, "SOFT", 3),
        LapCompletedEvent("ALB", 1, 1_600),  # campos nulos omitidos
    ]

    await StreamPublisher(redis_client, STREAM).publish_many(events)

    entries = await redis_client.xrange(STREAM)
    assert [parse_event(fields) for _, fields in entries] == events  # mesma ordem, nada perdido
    ids = [entry_id for entry_id, _ in entries]
    assert ids == sorted(ids, key=lambda i: tuple(map(int, i.split("-"))))  # IDs crescentes


async def test_maxlen_keeps_the_stream_bounded_and_drops_only_the_oldest(redis_client: Redis):
    publisher = StreamPublisher(redis_client, STREAM, maxlen=100)
    events = [PositionEvent("VER", i, i, i) for i in range(2000)]

    for start in range(0, len(events), 200):
        await publisher.publish_many(events[start : start + 200])

    length = await redis_client.xlen(STREAM)
    # `~` apara em blocos inteiros: nunca abaixo do limite, e passa dele por menos de um bloco
    # (~100 entradas). Exato seria MAXLEN = 100, mas custaria um corte por XADD.
    assert 100 <= length <= 200
    newest = await redis_client.xrevrange(STREAM, count=1)
    assert parse_event(newest[0][1]) == events[-1]  # o mais novo sobrevive
    oldest = await redis_client.xrange(STREAM, count=1)
    assert parse_event(oldest[0][1]).t == 2000 - length  # o que sobra é a CAUDA contínua


async def test_a_hundred_thousand_entries_use_a_few_megabytes(redis_client: Redis):
    """Guarda do formato: base do argumento de que STREAM_MAXLEN (100 mil) não é limitado pela
    memória (medido: 4,2 MB). Se alguém trocar os campos planos por blobs maiores, quebra aqui."""
    publisher = StreamPublisher(redis_client, STREAM, maxlen=STREAM_MAXLEN)
    events = [PositionEvent("VER", i % 9000, -(i % 9000), i) for i in range(STREAM_MAXLEN)]

    for start in range(0, len(events), 500):
        await publisher.publish_many(events[start : start + 500])

    assert await redis_client.xlen(STREAM) == STREAM_MAXLEN
    assert await redis_client.memory_usage(STREAM) < 15 * 1024 * 1024  # 3x o medido


async def test_reset_deletes_the_stream_together_with_its_consumer_groups(redis_client: Redis):
    publisher = StreamPublisher(redis_client, STREAM)
    await publisher.publish_many([PositionEvent("VER", 1, 1, 1)])
    await redis_client.xgroup_create(STREAM, "workers", id="0")

    await publisher.reset()

    assert await redis_client.exists(STREAM) == 0
    # É isto que o worker enxerga quando um novo replay começa: o grupo sumiu (NOGROUP) e ele
    # precisa recriá-lo. Ver a docstring de StreamPublisher.reset().
    with pytest.raises(ResponseError):
        await redis_client.xinfo_groups(STREAM)


# ---------------------------------------------------------------- lock


def lock_for(redis: Redis, ttl_ms: int = 10_000) -> ReplayLock:
    return ReplayLock(redis, "replay:lock:9999:1", ttl_ms)


async def test_only_one_owner_holds_the_lock_at_a_time(redis_client: Redis):
    first, second = lock_for(redis_client), lock_for(redis_client)

    assert await first.acquire() is True
    assert await second.acquire() is False  # SET NX: a 2ª tentativa perde
    assert 0 < await redis_client.pttl("replay:lock:9999:1") <= 10_000  # tem TTL: nunca é eterno


async def test_only_the_owner_can_release_or_renew(redis_client: Redis):
    owner, intruder = lock_for(redis_client), lock_for(redis_client)
    await owner.acquire()

    await intruder.release()  # tentativa de apagar o lock dos outros: nada acontece
    assert await intruder.renew() is False
    assert await redis_client.exists("replay:lock:9999:1") == 1

    assert await owner.renew() is True
    await owner.release()
    assert await redis_client.exists("replay:lock:9999:1") == 0
    assert await intruder.acquire() is True  # liberado de verdade


async def test_renew_pushes_the_expiration_forward(redis_client: Redis):
    lock = lock_for(redis_client, ttl_ms=10_000)
    await lock.acquire()
    await redis_client.pexpire("replay:lock:9999:1", 500)  # "quase expirando"

    assert await lock.renew() is True

    assert await redis_client.pttl("replay:lock:9999:1") > 5_000


async def test_an_abandoned_lock_expires_by_itself(redis_client: Redis):
    """O dono morreu sem liberar: depois do TTL a corrida volta a poder ser iniciada. (Único teste
    com espera de verdade: o TTL é medido pelo SERVIDOR, o relógio falso não o alcança.)"""
    await lock_for(redis_client, ttl_ms=100).acquire()
    assert await lock_for(redis_client).acquire() is False

    await asyncio.sleep(0.25)

    assert await lock_for(redis_client).acquire() is True


# ---------------------------------------------------------------- /health com o Redis real


def test_health_reports_redis_ok_against_the_real_redis(client: TestClient, redis_test_settings):
    app.dependency_overrides.pop(get_redis)  # sem o stub: usa o cliente real criado no lifespan

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["checks"]["redis"] == "ok"


def test_health_reports_redis_unavailable_against_a_dead_port_without_changing_the_status(
    client: TestClient,
):
    from app.redis_client import create_redis

    dead = create_redis(RedisSettings(_env_file=None, host="127.0.0.1", port=1))
    app.dependency_overrides[get_redis] = lambda: dead

    response = client.get("/health")

    assert response.status_code == 200  # banco ok => 200, mesmo com o Redis fora
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"] == {"process": "ok", "database": "ok", "redis": "unavailable"}
