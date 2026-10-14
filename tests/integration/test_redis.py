"""Testes contra o Redis REAL. Garantem, antes de construirmos em cima, que o servidor do Compose/CI
conecta e suporta as primitivas que a Fase 2 usa: Streams e consumer groups."""

from redis.asyncio import Redis


async def test_redis_answers_ping(redis_client: Redis):
    assert await redis_client.ping() is True


async def test_stream_and_consumer_group_roundtrip(redis_client: Redis):
    """XADD publica, o grupo entrega UMA vez a um consumidor, XACK confirma. É o ciclo inteiro que
    o motor de replay (XADD) e o worker (XREADGROUP + XACK) vão fazer."""
    stream, group = "race:9999:1", "workers"
    await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
    await redis_client.xadd(stream, {"type": "position", "driver": "VER"})

    reply = await redis_client.xreadgroup(group, "worker-1", {stream: ">"}, count=10)

    [[name, messages]] = reply
    assert name == stream
    [(message_id, fields)] = messages
    assert fields == {"type": "position", "driver": "VER"}  # decode_responses=True: str, não bytes

    # Ainda pendente (entregue, mas não confirmada): é o que permite reprocessar se o worker cair.
    assert (await redis_client.xpending(stream, group))["pending"] == 1
    assert await redis_client.xack(stream, group, message_id) == 1
    assert (await redis_client.xpending(stream, group))["pending"] == 0


async def test_test_database_is_isolated_from_the_application_database(redis_client: Redis):
    """O banco lógico dos testes não é o 0, onde fica a aplicação (e seus streams de dev)."""
    connection = redis_client.connection_pool.connection_kwargs
    assert connection["db"] == 15
