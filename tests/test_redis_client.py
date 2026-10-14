"""Testes UNITÁRIOS da configuração e da fábrica do cliente Redis (não precisam de Redis).

O que fala com um Redis de verdade fica em tests/integration/test_redis.py.
"""

import pytest

from app.config import RedisSettings
from app.redis_client import MAX_BLOCK_MS, SOCKET_TIMEOUT_S, create_redis


@pytest.fixture(autouse=True)
def _no_redis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isola o teste do ambiente: REDIS_* exportadas no shell ou no CI não mudam o resultado.
    for name in ("REDIS_HOST", "REDIS_PORT", "REDIS_DB"):
        monkeypatch.delenv(name, raising=False)


def test_settings_defaults_point_to_a_local_redis():
    # _env_file=None: ignora um .env real na raiz do projeto.
    settings = RedisSettings(_env_file=None)

    assert (settings.host, settings.port, settings.db) == ("localhost", 6379, 0)
    assert settings.url == "redis://localhost:6379/0"


def test_settings_are_read_from_REDIS_prefixed_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REDIS_HOST", "redis")  # como o docker-compose.yml define para o app
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "3")

    settings = RedisSettings(_env_file=None)

    assert settings.url == "redis://redis:6380/3"


def test_redis_settings_do_not_require_postgres_credentials():
    """O worker só fala com o Redis: instanciar RedisSettings não pode exigir POSTGRES_PASSWORD
    (que `Settings()` exige e para a qual não há default)."""
    RedisSettings(_env_file=None)  # não levanta ValidationError


def test_create_redis_uses_the_settings_and_decodes_responses():
    client = create_redis(RedisSettings(_env_file=None, host="redis", port=6380, db=2))

    kwargs = client.connection_pool.connection_kwargs
    assert (kwargs["host"], kwargs["port"], kwargs["db"]) == ("redis", 6380, 2)
    # Eventos do stream são JSON em texto: queremos str, não bytes.
    assert kwargs["decode_responses"] is True


def test_blocking_reads_must_end_before_the_socket_timeout():
    """Regra documentada em app/redis_client.py: se o BLOCK de um XREADGROUP fosse maior que o
    timeout do socket, a leitura estouraria TimeoutError numa conexão perfeitamente saudável."""
    assert MAX_BLOCK_MS / 1000 < SOCKET_TIMEOUT_S
