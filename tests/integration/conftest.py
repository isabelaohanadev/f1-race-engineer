"""Fixtures dos testes de INTEGRAÇÃO: usam o TimescaleDB e o Redis reais do Compose.

Como funciona (e por que é seguro rodar na sua máquina):
* Cada execução cria um banco DESCARTÁVEL (`f1_test_<hex>`) no mesmo servidor do Compose, roda
  `alembic upgrade head` nele (a migration REAL, com hypertable) e o apaga no final. Seus dados de
  desenvolvimento (o banco f1_race_engineer) nunca são tocados.
* No Redis, a fixture `redis_client` usa o banco lógico 15 (a aplicação usa o 0) e o esvazia antes e
  depois de cada teste. O FLUSHDB só atinge o 15, então streams de desenvolvimento ficam intactos.
* Se o servidor não estiver acessível, os testes são PULADOS (skip), a menos que
  REQUIRE_INTEGRATION_DB=1: no CI, "banco/Redis fora do ar" deve FALHAR, não passar em silêncio.
* Todos os testes desta pasta recebem o marker `integration` automaticamente:
      pytest -m "not integration"   # pula todos eles
      pytest -m integration         # roda só eles
"""

import asyncio
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import uvicorn
from pydantic import ValidationError
from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.config import RedisSettings, Settings, get_redis_settings
from app.deps import get_redis
from app.main import app
from app.redis_client import create_redis
from replay.state import RaceStateStore
from tests.integration.helpers import RACE
from tests.replay.conftest import ALB, NOR, VER, at, seed_race

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent

# Banco lógico do Redis só dos testes (a aplicação usa o 0). Cada teste começa e termina com ele
# vazio.
TEST_REDIS_DB = 15


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Este hook recebe os itens da sessão INTEIRA; filtramos só os desta pasta.
    for item in items:
        if HERE in item.path.parents:
            item.add_marker(pytest.mark.integration)


def _skip_or_fail(reason: str) -> None:
    if os.environ.get("REQUIRE_INTEGRATION_DB") == "1":
        pytest.fail(f"REQUIRE_INTEGRATION_DB=1, mas: {reason}")
    pytest.skip(reason)


def _alembic(db_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    # Subprocesso: isola o estado global do Alembic/logging e o cache de get_settings(), e roda
    # exatamente o mesmo comando que o serviço `migrate` do Compose.
    env = {**os.environ, "POSTGRES_DB": db_name}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="session")
def migrated_db_name() -> Iterator[str]:
    try:
        settings = Settings()
    except ValidationError:
        _skip_or_fail("POSTGRES_PASSWORD não definida (copie .env.example para .env)")

    admin = create_engine(
        settings.database_url,
        isolation_level="AUTOCOMMIT",  # CREATE/DROP DATABASE não rodam dentro de transação
        connect_args={"connect_timeout": 3},
    )
    try:
        with admin.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        admin.dispose()
        _skip_or_fail(
            f"TimescaleDB inacessível em {settings.postgres_host}:{settings.postgres_port} "
            "(suba com: docker compose up -d db)"
        )

    db_name = f"f1_test_{uuid.uuid4().hex[:8]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    try:
        result = _alembic(db_name, "upgrade", "head")
        assert result.returncode == 0, f"alembic upgrade falhou:\n{result.stderr}"
        yield db_name
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def pg_engine(migrated_db_name: str) -> Iterator[Engine]:
    settings = Settings()
    url = settings.database_url.set(database=migrated_db_name)
    engine = create_engine(url, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture
def clean_pg_engine(pg_engine: Engine) -> Engine:
    """O engine do banco migrado, com as tabelas vazias (isolamento entre testes).
    O CASCADE leva junto entries, laps e position_samples (todas dependem de races/drivers)."""
    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE races, drivers, teams CASCADE"))
    return pg_engine


@pytest.fixture(scope="session")
def redis_test_settings() -> RedisSettings:
    """Config do Redis de teste (banco lógico 15), após provar UMA vez por sessão que ele responde.

    A sonda é síncrona, de 1 s e SEM retry de propósito: o cliente da aplicação (create_redis) tenta
    de novo com backoff, o que faria "Redis fora do ar" levar ~4 s por teste antes de pular/falhar.
    """
    settings = RedisSettings(db=TEST_REDIS_DB)
    probe = SyncRedis(
        host=settings.host,
        port=settings.port,
        socket_connect_timeout=1,
        retry=Retry(NoBackoff(), 0),
    )
    error: Exception | None = None
    try:
        probe.ping()
    except (RedisError, OSError) as exc:
        error = exc
    finally:
        probe.close()
    if error is not None:  # fora do `except`: a mensagem não sai encadeada ao traceback do driver
        _skip_or_fail(
            f"Redis inacessível em {settings.host}:{settings.port} ({type(error).__name__}); "
            "suba com: docker compose up -d redis"
        )
    return settings


@pytest.fixture
async def redis_client(redis_test_settings: RedisSettings) -> AsyncIterator[Redis]:
    """Cliente do Redis real, apontando para o banco lógico de testes, vazio a cada teste.

    Função-escopo (e não session) de propósito: o cliente assíncrono fica preso ao event loop que o
    usou primeiro, e o pytest-asyncio cria um loop novo por teste.
    """
    client = create_redis(redis_test_settings)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


# ------------------------------------------------------- servidor real e dados de streaming


@pytest.fixture
async def server(monkeypatch: pytest.MonkeyPatch, redis_test_settings) -> AsyncIterator[str]:
    """Um uvicorn de verdade, no MESMO loop do teste, no Redis de testes (banco 15)."""
    # O stub de Redis do conftest raiz (para os testes da API) não vale aqui: este servidor é real.
    app.dependency_overrides.pop(get_redis, None)
    monkeypatch.setenv("REDIS_DB", str(redis_test_settings.db))
    get_redis_settings.cache_clear()  # o lifespan lê a config daqui
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    instance = uvicorn.Server(config)
    task = asyncio.create_task(instance.serve())
    async with asyncio.timeout(10):
        while not instance.started:
            await asyncio.sleep(0.01)
    port = instance.servers[0].sockets[0].getsockname()[1]
    yield f"ws://127.0.0.1:{port}"
    instance.should_exit = True
    await asyncio.wait_for(task, 10)
    get_redis_settings.cache_clear()


@pytest.fixture
def url(server: str) -> str:
    return f"{server}/ws/races/{RACE[0]}/{RACE[1]}"


@pytest.fixture
def store(redis_client: Redis) -> RaceStateStore:
    return RaceStateStore(redis_client)


@pytest.fixture
def synthetic_race(engine: Engine) -> sessionmaker:
    """40 s de corrida, 3 pilotos com amostras a cada 220 ms (EMPATES de horário entre eles) e uma
    volta a cada 10 s: ~550 eventos, que a 16x levam ~2,5 s de relógio de verdade."""
    positions = [(d, at(i * 220), i, d) for i in range(182) for d in (VER, NOR, ALB)]
    laps = [
        (d, k, at(10_000 * k + 400 * d), 80_000 + k, "SOFT", d)
        for k in range(1, 5)
        for d in (VER, NOR, ALB)
    ]
    seed_race(engine, positions=positions, laps=laps, total_laps=4)
    return sessionmaker(engine, expire_on_commit=False)


@pytest.fixture(scope="session")
def race_rows():
    """A corrida sintética (14 carros, 40 voltas) dos testes do consumidor de estratégia."""
    from tests.integration.strategy_helpers import make_rows

    return make_rows()


@pytest.fixture
def consumer_subprocess(redis_test_settings, migrated_db_name: str):
    """O consumidor de verdade (`python -m strategy.service`) num processo à parte, apontando para o
    Redis de teste e para o banco descartável (vazio: as constantes saem dos padrões)."""
    env = {
        **os.environ,
        "POSTGRES_DB": migrated_db_name,
        "REDIS_DB": str(redis_test_settings.db),
        "STRATEGY_METRICS_PORT": "0",
        "STRATEGY_WAIT_S": "0.05",
        "STRATEGY_LOCK_TTL_MS": "800",
        "STRATEGY_N_SIMS": "300",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "strategy.service"],
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield process
    if process.poll() is None:
        process.kill()
    process.wait(10)
