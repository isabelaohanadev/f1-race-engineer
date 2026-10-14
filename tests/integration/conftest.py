"""Fixtures dos testes de INTEGRAÇÃO: usam o TimescaleDB real do Compose.

Como funciona (e por que é seguro rodar na sua máquina):
* Cada execução cria um banco DESCARTÁVEL (`f1_test_<hex>`) no mesmo servidor do Compose, roda
  `alembic upgrade head` nele (a migration REAL, com hypertable) e o apaga no final. Seus dados de
  desenvolvimento (o banco f1_race_engineer) nunca são tocados.
* Se o servidor não estiver acessível, os testes são PULADOS (skip), a menos que
  REQUIRE_INTEGRATION_DB=1: no CI, "banco fora do ar" deve FALHAR, não passar em silêncio.
* Todos os testes desta pasta recebem o marker `integration` automaticamente:
      pytest -m "not integration"   # pula todos eles
      pytest -m integration         # roda só eles
"""

import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from app.config import Settings

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


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
