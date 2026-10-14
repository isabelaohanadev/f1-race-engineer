import time
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.config import RedisSettings
from app.deps import get_db, get_redis
from app.main import app
from app.redis_client import create_redis
from app.routers import health as health_module
from tests.conftest import StubRedis


def test_health_is_200_and_reports_each_check_when_everything_is_up(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"process": "ok", "database": "ok", "redis": "ok"},
    }


@pytest.fixture
def client_with_db_failing_on(request: pytest.FixtureRequest) -> Iterator[TestClient]:
    """Client cujo banco levanta `request.param` ao executar qualquer comando."""
    failing_session = Mock(spec=Session)
    failing_session.execute.side_effect = request.param

    def override() -> Iterator[Session]:
        yield failing_session

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "client_with_db_failing_on",
    [
        OperationalError(
            "SELECT 1", {}, Exception('connection to server at "10.0.0.5" failed: password=hunter2')
        )
    ],
    indirect=True,
)
def test_health_is_503_and_names_the_failing_dependency_when_database_is_down(
    client_with_db_failing_on: TestClient,
):
    response = client_with_db_failing_on.get("/health")

    assert response.status_code == 503
    # O corpo continua dizendo QUAL parte falhou: o processo segue vivo, só o banco caiu.
    assert response.json() == {
        "status": "degraded",
        "checks": {"process": "ok", "database": "unavailable", "redis": "ok"},
    }
    # Detalhes de infraestrutura nunca vazam na resposta (ficam só no log).
    assert "10.0.0.5" not in response.text
    assert "hunter2" not in response.text


def test_health_is_503_against_a_really_unreachable_postgres():
    """Sem mock: tenta conectar de verdade numa porta fechada.

    O teste acima prova a lógica com um erro fabricado; este prova que o erro REAL do driver
    (psycopg) é mesmo capturado pelo `except SQLAlchemyError` do endpoint. Não precisa de banco:
    a porta 1 recusa a conexão na hora.
    """
    engine = create_engine(
        "postgresql+psycopg://user:pw@127.0.0.1:1/db", connect_args={"connect_timeout": 1}
    )
    factory = sessionmaker(engine)

    def override() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    assert response.status_code == 503
    assert response.json()["checks"]["database"] == "unavailable"


# ---------------------------------------------------------------- Redis no /health


def test_health_stays_200_but_reports_degraded_when_only_redis_is_down(
    client: TestClient, redis_stub: StubRedis
):
    """Decisão de design: Redis fora NÃO muda o código HTTP. Só o streaming depende dele; as rotas
    REST seguem de pé, e o balanceador não deve tirar a instância do ar por isso."""
    redis_stub.error = RedisConnectionError(
        "connection to 10.0.0.9:6379 refused (password=hunter2)"
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "checks": {"process": "ok", "database": "ok", "redis": "unavailable"},
    }
    assert "10.0.0.9" not in response.text  # nenhum detalhe de infraestrutura na resposta
    assert "hunter2" not in response.text


def test_health_does_not_hang_when_redis_never_answers(
    client: TestClient, redis_stub: StubRedis, monkeypatch: pytest.MonkeyPatch
):
    redis_stub.hang = True
    monkeypatch.setattr(health_module, "REDIS_HEALTH_TIMEOUT_S", 0.05)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["checks"]["redis"] == "unavailable"


def test_health_is_503_when_database_is_down_even_if_redis_is_up(redis_stub: StubRedis):
    failing_session = Mock(spec=Session)
    failing_session.execute.side_effect = OperationalError("SELECT 1", {}, Exception("down"))

    def override() -> Iterator[Session]:
        yield failing_session

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 503
    assert response.json()["checks"] == {"process": "ok", "database": "unavailable", "redis": "ok"}


def test_health_is_fast_against_a_really_unreachable_redis(client: TestClient):
    """Sem stub: cliente REAL do redis-py numa porta fechada (a 1 recusa na hora). O retry com
    backoff do cliente leva ~4 s para desistir; o /health tem de responder muito antes disso."""
    real = create_redis(RedisSettings(_env_file=None, host="127.0.0.1", port=1))
    stub_override = app.dependency_overrides[get_redis]
    app.dependency_overrides[get_redis] = lambda: real
    try:
        started = time.monotonic()
        response = client.get("/health")
        elapsed = time.monotonic() - started
    finally:
        app.dependency_overrides[get_redis] = stub_override

    assert response.status_code == 200
    assert response.json()["checks"]["redis"] == "unavailable"
    assert elapsed < 3  # o healthcheck do Docker espera 3 s
