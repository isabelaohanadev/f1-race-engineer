from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.deps import get_db
from app.main import app


def test_health_is_200_and_reports_each_check_when_database_is_up(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"process": "ok", "database": "ok"},
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
        "checks": {"process": "ok", "database": "unavailable"},
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
    assert response.json()["checks"] == {"process": "ok", "database": "unavailable"}
