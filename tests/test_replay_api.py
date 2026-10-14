"""Testes unitários dos endpoints /replay (gerenciador e plano de controle trocados por dublês).

O caminho REAL (REST -> Redis -> dono do replay em OUTRO processo) é testado em
tests/integration/test_replay_control.py.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.deps import get_control, get_replay_manager
from app.main import app
from replay.control import CommandResult, ReplayInfo
from replay.engine import ReplayState, ReplayStatus
from replay.errors import (
    RaceHasNoData,
    RaceNotFound,
    ReplayAlreadyRunning,
    ReplayNotRunning,
    ReplayTimeout,
)


def make_info(state=ReplayState.RUNNING, speed=1.0) -> ReplayInfo:
    status = ReplayStatus(
        state=state,
        speed=speed,
        race_time_ms=1_730_649_000_000,
        lap=3,
        total_laps=69,
        events_published=1234,
        error=None,
    )
    return ReplayInfo(status, epoch=2, owner="api-1-4242", updated_at_ms=1_730_649_000_500)


class FakeManager:
    def __init__(self) -> None:
        self.started: list[tuple[int, int, float]] = []
        self.error: Exception | None = None

    async def start(self, year, round_number, speed=1.0):
        if self.error:
            raise self.error
        self.started.append((year, round_number, speed))


class FakeRemoteControl:
    def __init__(self) -> None:
        self.info: ReplayInfo | None = make_info()
        self.commands: list[tuple[str, float | None]] = []
        self.error: Exception | None = None
        self.result: CommandResult | None = None

    async def get_info(self, year, round_number):
        return self.info

    async def send_command(self, year, round_number, name, arg=None, **kwargs):
        if self.error:
            raise self.error
        self.commands.append((name, arg))
        return self.result or CommandResult(True, info=self.info)


@pytest.fixture
def manager() -> FakeManager:
    return FakeManager()


@pytest.fixture
def control() -> FakeRemoteControl:
    return FakeRemoteControl()


@pytest.fixture
def api(manager: FakeManager, control: FakeRemoteControl) -> Iterator[TestClient]:
    app.dependency_overrides[get_replay_manager] = lambda: manager
    app.dependency_overrides[get_control] = lambda: control
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_replay_manager, None)
        app.dependency_overrides.pop(get_control, None)


# ---------------------------------------------------------------- start


def test_start_is_202_with_the_status_and_defaults_to_1x(api, manager):
    response = api.post("/replay/2024/21/start")

    assert response.status_code == 202
    assert manager.started == [(2024, 21, 1)]
    body = response.json()
    assert (body["state"], body["lap"], body["total_laps"], body["owner"]) == (
        "running",
        3,
        69,
        "api-1-4242",
    )


def test_start_accepts_an_initial_speed(api, manager):
    assert api.post("/replay/2024/21/start", json={"speed": 16}).status_code == 202

    assert manager.started == [(2024, 21, 16)]


@pytest.mark.parametrize("speed", [0, 2, 3, 32, -4, "rápido"])
def test_start_rejects_speeds_other_than_1_4_16(api, manager, speed):
    response = api.post("/replay/2024/21/start", json={"speed": speed})

    assert response.status_code == 422
    assert manager.started == []


def test_start_of_a_race_that_was_not_ingested_is_404_with_the_ingestion_command(api, manager):
    manager.error = RaceNotFound("corrida 2024/21 não encontrada no banco")

    response = api.post("/replay/2024/21/start")

    assert response.status_code == 404
    assert "python -m ingestion --year 2024 --round 21" in response.json()["detail"]


def test_start_while_already_running_is_409(api, manager):
    manager.error = ReplayAlreadyRunning("já existe um replay ativo para 2024/21")

    response = api.post("/replay/2024/21/start")

    assert response.status_code == 409
    assert "já existe" in response.json()["detail"]


def test_start_of_a_race_without_data_is_422(api, manager):
    manager.error = RaceHasNoData("sem posições nem voltas")

    assert api.post("/replay/2024/21/start").status_code == 422


def test_start_with_redis_down_is_503_without_leaking_infrastructure_details(api, manager):
    manager.error = RedisConnectionError("Error 111 connecting to 10.0.0.9:6379 (senha=hunter2)")

    response = api.post("/replay/2024/21/start")

    assert response.status_code == 503
    assert "10.0.0.9" not in response.text and "hunter2" not in response.text


@pytest.mark.parametrize(
    "path", ["/replay/1900/21/start", "/replay/2024/0/start", "/replay/2024/99/start"]
)
def test_path_parameters_are_validated(api, path):
    assert api.post(path).status_code == 422


# ---------------------------------------------------------------- pause / resume / speed / stop


@pytest.mark.parametrize(
    ("path", "body", "command"),
    [
        ("pause", None, ("pause", None)),
        ("resume", None, ("resume", None)),
        ("speed", {"speed": 4}, ("speed", 4)),
        ("stop", None, ("stop", None)),
    ],
)
def test_controls_go_through_the_control_plane_and_return_the_new_status(
    api, control, path, body, command
):
    control.info = make_info(ReplayState.PAUSED, 4.0)

    response = api.post(f"/replay/2024/21/{path}", json=body)

    assert response.status_code == 200
    assert control.commands == [command]
    assert response.json()["state"] == "paused" and response.json()["speed"] == 4.0


def test_the_speed_endpoint_validates_the_body(api, control):
    assert api.post("/replay/2024/21/speed", json={"speed": 3}).status_code == 422
    assert api.post("/replay/2024/21/speed", json={}).status_code == 422
    assert control.commands == []  # nada chegou ao dono


@pytest.mark.parametrize("path", ["pause", "resume", "speed", "stop"])
def test_controls_on_a_race_without_active_replay_are_409(api, control, path):
    control.error = ReplayNotRunning("não há replay ativo para a corrida 2024/21")

    response = api.post(f"/replay/2024/21/{path}", json={"speed": 4} if path == "speed" else None)

    assert response.status_code == 409


def test_an_owner_that_does_not_answer_is_504(api, control):
    control.error = ReplayTimeout("o dono do replay 2024/21 não confirmou o comando")

    response = api.post("/replay/2024/21/pause")

    assert response.status_code == 504


def test_a_command_rejected_by_the_owner_is_422(api, control):
    control.result = CommandResult(False, error="invalid_speed", detail="velocidade 4.5 inválida")

    response = api.post("/replay/2024/21/speed", json={"speed": 4})

    assert response.status_code == 422 and "inválida" in response.json()["detail"]


# ---------------------------------------------------------------- status


def test_get_status_reads_it_from_the_shared_state(api, control):
    control.info = make_info(ReplayState.FINISHED)

    response = api.get("/replay/2024/21")

    assert response.status_code == 200 and response.json()["state"] == "finished"


def test_get_status_without_any_replay_is_404(api, control):
    control.info = None

    assert api.get("/replay/2024/21").status_code == 404
