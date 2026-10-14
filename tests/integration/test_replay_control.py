"""Controle e status do replay ENTRE PROCESSOS, contra o Redis real.

"Processo A" = o gerenciador que roda o motor (o dono). "Processo B" = um plano de controle com
OUTRA conexão Redis, sem nenhuma referência ao motor: só enxerga o que está no Redis, exatamente
como uma segunda instância da API. O relógio é o ManualClock (o motor fica vivo, esperando, até
um comando mexer nele), então o teste é determinístico mesmo com I/O de rede de verdade.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest
from httpx import ASGITransport
from redis.asyncio import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

import replay.manager as manager_module
from app.config import RedisSettings
from app.deps import get_control, get_replay_manager
from app.main import app
from app.redis_client import create_redis
from replay.control import RedisControlPlane, ReplayInfo, status_key
from replay.engine import ReplayState
from replay.errors import ReplayAlreadyRunning, ReplayNotRunning, ReplayTimeout
from replay.manager import ReplayManager
from tests.replay.conftest import VER, at, seed_race
from tests.replay.fakes import ManualClock

RACE = (2024, 21)


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    # 2 posições com 60 s de corrida entre elas: o motor publica a 1ª e fica ESPERANDO a 2ª.
    seed_race(engine, positions=[(VER, at(0), 0, 0), (VER, at(60_000), 1, 1)], total_laps=3)
    return sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def other_redis(redis_test_settings: RedisSettings) -> AsyncIterator[Redis]:
    """A conexão do "processo B": cliente separado, sem estado compartilhado com o do dono."""
    client = create_redis(redis_test_settings)
    yield client
    await client.aclose()


@pytest.fixture
async def owner(redis_client: Redis, session_factory) -> AsyncIterator[ReplayManager]:
    manager = ReplayManager(session_factory, redis_client, clock=ManualClock(), owner="processo-A")
    yield manager
    await manager.shutdown()


@pytest.fixture
def remote(other_redis: Redis) -> RedisControlPlane:
    return RedisControlPlane(other_redis)


async def wait_for(
    predicate: Callable[[], Awaitable[bool]], what: str = "condição", timeout_s: float = 5.0
) -> None:
    try:
        async with asyncio.timeout(timeout_s):
            while not await predicate():
                await asyncio.sleep(0.01)
    except TimeoutError:
        pytest.fail(f"timeout esperando: {what}")


async def wait_state(remote: RedisControlPlane, state: ReplayState) -> ReplayInfo:
    async def reached() -> bool:
        info = await remote.get_info(*RACE)
        return info is not None and info.status.state is state

    await wait_for(reached, f"estado {state.value}")
    info = await remote.get_info(*RACE)
    assert info is not None
    return info


# ================================================================ o controle entre processos


async def test_another_process_can_pause_change_speed_resume_and_stop_the_replay(
    owner: ReplayManager, remote: RedisControlPlane
):
    await owner.start(*RACE)

    info = await wait_state(remote, ReplayState.RUNNING)  # B enxerga o replay de A
    assert (info.owner, info.epoch) == ("processo-A", 1)

    paused = await remote.send_command(*RACE, "pause")  # B manda, A executa
    assert paused.ok and paused.info is not None
    assert paused.info.status.state is ReplayState.PAUSED
    owner_status = owner.status(*RACE)
    assert owner_status is not None and owner_status.state is ReplayState.PAUSED  # A MUDOU de fato

    faster = await remote.send_command(*RACE, "speed", 16)
    assert faster.info is not None and faster.info.status.speed == 16.0
    assert faster.info.status.state is ReplayState.PAUSED  # trocar a velocidade não retoma

    resumed = await remote.send_command(*RACE, "resume")
    assert resumed.info is not None and resumed.info.status.state is ReplayState.RUNNING

    stopped = await remote.send_command(*RACE, "stop")
    assert stopped.info is not None and stopped.info.status.state is ReplayState.STOPPED  # FINAL

    final = await wait_state(remote, ReplayState.STOPPED)
    assert final.status.events_published >= 2  # `started` + a 1ª posição
    with pytest.raises(ReplayNotRunning):  # terminou: não há mais o que comandar
        await remote.send_command(*RACE, "pause")


async def test_an_invalid_speed_is_rejected_by_the_owner_and_changes_nothing(
    owner: ReplayManager, remote: RedisControlPlane
):
    await owner.start(*RACE)
    await wait_state(remote, ReplayState.RUNNING)

    result = await remote.send_command(*RACE, "speed", 3)

    assert (result.ok, result.error) == (False, "invalid_speed")
    info = await remote.get_info(*RACE)
    assert info is not None and info.status.speed == 1.0


async def test_status_has_a_short_ttl_while_active_and_a_long_one_when_finished(
    owner: ReplayManager, remote: RedisControlPlane, redis_client: Redis
):
    await owner.start(*RACE)
    await wait_state(remote, ReplayState.RUNNING)
    assert 0 < await redis_client.pttl(status_key(*RACE)) <= 10_000  # some se o dono morrer

    await remote.send_command(*RACE, "stop")
    await wait_state(remote, ReplayState.STOPPED)

    assert await redis_client.pttl(status_key(*RACE)) > 3_000_000  # o desfecho fica ~1 h


async def test_the_owner_keeps_renewing_the_status_ttl(
    owner: ReplayManager, remote: RedisControlPlane, redis_client: Redis, monkeypatch
):
    """É a detecção de falha: se o dono PARA de renovar, o status expira e o replay deixa de
    aparecer como ativo. Enquanto ele vive, o TTL é empurrado de volta (heartbeat)."""
    monkeypatch.setattr(manager_module, "STATUS_INTERVAL_S", 0.05)
    await owner.start(*RACE)
    await wait_state(remote, ReplayState.RUNNING)

    await redis_client.pexpire(status_key(*RACE), 800)  # "quase expirando"

    async def renewed() -> bool:
        return await redis_client.pttl(status_key(*RACE)) > 5_000

    await wait_for(renewed, "TTL do status renovado pelo dono")


async def test_the_status_of_a_dead_owner_expires(remote: RedisControlPlane, redis_client: Redis):
    """Dono que morreu sem limpar nada: o hash some sozinho e o replay não aparece mais."""
    await redis_client.hset(status_key(*RACE), mapping=_fake_status_fields("running"))
    await redis_client.pexpire(status_key(*RACE), 100)
    assert await remote.get_info(*RACE) is not None

    await asyncio.sleep(0.25)  # o único tempo real: o TTL é do SERVIDOR

    assert await remote.get_info(*RACE) is None


def _fake_status_fields(state: str) -> dict[str, str]:
    return {
        "state": state,
        "speed": "1.0",
        "events_published": "0",
        "epoch": "1",
        "owner": "fantasma",
        "updated_at_ms": "0",
    }


async def test_commands_without_an_active_replay_fail_fast(
    remote: RedisControlPlane, redis_client: Redis
):
    with pytest.raises(ReplayNotRunning):  # nunca houve replay
        await remote.send_command(*RACE, "pause")

    # Há um status "ativo" no Redis, mas NENHUM dono o está executando (ex.: ele travou).
    await redis_client.hset(status_key(*RACE), mapping=_fake_status_fields("running"))
    with pytest.raises(ReplayTimeout):
        await remote.send_command(*RACE, "pause", timeout_s=0.2)


async def test_a_second_process_cannot_start_the_same_race_through_the_real_lock(
    redis_client: Redis, other_redis: Redis, session_factory
):
    one = ReplayManager(session_factory, redis_client, clock=ManualClock(), owner="A")
    two = ReplayManager(session_factory, other_redis, clock=ManualClock(), owner="B")
    try:
        results = await asyncio.gather(one.start(*RACE), two.start(*RACE), return_exceptions=True)

        assert sum(isinstance(r, ReplayAlreadyRunning) for r in results) == 1  # só um dono
        assert sum(not isinstance(r, Exception) for r in results) == 1
    finally:
        await asyncio.gather(one.shutdown(), two.shutdown())


async def test_commands_of_a_previous_replay_are_not_replayed_by_the_next_one(
    owner: ReplayManager, remote: RedisControlPlane
):
    await owner.start(*RACE)
    await wait_state(remote, ReplayState.RUNNING)
    await remote.send_command(*RACE, "pause")  # um comando do replay 1...
    await remote.send_command(*RACE, "stop")
    await wait_state(remote, ReplayState.STOPPED)

    async def lock_released() -> bool:
        return owner._active(RACE) is None

    await wait_for(lock_released, "dono liberou o lock")
    await owner.start(*RACE)  # replay 2

    info = await wait_state(remote, ReplayState.RUNNING)
    assert info.epoch == 2  # ...e o `pause` antigo NÃO foi executado de novo: continua rodando


# ================================================================ pela API REST


async def test_the_rest_endpoints_drive_a_replay_owned_by_another_process(
    owner: ReplayManager, remote: RedisControlPlane
):
    """A API que recebe o POST (a de baixo) não tem o motor: só o plano de controle no Redis."""
    app.dependency_overrides[get_replay_manager] = lambda: owner
    app.dependency_overrides[get_control] = lambda: remote
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as api:
            started = await api.post("/replay/2024/21/start", json={"speed": 4})
            assert started.status_code == 202 and started.json()["speed"] == 4.0
            await wait_state(remote, ReplayState.RUNNING)

            again = await api.post("/replay/2024/21/start")
            assert again.status_code == 409  # um replay por corrida

            paused = await api.post("/replay/2024/21/pause")
            assert paused.status_code == 200 and paused.json()["state"] == "paused"
            assert (await api.get("/replay/2024/21")).json()["state"] == "paused"

            speed = await api.post("/replay/2024/21/speed", json={"speed": 16})
            assert speed.json()["speed"] == 16.0
            assert (await api.post("/replay/2024/21/speed", json={"speed": 3})).status_code == 422

            resumed = await api.post("/replay/2024/21/resume")
            assert resumed.json()["state"] == "running"

            stopped = await api.post("/replay/2024/21/stop")
            assert stopped.status_code == 200 and stopped.json()["state"] == "stopped"
            started_at = asyncio.get_running_loop().time()
            assert (await api.post("/replay/2024/21/pause")).status_code == 409  # já terminou
            assert (
                asyncio.get_running_loop().time() - started_at < 1.0
            )  # imediato, sem esperar timeout
    finally:
        app.dependency_overrides.pop(get_replay_manager, None)
        app.dependency_overrides.pop(get_control, None)
