"""Testes do gerenciador: um replay por corrida, controles, heartbeat do lock e desligamento.

Banco SQLite semeado + relógio falso + publisher/lock em memória: nenhum Redis, nenhum sleep real.
O lock distribuído de verdade (SET NX PX + Lua) e o publisher real ficam em tests/integration.
"""

import asyncio
from unittest.mock import Mock

import pytest

from replay.engine import ReplayState, ReplayStatus
from replay.errors import (
    InvalidSpeed,
    RaceHasNoData,
    RaceNotFound,
    ReplayAlreadyRunning,
    ReplayNotRunning,
)
from replay.lock import lock_key
from replay.manager import ReplayManager
from tests.replay.conftest import NOR, VER, at
from tests.replay.fakes import (
    FakeClock,
    FakeControl,
    FakeLock,
    FakeLockRegistry,
    FakePublisher,
    FakeState,
    GatedControl,
    ManualClock,
    TickSleep,
    settle,
)

RACE = (2024, 21)


async def fast_sleep(_: float) -> None:
    await asyncio.sleep(0)  # o heartbeat "espera" um turno do loop, não segundos de verdade


class World:
    """Um gerenciador ligado a dublês, com acesso a tudo que ele criou."""

    def __init__(
        self,
        session_factory,
        registry: FakeLockRegistry | None = None,
        clock=None,
        sleep=None,
        control: FakeControl | None = None,
    ) -> None:
        self.clock = clock or FakeClock()
        self.control = control or FakeControl()
        self.registry = registry or FakeLockRegistry()
        self.publisher = FakePublisher(self.clock)
        self.shared = FakeState()
        self.locks: list[FakeLock] = []
        self.manager = ReplayManager(
            session_factory,
            Mock(),  # o Redis nunca é tocado: lock e publisher são injetados
            clock=self.clock,
            lock_factory=self._new_lock,
            publisher_factory=lambda stream, epoch: self.publisher,
            state=self.shared,
            control=self.control,
            sleep=sleep or fast_sleep,
        )

    def _new_lock(self, year: int, round_number: int) -> FakeLock:
        lock = self.registry.lock(lock_key(year, round_number))
        self.locks.append(lock)
        return lock

    def state(self, race=RACE) -> ReplayState:
        status = self.manager.status(*race)
        assert status is not None
        return status.state

    async def start_paused(self, race=RACE, speed: float = 1.0) -> ReplayStatus:
        """Inicia e pausa NA HORA: a corrida fica ativa de forma determinística (com o relógio
        falso um replay sem pausa terminaria antes de o teste conseguir olhar para ele)."""
        status = await self.manager.start(*race, speed=speed)
        self.manager.pause(*race)
        await settle(lambda: self.state(race) is ReplayState.PAUSED)
        return status


@pytest.fixture
def world(seed, session_factory) -> World:
    seed(
        positions=[(VER, at(i * 1000), i, i) for i in range(5)],
        laps=[(VER, 1, at(2000), 90_000, "SOFT", 1)],
        total_laps=3,
    )
    return World(session_factory)


# ---------------------------------------------------------------- ciclo de vida


async def test_start_replays_the_race_into_the_stream_and_releases_the_lock(world):
    status = await world.manager.start(*RACE, speed=4)

    assert (status.state, status.speed) == (ReplayState.PENDING, 4.0)
    await settle(lambda: world.state() is ReplayState.FINISHED)
    await settle(lambda: world.registry.owners == {})  # lock liberado ao terminar

    kinds = [type(e).__name__ for e in world.publisher.events]
    assert kinds[0] == kinds[-1] == "RaceStatusEvent" and kinds.count("PositionEvent") == 5
    assert world.publisher.resets == 1  # o stream começou limpo, uma vez


async def test_start_after_the_previous_replay_finished_starts_a_new_one(world):
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.FINISHED)
    await settle(lambda: world.registry.owners == {})

    await world.manager.start(*RACE)
    await settle(lambda: world.publisher.resets == 2)
    await settle(lambda: world.state() is ReplayState.FINISHED)

    assert world.publisher.resets == 2  # cada replay limpa o stream


async def test_each_replay_gets_a_new_epoch_and_announces_its_stream(world):
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.FINISHED)
    await settle(lambda: world.registry.owners == {})
    assert world.shared.epochs[RACE] == 1 and RACE in world.shared.streams  # os workers o descobrem

    await world.manager.start(*RACE)

    assert world.shared.epochs[RACE] == 2  # outro replay, outro número


# ---------------------------------------------------------------- um replay por corrida


async def test_calling_start_twice_does_not_create_two_replays(world):
    await world.start_paused()

    with pytest.raises(ReplayAlreadyRunning):
        await world.manager.start(*RACE)

    assert len(world.locks) == 1 and world.publisher.resets == 1  # o 2º nem chegou ao Redis


async def test_concurrent_starts_create_exactly_one_replay(world):
    results = await asyncio.gather(
        *(world.manager.start(*RACE) for _ in range(5)), return_exceptions=True
    )

    started = [r for r in results if isinstance(r, ReplayStatus)]
    refused = [r for r in results if isinstance(r, ReplayAlreadyRunning)]
    assert (len(started), len(refused)) == (1, 4)
    assert world.publisher.resets == 1


async def test_another_process_holding_the_lock_blocks_the_start(world, session_factory):
    """Dois "processos" (dois gerenciadores) compartilhando o mesmo Redis (o registro de locks)."""
    other = World(session_factory, registry=world.registry)
    await world.start_paused()

    with pytest.raises(ReplayAlreadyRunning, match="outro processo"):
        await other.manager.start(*RACE)

    assert other.publisher.resets == 0  # o stream do replay que já roda NÃO foi apagado
    assert other.manager.status(*RACE) is None


async def test_different_races_replay_in_parallel(world, seed):
    seed(positions=[(VER, at(0), 1, 1)], year=2023, round_number=1, race_id=2, total_laps=1)

    await world.start_paused(RACE)
    await world.start_paused((2023, 1))

    assert world.state(RACE) is world.state((2023, 1)) is ReplayState.PAUSED
    assert len(world.registry.owners) == 2


# ---------------------------------------------------------------- falhas na partida


async def test_unknown_race_fails_before_touching_the_stream_and_frees_the_lock(world):
    with pytest.raises(RaceNotFound):
        await world.manager.start(2099, 1)

    assert world.registry.owners == {} and world.publisher.resets == 0
    assert world.manager.status(2099, 1) is None


async def test_race_without_data_fails_cleanly(seed, session_factory):
    seed(positions=[], laps=[(VER, 1, None, None, None, None)])
    world = World(session_factory)

    with pytest.raises(RaceHasNoData):
        await world.manager.start(*RACE)

    assert world.registry.owners == {} and world.publisher.resets == 0


async def test_after_a_failed_start_the_race_can_be_started_again(world):
    with pytest.raises(RaceNotFound):
        await world.manager.start(2099, 1)

    await world.manager.start(*RACE)  # nada ficou "preso" no registro

    await settle(lambda: world.state() is ReplayState.FINISHED)


@pytest.mark.parametrize("speed", [0, 2, 3.5, 32, -1])
async def test_only_1x_4x_and_16x_are_accepted(world, speed):
    with pytest.raises(InvalidSpeed):
        await world.manager.start(*RACE, speed=speed)

    assert world.registry.owners == {}  # a validação vem antes de qualquer lock


@pytest.mark.parametrize("speed", [1, 4, 16])
async def test_the_allowed_speeds_are_accepted(world, speed):
    status = await world.start_paused(speed=speed)

    assert status.speed == speed


# ---------------------------------------------------------------- controles


@pytest.mark.parametrize("control", ["pause", "resume", "stop"])
async def test_controls_on_a_race_without_replay_raise_not_running(world, control):
    with pytest.raises(ReplayNotRunning):
        result = getattr(world.manager, control)(*RACE)
        if asyncio.iscoroutine(result):
            await result

    with pytest.raises(ReplayNotRunning):
        world.manager.set_speed(*RACE, 4)


async def test_pause_resume_and_speed_drive_the_engine(world):
    await world.start_paused()

    status = world.manager.set_speed(*RACE, 16)
    assert (status.state, status.speed) == (ReplayState.PAUSED, 16.0)
    with pytest.raises(InvalidSpeed):
        world.manager.set_speed(*RACE, 3)

    world.manager.resume(*RACE)
    await settle(lambda: world.state() is ReplayState.FINISHED)
    await settle(lambda: world.registry.owners == {})  # o supervisor terminou (lock liberado)

    with pytest.raises(ReplayNotRunning):  # terminou: não há mais o que pausar
        world.manager.pause(*RACE)
    final = world.manager.status(*RACE)
    assert final is not None and final.events_published == len(world.publisher.events)


async def test_stop_returns_only_after_the_lock_is_released(world):
    await world.start_paused()

    status = await world.manager.stop(*RACE)

    assert status.state is ReplayState.STOPPED
    assert world.registry.owners == {}  # já liberado quando stop() devolve


# ---------------------------------------------------------------- heartbeat do lock


async def test_the_lock_is_renewed_while_the_replay_runs(world):
    await world.start_paused()

    await settle(lambda: world.locks[0].renewals >= 3)

    assert world.state() is ReplayState.PAUSED  # renovar não atrapalha o replay
    await world.manager.stop(*RACE)


async def test_a_transient_redis_error_while_renewing_does_not_kill_the_replay(world):
    await world.start_paused()
    world.locks[0].renew_errors = 2  # duas renovações seguidas falham...

    await settle(lambda: world.locks[0].renewals >= 5)  # ...e as seguintes voltam a funcionar

    assert world.state() is ReplayState.PAUSED  # continua vivo
    await world.manager.stop(*RACE)


async def test_losing_the_lock_aborts_the_replay_instead_of_writing_to_the_same_stream(world):
    await world.start_paused()
    world.locks[0].lose_after_renewals = 1  # expirou e outro processo assumiu a corrida

    await settle(lambda: world.state() is ReplayState.FAILED)

    status = world.manager.status(*RACE)
    assert status is not None and "lock" in (status.error or "")
    # E NÃO apagou o lock do novo dono ao terminar (o token é do outro agora).
    assert world.registry.owners == {lock_key(*RACE): "someone-else"}


# ---------------------------------------------------------------- desligamento


async def test_shutdown_stops_every_replay_and_releases_every_lock(world, seed):
    seed(positions=[(NOR, at(0), 1, 1)], year=2023, round_number=1, race_id=2, total_laps=1)
    await world.start_paused(RACE)
    await world.start_paused((2023, 1))

    await world.manager.shutdown()

    assert world.state(RACE) is world.state((2023, 1)) is ReplayState.STOPPED
    assert world.registry.owners == {}


# ---------------------------------------------------------------- status e controle REMOTOS


async def test_start_publishes_the_status_before_returning(world):
    await world.manager.start(*RACE)

    assert world.control.resets == 1  # a fila de comandos começou limpa
    status, epoch, owner = world.control.statuses[0]
    # Já existe para quem consultar de outra instância; e diz de quem é o replay.
    assert (status.state, epoch, owner) == (ReplayState.PENDING, 1, world.manager.owner)


async def test_the_heartbeat_keeps_republishing_the_status(world):
    await world.start_paused()
    published = len(world.control.statuses)

    await settle(lambda: len(world.control.statuses) >= published + 3)

    assert world.control.last_status.state is ReplayState.PAUSED


async def test_the_final_status_is_terminal(world):
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.FINISHED)
    await settle(lambda: world.registry.owners == {})  # supervisor terminou

    assert world.control.last_status.state is ReplayState.FINISHED  # vira TTL longo no Redis


@pytest.fixture
def manual_world(seed, session_factory) -> World:
    """Um replay que fica VIVO (esperando) até um comando mexer nele."""
    seed(positions=[(VER, at(0), 0, 0), (VER, at(60_000), 1, 1)], total_laps=3)
    return World(session_factory, clock=ManualClock())


async def test_remote_commands_are_executed_by_the_owner_and_answered(manual_world):
    world = manual_world
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.RUNNING)

    pause = world.control.send("pause")  # (como se viesse de OUTRA instância da API)
    await settle(lambda: pause in world.control.replies)
    assert world.control.replies[pause].ok
    assert world.state() is ReplayState.PAUSED  # o dono executou
    assert (
        world.control.replies[pause].info.status.state is ReplayState.PAUSED
    )  # e respondeu o novo

    speed = world.control.send("speed", 16)
    await settle(lambda: speed in world.control.replies)
    assert world.control.replies[speed].info.status.speed == 16.0
    assert world.state() is ReplayState.PAUSED  # trocar a velocidade não retoma

    resume = world.control.send("resume")
    await settle(lambda: resume in world.control.replies)
    assert world.state() is ReplayState.RUNNING

    stop = world.control.send("stop")
    await settle(lambda: stop in world.control.replies)
    assert world.control.replies[stop].info.status.state is ReplayState.STOPPED  # estado FINAL
    await settle(lambda: world.registry.owners == {})
    assert world.control.last_status.state is ReplayState.STOPPED


@pytest.mark.parametrize(
    ("name", "arg", "error"),
    [
        ("speed", 3, "invalid_speed"),
        ("speed", None, "invalid_speed"),
        ("dance", None, "unknown_command"),
    ],
)
async def test_invalid_remote_commands_get_an_error_reply_and_change_nothing(
    manual_world, name, arg, error
):
    world = manual_world
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.RUNNING)

    command = world.control.send(name, arg)
    await settle(lambda: command in world.control.replies)

    reply = world.control.replies[command]
    assert (reply.ok, reply.error) == (False, error)
    assert world.state() is ReplayState.RUNNING and world.manager.status(*RACE).speed == 1.0
    await world.manager.stop(*RACE)


async def test_a_command_sent_right_after_start_is_not_lost(manual_world):
    """A janela clássica: o POST /start devolve e, no mesmo instante, outra instância manda
    /pause, antes de o dono ter começado a ler. O comando espera na fila e é executado."""
    world = manual_world
    await world.manager.start(*RACE)
    command = world.control.send("pause")  # nem deu tempo de o motor sair de PENDING

    await settle(lambda: command in world.control.replies)

    assert world.state() is ReplayState.PAUSED
    await world.manager.stop(*RACE)


async def never_ticks(_: float) -> None:
    await asyncio.Event().wait()  # o heartbeat não bate: só o supervisor publica status


async def test_the_terminal_status_is_published_before_the_stop_is_answered(seed, session_factory):
    """Quem recebe a resposta do `stop` e manda outro comando em seguida tem de encontrar o status
    JÁ terminal (409 imediato). Se a resposta saísse antes, ele veria um status "ativo" velho e
    esperaria o timeout inteiro (visto em teste real com duas instâncias da API).

    O heartbeat fica mudo de propósito: com o `sleep` rápido dos outros testes ele publicaria um
    "stopped" espúrio por conta própria, e o teste passaria mesmo com a ordem errada."""
    seed(positions=[(VER, at(0), 0, 0), (VER, at(60_000), 1, 1)], total_laps=3)
    world = World(session_factory, clock=ManualClock(), sleep=never_ticks)
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.RUNNING)

    stop = world.control.send("stop")
    await settle(lambda: stop in world.control.replies)

    log = world.control.log
    reply_at = log.index(("reply", stop))
    assert log[reply_at - 1] == ("status", "stopped")  # o terminal vem logo ANTES da resposta
    assert ("status", "running") not in log[reply_at:]  # e nada "ativo" sobrescreve depois


# ---------------------------------------------------------------- o status nunca anda para trás
#
# Corrida que fazia `test_the_rest_endpoints_drive_a_replay_owned_by_another_process` falhar de vez
# em quando (2 vezes em 14 execuções completas): `POST /pause` devolvia `paused` e o `GET` seguinte
# lia `running`. Eram DUAS corridas, e cada teste abaixo reproduz uma com a ordem sob controle.


async def test_the_status_is_published_before_a_command_is_answered(seed, session_factory):
    """Corrida 1: a resposta saía ANTES do status. Quem recebe `paused` e consulta o status logo em
    seguida (outra instância da API, via Redis) lia o status velho, porque a escrita do novo ainda
    estava a caminho. O `stop` já publicava antes de responder; pause, resume e speed não.

    O heartbeat fica mudo: se ele publicasse por conta própria, o teste passaria mesmo com a ordem
    errada (a mesma armadilha do teste do `stop`)."""
    seed(positions=[(VER, at(0), 0, 0), (VER, at(60_000), 1, 1)], total_laps=3)
    world = World(session_factory, clock=ManualClock(), sleep=never_ticks)
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.RUNNING)

    for name, arg in (("pause", None), ("speed", 16), ("resume", None)):
        command = world.control.send(name, arg)
        await settle(lambda command=command: command in world.control.replies)

        answered = world.control.replies[command].info.status
        published = world.control.status_at_reply[command]
        assert (published.state, published.speed) == (answered.state, answered.speed), name

    await world.manager.stop(*RACE)


async def test_a_stale_heartbeat_write_cannot_land_on_top_of_a_newer_status(seed, session_factory):
    """Corrida 2: o heartbeat LÊ o status (`running`) e a escrita demora; nesse intervalo o pause
    executa e publica `paused`; a escrita velha chega depois e o Redis volta a dizer `running`
    (até o heartbeat seguinte, ~1 s). A latência do Redis é simulada: a publicação do heartbeat
    fica no ar até o teste soltá-la."""
    seed(positions=[(VER, at(0), 0, 0), (VER, at(60_000), 1, 1)], total_laps=3)
    control, tick = GatedControl(), TickSleep()
    world = World(session_factory, clock=ManualClock(), sleep=tick, control=control)
    await world.manager.start(*RACE)
    await settle(lambda: world.state() is ReplayState.RUNNING)

    control.hold_next_publish()
    tick.tick()  # o heartbeat lê "running"...
    await settle(lambda: control.in_flight)  # ...e a escrita fica a caminho
    pause = control.send("pause")
    await settle(lambda: world.state() is ReplayState.PAUSED)  # o dono executou o pause
    control.release()  # a escrita velha ("running") termina AGORA, depois do pause
    await settle(lambda: pause in control.replies)
    for _ in range(50):  # deixa qualquer publicação pendente terminar
        await asyncio.sleep(0)

    assert control.last_status.state is ReplayState.PAUSED
    await world.manager.stop(*RACE)
