"""Registro dos replays em andamento: é a API de controle (start/pause/resume/speed) do motor.

GARANTIA "UM REPLAY POR CORRIDA", em duas camadas:
1. Dentro do processo: `start()` reserva a corrida SEM nenhum `await` entre checar e reservar. O
   event loop só alterna tarefas em um `await`, então dois `start()` concorrentes não passam juntos
   pelo `if`: o segundo vê a reserva do primeiro e recebe `ReplayAlreadyRunning`.
2. Entre processos: o lock no Redis (lock.py). É o stream, o recurso compartilhado, que fica
   protegido, não só a memória de um processo.

Quem segura o lock o renova a cada TTL/3 (heartbeat). Se o processo morrer, o lock expira sozinho em
até TTL e a corrida pode ser iniciada de novo; se a renovação descobrir que o lock passou a ser de
outro, o replay local se aborta (FAILED) em vez de escrever no mesmo stream que o novo dono.

CONTROLE ENTRE PROCESSOS (ver control.py): este processo é o DONO do replay que iniciou. Ele
publica o status no Redis a cada ~1 s e a cada mudança, e escuta o stream de comandos: um
`POST /pause` recebido por QUALQUER instância da API vira um comando no Redis que o dono executa
(`_execute`) e responde. Os métodos locais (`pause`, `resume`, `set_speed`, `stop`) continuam
existindo para o próprio processo dono.
"""

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.orm import Session, sessionmaker

from app.redis_client import MAX_BLOCK_MS
from replay.clock import Clock
from replay.control import (
    Command,
    CommandResult,
    ControlPlane,
    RedisControlPlane,
    ReplayInfo,
)
from replay.engine import ReplayEngine, ReplayStatus
from replay.errors import InvalidSpeed, ReplayAlreadyRunning, ReplayNotRunning
from replay.events import laps_stream_of, stream_key
from replay.lock import DEFAULT_TTL_MS, ReplayLock, lock_key
from replay.publisher import STREAM_MAXLEN, Publisher, StreamPublisher
from replay.source import DEFAULT_PAGE_SIZE, RaceSource
from replay.state import RaceStateStore

logger = logging.getLogger(__name__)

ALLOWED_SPEEDS = (1.0, 4.0, 16.0)

# O status é republicado a cada ~1 s (e o TTL dele é 10 s); o lock é renovado no mesmo ritmo.
STATUS_INTERVAL_S = 1.0
CONTROL_BLOCK_MS = 1_000  # quanto o dono espera por comandos por leitura (< MAX_BLOCK_MS)
assert CONTROL_BLOCK_MS < MAX_BLOCK_MS

RaceKey = tuple[int, int]  # (year, round)


class StateRegistry(Protocol):
    """O que o gerenciador precisa do estado compartilhado: numerar o replay e anunciar o stream."""

    async def new_epoch(self, year: int, round_number: int) -> int: ...
    async def register_stream(self, year: int, round_number: int) -> None: ...


class Lock(Protocol):
    @property
    def ttl_ms(self) -> int: ...
    async def acquire(self) -> bool: ...
    async def renew(self) -> bool: ...
    async def release(self) -> None: ...


@dataclass(slots=True)
class _Run:
    year: int
    round_number: int
    epoch: int
    engine: ReplayEngine
    lock: Lock
    task: asyncio.Task[None] | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)  # o motor terminou
    # Comandos `stop` já executados cuja RESPOSTA espera o motor terminar (ver `_execute`).
    stop_replies: list[str] = field(default_factory=list)
    # Serializa as publicações de status (heartbeat, comandos, supervisor): ver `_publish_status`.
    status_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ReplayManager:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        redis: Redis,
        *,
        clock: Clock | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        maxlen: int = STREAM_MAXLEN,
        lock_ttl_ms: int = DEFAULT_TTL_MS,
        lock_factory: Callable[[int, int], Lock] | None = None,
        publisher_factory: Callable[[str, int], Publisher] | None = None,
        state: StateRegistry | None = None,
        control: ControlPlane | None = None,
        owner: str | None = None,
        on_events_published: Callable[[int], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock
        self._page_size = page_size
        self._on_events_published = on_events_published
        self._sleep = sleep  # do heartbeat do lock; injetável para os testes não esperarem
        self._lock_factory = lock_factory or (
            lambda year, rnd: ReplayLock(redis, lock_key(year, rnd), lock_ttl_ms)
        )
        self._publisher_factory = publisher_factory or (
            lambda stream, epoch: StreamPublisher(
                redis, stream, epoch, maxlen, laps_stream=laps_stream_of(stream)
            )
        )
        self._state: StateRegistry = state or RaceStateStore(redis)
        self._control: ControlPlane = control or RedisControlPlane(redis)
        # Quem sou eu, para quem consulta o status saber QUAL instância é a dona do replay.
        self.owner = owner or f"{socket.gethostname()}-{os.getpid()}"
        self._runs: dict[RaceKey, _Run] = {}  # o último replay de cada corrida (ativo ou não)
        self._starting: set[RaceKey] = set()  # reservas de start() ainda em andamento

    # ------------------------------------------------------------------ consulta

    def _active(self, key: RaceKey) -> _Run | None:
        run = self._runs.get(key)
        return run if run and run.task is not None and not run.task.done() else None

    def _require(self, year: int, round_number: int) -> _Run:
        run = self._active((year, round_number))
        if run is None:
            raise ReplayNotRunning(f"não há replay ativo para a corrida {year}/{round_number}")
        return run

    def status(self, year: int, round_number: int) -> ReplayStatus | None:
        """Status do último replay da corrida (inclusive já terminado), ou None se nunca houve."""
        run = self._runs.get((year, round_number))
        return run.engine.status() if run else None

    # ------------------------------------------------------------------ controles

    @staticmethod
    def _check_speed(speed: float) -> float:
        if speed not in ALLOWED_SPEEDS:
            allowed = ", ".join(f"{s:g}x" for s in ALLOWED_SPEEDS)
            raise InvalidSpeed(f"velocidade {speed!r} inválida; use uma de: {allowed}")
        return float(speed)

    async def start(self, year: int, round_number: int, speed: float = 1.0) -> ReplayStatus:
        speed = self._check_speed(speed)
        key = (year, round_number)

        # Checar e reservar SEM await no meio (ver docstring do módulo, camada 1).
        if key in self._starting or self._active(key) is not None:
            raise ReplayAlreadyRunning(f"já existe um replay ativo para {year}/{round_number}")
        self._starting.add(key)
        try:
            lock = self._lock_factory(year, round_number)
            if not await lock.acquire():  # camada 2: outro processo pode estar com a corrida
                raise ReplayAlreadyRunning(
                    f"{year}/{round_number} já está em replay em outro processo (lock no Redis)"
                )
            try:
                # Falha cedo, com erro claro (RaceNotFound/RaceHasNoData), antes de tocar no stream.
                source = await RaceSource.open(
                    self._session_factory, year, round_number, page_size=self._page_size
                )
                # Um número por replay, que viaja em toda entrada do stream: é o que isola o estado
                # deste replay do de qualquer replay anterior (ver replay/state.py).
                epoch = await self._state.new_epoch(year, round_number)
                publisher = self._publisher_factory(stream_key(year, round_number), epoch)
                await self._state.register_stream(year, round_number)  # os workers o descobrem
                await publisher.reset()
                await self._control.reset(year, round_number)  # fila de comandos limpa
                engine = ReplayEngine(
                    source,
                    publisher,
                    clock=self._clock,
                    speed=speed,
                    on_published=self._on_events_published,
                )
                run = _Run(year, round_number, epoch, engine, lock)
                # O status existe ANTES de o start devolver: um `POST /pause` que chegue em outra
                # instância logo em seguida já encontra o replay (e o comando fica na fila até o
                # dono começar a ler, que parte do "0-0").
                await self._publish_status(run)
            except BaseException:  # inclui cancelamento (cliente desconectou no meio do start)
                await self._release_quietly(lock)
                raise

            run.task = asyncio.create_task(
                self._supervise(run), name=f"replay-{year}-{round_number}"
            )
            self._runs[key] = run
            return engine.status()
        finally:
            self._starting.discard(key)

    def pause(self, year: int, round_number: int) -> ReplayStatus:
        run = self._require(year, round_number)
        run.engine.pause()
        return run.engine.status()

    def resume(self, year: int, round_number: int) -> ReplayStatus:
        run = self._require(year, round_number)
        run.engine.resume()
        return run.engine.status()

    def set_speed(self, year: int, round_number: int, speed: float) -> ReplayStatus:
        speed = self._check_speed(speed)
        run = self._require(year, round_number)
        run.engine.set_speed(speed)
        return run.engine.status()

    async def stop(self, year: int, round_number: int) -> ReplayStatus:
        run = self._require(year, round_number)
        run.engine.stop()
        assert run.task is not None
        await asyncio.shield(run.task)  # só devolve com o lock já liberado
        return run.engine.status()

    async def shutdown(self, timeout_s: float = 5.0) -> None:
        """Para todos os replays (desligamento da API) e espera liberarem o lock."""
        runs = [run for key in list(self._runs) if (run := self._active(key)) is not None]
        for run in runs:
            run.engine.stop()
        tasks = [run.task for run in runs if run.task is not None]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=timeout_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ supervisão

    def _info(self, run: _Run) -> ReplayInfo:
        return ReplayInfo(run.engine.status(), run.epoch, self.owner, int(time.time() * 1000))

    async def _publish_status(self, run: _Run) -> None:
        """Grava o status do replay no plano de controle, de modo que o ÚLTIMO a chegar ao Redis é
        sempre o mais novo.

        O heartbeat (a cada ~1 s) e o laço de comandos publicam concorrentemente. Se cada um LESSE o
        status e depois esperasse a escrita, uma leitura velha podia terminar de escrever DEPOIS de
        uma nova: o `pause` já respondido e o Redis voltando a dizer `running` até o heartbeat
        seguinte (era o que fazia um teste de integração falhar). A trava serializa as escritas e o
        status é lido DENTRO dela, imediatamente antes de escrever, então quem escreve depois
        sempre leu depois."""
        async with run.status_lock:
            await self._control.publish_status(
                run.year, run.round_number, run.engine.status(), epoch=run.epoch, owner=self.owner
            )

    async def _publish_status_quietly(self, run: _Run) -> None:
        try:
            await self._publish_status(run)
        except (RedisError, OSError):
            # O status é republicado no próximo ciclo; falhar em publicá-lo não pode matar o replay.
            logger.warning("Não consegui publicar o status do replay", exc_info=True)

    async def _supervise(self, run: _Run) -> None:
        heartbeat = asyncio.create_task(self._heartbeat(run))
        control = asyncio.create_task(self._control_loop(run))
        engine_task = asyncio.create_task(run.engine.run())
        engine_task.add_done_callback(lambda _: run.finished.set())
        try:
            await asyncio.sleep(0)  # deixa o motor sair de PENDING...
            await self._publish_status_quietly(run)  # ...e publica já como "running"
            await engine_task
        finally:
            # ORDEM IMPORTA. 1) Para os laços que também publicam status: nenhum pode gravar um
            # status "ativo" velho POR CIMA do final.
            for task in (heartbeat, control):
                task.cancel()
            for task in (heartbeat, control):
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            # 2) Publica o status FINAL (terminal, TTL longo) ANTES de responder ao `stop`: quem
            # recebe a resposta e manda outro comando logo em seguida já encontra "terminou" (409
            # imediato), em vez de achar um status ativo velho e esperar o timeout.
            await self._publish_status_quietly(run)
            # 3) A resposta do `stop` (estado FINAL) sai DAQUI, não do laço de comandos: o
            # cancelamento acima o interromperia antes de responder, e o `POST /stop` daria
            # timeout mesmo tendo funcionado.
            await self._reply_to_stops(run)
            await self._release_quietly(run.lock)

    async def _heartbeat(self, run: _Run) -> None:
        # 3 tentativas de renovar o lock antes de expirar, e no máximo 1 s entre status.
        interval_s = min(run.lock.ttl_ms / 1000 / 3, STATUS_INTERVAL_S)
        while True:
            await self._sleep(interval_s)
            try:
                still_mine = await run.lock.renew()
            except (RedisError, OSError):
                # Redis indisponível por um instante: tenta de novo no próximo ciclo. O TTL tem
                # margem para 2 falhas seguidas; se o Redis ficar fora mais que isso, o publisher
                # também falha e o replay termina como FAILED.
                logger.warning(
                    "Não consegui renovar o lock do replay; tento de novo", exc_info=True
                )
                still_mine = True
            if not still_mine:
                logger.error("Lock do replay perdido: abortando para não duplicar eventos")
                run.engine.abort("lock do replay perdido (outro processo assumiu a corrida?)")
                await self._publish_status_quietly(run)
                return
            await self._publish_status_quietly(run)

    # ------------------------------------------------------------------ comandos remotos

    async def _control_loop(self, run: _Run) -> None:
        """Executa, neste processo (o dono), os comandos que qualquer instância mandou."""
        after_id = "0-0"  # a fila foi limpa no start: tudo o que há nela é para ESTE replay
        while True:
            try:
                commands, after_id = await self._control.read_commands(
                    run.year, run.round_number, after_id, CONTROL_BLOCK_MS
                )
            except (RedisError, OSError):
                logger.warning("Não consegui ler os comandos do replay", exc_info=True)
                await self._sleep(1.0)
                continue
            for command in commands:
                result = await self._execute(run, command)
                if result is None:  # resposta adiada (stop): quem responde é o supervisor
                    continue
                # O status ANTES da resposta (como o `stop` já fazia): quem recebe `paused` e
                # consulta o status em seguida, de outra instância, tem de achar `paused` e não o
                # status velho de antes do comando.
                await self._publish_status_quietly(run)
                await self._reply(run, command.id, result)

    async def _reply(self, run: _Run, command_id: str, result: CommandResult) -> None:
        try:
            await self._control.reply(run.year, run.round_number, command_id, result)
        except (RedisError, OSError):
            logger.warning("Não consegui responder ao comando", exc_info=True)

    async def _reply_to_stops(self, run: _Run) -> None:
        for command_id in run.stop_replies:
            await self._reply(run, command_id, CommandResult(True, info=self._info(run)))
        run.stop_replies.clear()

    async def _execute(self, run: _Run, command: Command) -> CommandResult | None:
        engine = run.engine
        match command.name:
            case "pause":
                engine.pause()
            case "resume":
                engine.resume()
            case "speed":
                try:
                    if command.arg is None:
                        raise InvalidSpeed("o comando `speed` precisa de uma velocidade")
                    engine.set_speed(self._check_speed(command.arg))
                except InvalidSpeed as exc:
                    return CommandResult(False, error="invalid_speed", detail=str(exc))
            case "stop":
                engine.stop()
                run.stop_replies.append(command.id)  # responde com o estado FINAL (ver supervisor)
                return None
            case _:
                return CommandResult(
                    False, error="unknown_command", detail=f"comando desconhecido: {command.name!r}"
                )
        return CommandResult(True, info=self._info(run))

    @staticmethod
    async def _release_quietly(lock: Lock) -> None:
        try:
            await lock.release()
        except (RedisError, OSError):
            # Não esconde o erro original de quem chamou. Sem liberar, o TTL expira sozinho.
            logger.warning(
                "Não consegui liberar o lock do replay; vai expirar por TTL", exc_info=True
            )
