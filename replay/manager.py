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
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.orm import Session, sessionmaker

from replay.clock import Clock
from replay.engine import ReplayEngine, ReplayStatus
from replay.errors import InvalidSpeed, ReplayAlreadyRunning, ReplayNotRunning
from replay.events import stream_key
from replay.lock import DEFAULT_TTL_MS, ReplayLock, lock_key
from replay.publisher import STREAM_MAXLEN, Publisher, StreamPublisher
from replay.source import DEFAULT_PAGE_SIZE, RaceSource
from replay.state import RaceStateStore

logger = logging.getLogger(__name__)

ALLOWED_SPEEDS = (1.0, 4.0, 16.0)

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
    engine: ReplayEngine
    lock: Lock
    task: asyncio.Task[None] | None = None


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
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock
        self._page_size = page_size
        self._sleep = sleep  # do heartbeat do lock; injetável para os testes não esperarem
        self._lock_factory = lock_factory or (
            lambda year, rnd: ReplayLock(redis, lock_key(year, rnd), lock_ttl_ms)
        )
        self._publisher_factory = publisher_factory or (
            lambda stream, epoch: StreamPublisher(redis, stream, epoch, maxlen)
        )
        self._state: StateRegistry = state or RaceStateStore(redis)
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
                engine = ReplayEngine(source, publisher, clock=self._clock, speed=speed)
            except BaseException:  # inclui cancelamento (cliente desconectou no meio do start)
                await self._release_quietly(lock)
                raise

            run = _Run(engine, lock)
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

    async def _supervise(self, run: _Run) -> None:
        heartbeat = asyncio.create_task(self._heartbeat(run))
        try:
            await run.engine.run()
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            await self._release_quietly(run.lock)

    async def _heartbeat(self, run: _Run) -> None:
        interval_s = run.lock.ttl_ms / 1000 / 3  # 3 tentativas de renovar antes de expirar
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
                continue
            if not still_mine:
                logger.error("Lock do replay perdido: abortando para não duplicar eventos")
                run.engine.abort("lock do replay perdido (outro processo assumiu a corrida?)")
                return

    @staticmethod
    async def _release_quietly(lock: Lock) -> None:
        try:
            await lock.release()
        except (RedisError, OSError):
            # Não esconde o erro original de quem chamou. Sem liberar, o TTL expira sozinho.
            logger.warning(
                "Não consegui liberar o lock do replay; vai expirar por TTL", exc_info=True
            )
