"""O motor do replay: dá a cada evento a sua hora e o publica.

Ciclo de vida: PENDING -> RUNNING <-> PAUSED -> FINISHED | STOPPED | FAILED.

Controles (`pause`, `resume`, `set_speed`, `stop`) são síncronos e só mexem em memória: rodam no
mesmo event loop do `run()`, então nunca há duas coisas mexendo na Timeline ao mesmo tempo (sem
locks). PAUSA e RETOMADA preservam a posição porque o estado do replay É a Timeline (o tempo de
corrida congelado) mais o ponto onde o gerador de eventos parou: nada é relido nem reiniciado.
"""

import asyncio
import contextlib
import enum
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from replay.clock import Clock, SystemClock
from replay.events import RaceStatusEvent, ReplayEvent
from replay.publisher import Publisher
from replay.timeline import Timeline, validate_speed

logger = logging.getLogger(__name__)

# Eventos que vencem nos próximos 5 ms são publicados JUNTOS, num único lote. A 16x chegam ~1.760
# eventos/s (um a cada ~0,6 ms): sem isso seriam ~1.760 ida-e-voltas ao Redis por segundo e o loop
# acordaria 1.760 vezes por segundo. A troca: um evento pode sair até 5 ms ANTES da hora exata (um
# frame de tela a 60 Hz são 16 ms, ninguém vê). Não acumula: a hora de cada evento vem da Timeline.
DEFAULT_LOOKAHEAD_S = 0.005

# Piso da tolerância: um evento a menos de 1 µs de vencer já venceu. Sem isso, o erro de
# arredondamento do ponto flutuante poderia deixar o loop pedindo esperas de ~1e-13 s que nunca
# avançam o relógio.
_MIN_TOLERANCE_S = 1e-6

# Teto do lote: limita o tamanho de cada transação MULTI/EXEC e a latência do 1º evento do lote.
MAX_BATCH = 500


class EventSource(Protocol):
    t0_ms: int

    def events(self) -> AsyncIterator[ReplayEvent]: ...


class ReplayState(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"  # reproduziu a corrida inteira
    STOPPED = "stopped"  # parado por pedido (stop/shutdown)
    FAILED = "failed"  # erro (ver `error`)


ACTIVE_STATES = frozenset({ReplayState.PENDING, ReplayState.RUNNING, ReplayState.PAUSED})


@dataclass(frozen=True, slots=True)
class ReplayStatus:
    state: ReplayState
    speed: float
    race_time_ms: int | None  # onde o replay está AGORA, no tempo da corrida (epoch ms)
    lap: int | None  # volta do líder
    total_laps: int | None
    events_published: int
    error: str | None


class ReplayEngine:
    def __init__(
        self,
        source: EventSource,
        publisher: Publisher,
        *,
        clock: Clock | None = None,
        speed: float = 1.0,
        lookahead_s: float = DEFAULT_LOOKAHEAD_S,
        max_batch: int = MAX_BATCH,
    ) -> None:
        self._source = source
        self._publisher = publisher
        self._clock = clock or SystemClock()
        self._tolerance_s = max(lookahead_s, _MIN_TOLERANCE_S)
        self._max_batch = max_batch
        self._initial_speed = validate_speed(speed)

        self._timeline: Timeline | None = None
        self._state = ReplayState.PENDING
        self._error: str | None = None
        self._stopping = False
        self._start_paused = False  # pause() pedido antes de run() criar a Timeline
        self._lap: int | None = None
        self._total_laps: int | None = None
        self._published = 0
        # `_wake` interrompe a espera pelo próximo evento (mudou velocidade, pausou, parou).
        # `_running` é o "portão" da pausa: aberto = andando, fechado = pausado.
        self._wake = asyncio.Event()
        self._running = asyncio.Event()
        self._running.set()

    # ------------------------------------------------------------------ controles

    def pause(self) -> None:
        """Congela o tempo de corrida. Idempotente. Antes de `run()`, vale desde o início."""
        if self._state is ReplayState.PENDING:
            self._start_paused = True
            return
        if self._timeline is None or self._state not in ACTIVE_STATES:
            return
        self._timeline.pause(self._clock.monotonic())
        self._running.clear()
        self._state = ReplayState.PAUSED
        self._wake.set()

    def resume(self) -> None:
        """Retoma de ONDE parou: o tempo de corrida continua do ponto congelado. Idempotente."""
        if self._state is ReplayState.PENDING:
            self._start_paused = False
            return
        if self._timeline is None or self._state is not ReplayState.PAUSED:
            return
        self._timeline.resume(self._clock.monotonic())
        self._state = ReplayState.RUNNING
        self._running.set()
        self._wake.set()

    def set_speed(self, speed: float) -> None:
        """Muda a velocidade SEM reiniciar nem pular: a Timeline reancora no instante atual.
        Também interrompe a espera em andamento (calculada com a velocidade antiga)."""
        speed = validate_speed(speed)
        if self._timeline is None:
            self._initial_speed = speed  # ainda não começou: vale a partir do início
        elif self._state in ACTIVE_STATES:
            self._timeline.set_speed(speed, self._clock.monotonic())
        self._wake.set()

    def stop(self) -> None:
        self._stopping = True
        self._running.set()  # solta quem está parado no portão da pausa
        self._wake.set()

    def abort(self, reason: str) -> None:
        """Para por um motivo que é um ERRO (ex.: perdeu o lock): termina como FAILED."""
        self._error = reason
        self.stop()

    # ------------------------------------------------------------------ consulta

    @property
    def state(self) -> ReplayState:
        return self._state

    @property
    def speed(self) -> float:
        return self._timeline.speed if self._timeline is not None else self._initial_speed

    def status(self) -> ReplayStatus:
        race_ms = int(self._timeline.race_ms(self._clock.monotonic())) if self._timeline else None
        return ReplayStatus(
            state=self._state,
            speed=self.speed,
            race_time_ms=race_ms,
            lap=self._lap,
            total_laps=self._total_laps,
            events_published=self._published,
            error=self._error,
        )

    # ------------------------------------------------------------------ o loop

    async def run(self) -> None:
        """Reproduz a corrida até o fim, ou até `stop()`/erro. Nunca levanta (exceto cancelamento):
        a falha vira `state == FAILED` com `error`, para quem supervisiona decidir o que fazer."""
        self._timeline = Timeline(self._source.t0_ms, self._clock.monotonic(), self._initial_speed)
        self._state = ReplayState.RUNNING
        if self._start_paused:
            self.pause()
        try:
            await self._play()
        except asyncio.CancelledError:
            self._state = ReplayState.STOPPED
            raise
        except Exception as exc:
            logger.exception("Replay falhou")
            self._error = repr(exc)
            self._state = ReplayState.FAILED
            return
        if self._error is not None:
            self._state = ReplayState.FAILED
        elif self._stopping:
            self._state = ReplayState.STOPPED
        else:
            self._state = ReplayState.FINISHED

    async def _play(self) -> None:
        timeline = self._timeline
        assert timeline is not None
        batch: list[ReplayEvent] = []

        async with contextlib.aclosing(self._source.events()) as events:
            async for event in events:
                # Espera até o evento "vencer". Cada volta recalcula tudo a partir da Timeline.
                while True:
                    if self._stopping:
                        await self._flush(batch)
                        return
                    delay = timeline.delay_s(event.t, self._clock.monotonic())
                    if delay is not None and delay <= self._tolerance_s:
                        break  # venceu (ou vence dentro do lookahead): entra no lote
                    if batch:
                        # Vai esperar: antes, entrega o que já venceu (não segura evento atrasado).
                        # O await abaixo pode deixar um controle mudar a Timeline, por isso o
                        # `continue`: recalcula o delay em vez de usar o valor velho.
                        await self._flush(batch)
                        continue
                    if delay is None:
                        await self._running.wait()  # pausado: sem relógio, só o portão
                    else:
                        # Sem nenhum `await` entre calcular `delay` e este clear(): um set_speed()
                        # não consegue acontecer "no meio" e ter o aviso apagado.
                        self._wake.clear()
                        await self._clock.wait(delay, self._wake)
                batch.append(event)
                if len(batch) >= self._max_batch:
                    await self._flush(batch)
            await self._flush(batch)

    async def _flush(self, batch: list[ReplayEvent]) -> None:
        if not batch:
            return
        await self._publisher.publish_many(batch)
        self._published += len(batch)
        for event in batch:
            if isinstance(event, RaceStatusEvent):
                self._lap, self._total_laps = event.lap, event.total_laps
        batch.clear()
