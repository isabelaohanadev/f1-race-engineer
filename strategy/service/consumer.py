"""O consumidor de estratégia: lê o stream de voltas, calcula a cada volta do líder e grava o
resultado no Redis. Um processo à parte da API e do worker de estado (`python -m strategy.service`).

UMA INSTÂNCIA ATIVA POR VEZ, COM LOCK E STANDBY
O cálculo é pesado (de 0,05 s a ~1,3 s de CPU por volta) e o estado de cada corrida mora na memória
de quem calcula. Duas instâncias calculando a mesma volta gastariam CPU em dobro para escrever o
mesmo número; repartir as voltas entre elas obrigaria cada uma a ter o estado inteiro mesmo assim.
Por isso só UMA calcula, a que segura o lock `strategy:consumer:lock` (o mesmo lock simples do
replay: SET NX PX, renovado a cada TTL/3, token para nunca apagar o lock de outro). As outras ficam
em STANDBY, tentando adquiri-lo a cada segundo. Se a ativa morre sem liberar, o lock expira em até o
TTL e uma standby assume; num desligamento limpo ele é liberado e a assunção é imediata.

QUEM ASSUME RECONSTRÓI O ESTADO DO STREAM, e não do consumer group. O estado de uma corrida é o
conjunto de todas as voltas (`state_until` precisa do histórico), e o stream de voltas guarda a
corrida inteira (~1.400 entradas, nunca aparado no meio de uma corrida): quem assume lê tudo com
XRANGE, aplica pelo mesmo caminho do ao vivo (dedupe incluído) e só então passa a ler as entradas
novas. O consumer group existe para o Redis lembrar até onde se leu e quem tem entrega pendente (o
lag é visível em XINFO GROUPS), mas a correção NÃO depende dele: depois da reconstrução o grupo é
posicionado no fim do que foi lido (`XGROUP SETID`) e as pendências de quem morreu são confirmadas
(já estão no estado). Se a ativa morreu entre receber a volta N e gravar o resultado, a que assume
calcula a última volta no mesmo instante, a menos que o resultado já esteja lá.

O QUE NÃO SE RECUPERA numa troca de instância: a histerese do alerta de undercut (um par que estava
"ligado" a 0,55 recomeça "desligado") e o cálculo que estava em voo (a que assume refaz a última
volta, e o primeiro a chegar ao armazém vence).

PERDEU O LOCK = PARA JÁ. Se a renovação descobre que o lock é de outro (ou o Redis ficou fora por
mais que o TTL), a instância para de consumir e de entregar resultados na hora e volta ao standby.
Um cálculo em voo dela termina e é DESCARTADO na entrega. Resta a janela do processo congelado por
mais que o TTL (pausa de GC, VM suspensa): ele pode entregar um resultado depois de perder o lock. O
armazém limita o estrago (recusa epoch antigo; a mesma volta nunca é publicada duas vezes), e como o
cálculo é determinístico o segundo a chegar teria os mesmos números.

Cada corrida tem o seu executor "só a última" (strategy/forecast_runner.py): duas corridas em replay
ao mesmo tempo não cancelam as simulações uma da outra.
"""

import asyncio
import contextlib
import logging
import os
import re
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from app.redis_client import MAX_BLOCK_MS
from replay.events import MalformedEvent, laps_stream_of, parse_entry
from replay.lock import ReplayLock
from replay.state import STREAMS_REGISTRY_KEY
from strategy import metrics
from strategy.forecast_runner import LatestOnlyRunner
from strategy.fuel import FuelVariant
from strategy.live import (
    DEFAULT_WAIT_S,
    Computation,
    Constants,
    RaceTracker,
    Trigger,
    computation_job,
)
from strategy.montecarlo import SimConfig
from strategy.payload import SCHEMA_VERSION, build_payload, dumps
from strategy.protocol import frozen_sim_config, frozen_undercut_config
from strategy.service.results import PutOutcome, ResultStore
from strategy.undercut import AlertTracker, UndercutConfig

logger = logging.getLogger(__name__)

STRATEGY_GROUP = "strategy"
LOCK_KEY = "strategy:consumer:lock"
REBUILD_PAGE = 500

_STREAM_RE = re.compile(r"^race:(\d+):(\d+)$")


class ConstantsSource(Protocol):
    async def for_race(self, year: int, round_number: int) -> Constants: ...


@dataclass(slots=True)
class ConsumerStats:
    leaderships: int = 0  # quantas vezes esta instância assumiu o lock
    lost_lock: int = 0
    rebuilt: int = 0  # corridas reconstruídas do stream
    submitted: int = 0
    written: int = 0
    exists: int = 0
    fenced: int = 0
    skipped: int = 0
    dropped: int = 0


def default_consumer_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


@dataclass(slots=True)
class RaceRuntime:
    """O que o consumidor ativo guarda de UMA corrida."""

    year: int
    round: int
    stream: str  # o stream de voltas
    tracker: RaceTracker
    alerts: AlertTracker
    runner: LatestOnlyRunner
    constants_task: asyncio.Task[Constants] | None = None


class StrategyConsumer:
    def __init__(
        self,
        redis: Redis,
        constants: ConstantsSource,
        *,
        consumer: str | None = None,
        store: ResultStore | None = None,
        wait_s: float = DEFAULT_WAIT_S,
        lock_ttl_ms: int = 10_000,
        standby_poll_s: float = 1.0,
        block_ms: int = 500,
        batch_size: int = 200,
        discover_interval_s: float = 1.0,
        sim_config: SimConfig | None = None,
        undercut_config: UndercutConfig | None = None,
        variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if block_ms >= MAX_BLOCK_MS:
            raise ValueError(
                f"block_ms deve ser < {MAX_BLOCK_MS} (o BLOCK precisa vencer antes do timeout)"
            )
        self._redis = redis
        self._constants = constants
        self.consumer = consumer or default_consumer_name()
        self._store = store or ResultStore(redis)
        self._wait_s = wait_s
        self._lock = ReplayLock(redis, LOCK_KEY, lock_ttl_ms)
        self._standby_poll_s = standby_poll_s
        self._block_ms = block_ms
        self._batch_size = batch_size
        self._discover_interval_s = discover_interval_s
        self._sim_config = sim_config or frozen_sim_config()
        self._undercut_config = undercut_config or frozen_undercut_config()
        self._variant = variant
        self._clock = clock
        self._wall_clock = wall_clock
        self._races: dict[tuple[int, int], RaceRuntime] = {}
        self._by_stream: dict[str, RaceRuntime] = {}
        self._lost = asyncio.Event()
        self.active = False
        self.stats = ConsumerStats()

    # ------------------------------------------------------------ o laço externo: lock e standby

    async def run(self, stop: asyncio.Event) -> None:
        """Roda até `stop`. Alterna entre standby (sem o lock) e ativo (com o lock)."""
        logger.info("Consumidor de estratégia %s iniciado (standby)", self.consumer)
        while not stop.is_set():
            try:
                acquired = await self._lock.acquire()
            except (RedisError, OSError):
                logger.warning(
                    "Redis indisponível ao tentar o lock; tentando de novo", exc_info=True
                )
                acquired = False
            if not acquired:
                await self._idle(stop, self._standby_poll_s)
                continue
            await self._lead(stop)
        logger.info("Consumidor de estratégia %s parou: %s", self.consumer, self.stats)

    async def _lead(self, stop: asyncio.Event) -> None:
        self.stats.leaderships += 1
        self.active = True
        metrics.CONSUMER_ACTIVE.set(1)
        self._lost = asyncio.Event()
        logger.info("Consumidor %s assumiu o lock: está ATIVO", self.consumer)
        heartbeat = asyncio.create_task(self._heartbeat(), name="strategy-lock-heartbeat")
        try:
            await self._consume(stop)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            await self._step_down()

    async def _step_down(self) -> None:
        self.active = False
        metrics.CONSUMER_ACTIVE.set(0)
        runtimes, self._races, self._by_stream = list(self._races.values()), {}, {}
        for runtime in runtimes:
            if runtime.constants_task is not None:
                runtime.constants_task.cancel()
            await runtime.runner.close()
        # Libera só se ainda é o meu (o token): sem isso a standby esperaria o TTL numa saída limpa.
        with contextlib.suppress(RedisError, OSError):
            await self._lock.release()
        logger.info("Consumidor %s saiu do papel de ativo", self.consumer)

    async def _heartbeat(self) -> None:
        """Renova o lock a cada TTL/3. Perdeu (outro o tem) ou ficou sem renovar por mais que o TTL:
        marca `_lost` e o laço de consumo para."""
        ttl_s = self._lock.ttl_ms / 1000
        last_ok = self._clock()
        while True:
            await asyncio.sleep(ttl_s / 3)
            try:
                still_mine = await self._lock.renew()
            except (RedisError, OSError):
                if self._clock() - last_ok <= ttl_s:
                    logger.warning("Não consegui renovar o lock; tento de novo", exc_info=True)
                    continue
                still_mine = False  # o lock já pode ter expirado: não se pode mais confiar nele
            if not still_mine:
                logger.error("Lock da estratégia perdido: parando de calcular e de entregar")
                self.stats.lost_lock += 1
                self._lost.set()
                return
            last_ok = self._clock()

    async def _idle(self, stop: asyncio.Event, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), seconds)

    # ------------------------------------------------------------ o laço de consumo (ativo)

    async def _consume(self, stop: asyncio.Event) -> None:
        last_discover = float("-inf")
        while not stop.is_set() and not self._lost.is_set():
            try:
                now = self._clock()
                if now - last_discover >= self._discover_interval_s:
                    await self._discover()
                    last_discover = now
                await self._fire_due()
                if not self._by_stream:
                    await self._idle(stop, min(self._discover_interval_s, 0.2))
                    continue
                reply = await self._redis.xreadgroup(
                    STRATEGY_GROUP,
                    self.consumer,
                    {stream: ">" for stream in self._by_stream},
                    count=self._batch_size,
                    block=self._block_timeout_ms(),
                )
                for stream, messages in reply or []:
                    await self._process(stream, messages)
                await self._fire_due()
            except ResponseError as exc:
                if "NOGROUP" not in str(exc):
                    raise
                # Um replay novo apagou o stream de voltas (e o grupo junto): recria e segue.
                for runtime in list(self._by_stream.values()):
                    await self._ensure_group(runtime.stream)
            except (RedisError, OSError):
                logger.warning("Redis indisponível; tentando de novo", exc_info=True)
                await self._idle(stop, 1.0)

    def _block_timeout_ms(self) -> int:
        """Bloqueia na leitura até o fim da janela mais próxima (e no máximo `block_ms`)."""
        deadlines = [rt.tracker.next_deadline for rt in self._races.values()]
        pending = [d for d in deadlines if d is not None]
        if not pending:
            return self._block_ms
        wait_ms = int((min(pending) - self._clock()) * 1000) + 1
        return max(1, min(self._block_ms, wait_ms))  # BLOCK 0 seria "para sempre"

    # ------------------------------------------------------------ descoberta e reconstrução

    async def _discover(self) -> None:
        members = await self._redis.smembers(STREAMS_REGISTRY_KEY)  # type: ignore[misc]
        for stream in sorted(members):
            match = _STREAM_RE.match(stream)
            if match is None:
                continue
            key = (int(match.group(1)), int(match.group(2)))
            if key not in self._races:
                await self._open_race(*key)

    async def _ensure_group(self, stream: str) -> None:
        try:
            await self._redis.xgroup_create(stream, STRATEGY_GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):  # BUSYGROUP = já existe: o caso normal
                raise

    def _new_runner(self) -> LatestOnlyRunner:
        return LatestOnlyRunner(self._deliver, observe=self._observe)

    @staticmethod
    def _observe(duration_s: float, result: str) -> None:
        metrics.SIMULATION_DURATION.labels(result).observe(duration_s)
        metrics.SIMULATIONS.labels(result).inc()

    async def _open_race(self, year: int, round_number: int) -> None:
        stream = laps_stream_of(f"race:{year}:{round_number}")
        runtime = RaceRuntime(
            year,
            round_number,
            stream,
            RaceTracker(year, round_number, wait_s=self._wait_s),
            AlertTracker(self._undercut_config),
            self._new_runner(),
        )
        # Registrada ANTES de reconstruir: o cálculo da última volta (lançado pela reconstrução)
        # entrega o resultado procurando a corrida aqui.
        self._races[(year, round_number)] = runtime
        self._by_stream[stream] = runtime
        try:
            await self._ensure_group(stream)
            await self._rebuild(runtime)
        except BaseException:
            del self._races[(year, round_number)], self._by_stream[stream]
            await runtime.runner.close()
            raise
        self.stats.rebuilt += 1

    async def _rebuild(self, runtime: RaceRuntime) -> None:
        """Reconstrói o estado da corrida do stream de voltas (quem assume, ou o 1º a ver a corrida)
        e posiciona o grupo no fim do que leu. Se a corrida está em andamento e a última volta ainda
        não tem resultado, calcula-a já, sem janela."""
        cursor, last_id = "-", None
        while True:
            page = await self._redis.xrange(runtime.stream, min=cursor, max="+", count=REBUILD_PAGE)
            for entry_id, fields in page:
                last_id = entry_id
                try:
                    entry = parse_entry(entry_id, fields)
                except MalformedEvent:
                    continue
                applied = runtime.tracker.apply(entry, self._clock(), schedule=False)
                if applied.new_epoch:
                    runtime.alerts = AlertTracker(self._undercut_config)
            if len(page) < REBUILD_PAGE:
                break
            cursor = f"({page[-1][0]}"
        if last_id is None:
            return
        await self._settle_group(runtime.stream, last_id)
        if runtime.tracker.finished:
            return  # uma corrida que já acabou: não há "última volta" a calcular agora
        trigger = runtime.tracker.resume()
        if trigger is not None and not await self._store.has(
            runtime.year, runtime.round, epoch=trigger.epoch, lap=trigger.lap
        ):
            await self._submit(runtime, trigger)

    async def _settle_group(self, stream: str, last_id: str) -> None:
        """O que se leu do stream está no estado: o grupo passa a entregar só o que vier depois, e
        as entregas pendentes (de quem morreu) são confirmadas."""
        await self._redis.xgroup_setid(stream, STRATEGY_GROUP, last_id)
        while True:
            pending = await self._redis.xpending_range(
                stream, STRATEGY_GROUP, min="-", max=last_id, count=REBUILD_PAGE
            )
            if not pending:
                return
            await self._redis.xack(stream, STRATEGY_GROUP, *[p["message_id"] for p in pending])

    # ------------------------------------------------------------ entradas e disparos

    async def _process(self, stream: str, messages: list[tuple[str, dict[str, str]]]) -> None:
        runtime = self._by_stream[stream]
        now = self._clock()
        for entry_id, fields in messages:
            try:
                entry = parse_entry(entry_id, fields)
            except MalformedEvent:
                logger.warning("Entrada ilegível descartada: %s %s", entry_id, fields)
                metrics.LAPS.labels("malformed").inc()
                continue
            applied = runtime.tracker.apply(entry, now)
            metrics.LAPS.labels(applied.outcome.value).inc()
            if applied.new_epoch:
                await self._new_replay(runtime)
        # A confirmação vem depois de aplicar ao estado em memória: é barato e síncrono. Se a
        # instância morrer antes, quem assume reconstrói tudo do stream (não depende do grupo).
        await self._redis.xack(stream, STRATEGY_GROUP, *[entry_id for entry_id, _ in messages])

    async def _new_replay(self, runtime: RaceRuntime) -> None:
        """Um replay novo: o que estava calculando é de um replay que passou, e a histerese dos
        alertas também. O armazém já recusaria o resultado velho; cancelar poupa a CPU."""
        old, runtime.runner = runtime.runner, self._new_runner()
        runtime.alerts = AlertTracker(self._undercut_config)
        await old.close()

    async def _fire_due(self) -> None:
        now = self._clock()
        for runtime in list(self._races.values()):
            for trigger in runtime.tracker.due(now):
                await self._submit(runtime, trigger)

    async def _constants_for(self, runtime: RaceRuntime) -> Constants:
        task = runtime.constants_task
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            task = runtime.constants_task = asyncio.create_task(
                self._constants.for_race(runtime.year, runtime.round)
            )
        return await asyncio.shield(task)

    async def _submit(self, runtime: RaceRuntime, trigger: Trigger) -> None:
        try:
            constants = await self._constants_for(runtime)
        except Exception:
            logger.exception("Não consegui as constantes de %s/%s", runtime.year, runtime.round)
            return
        job = computation_job(
            trigger,
            constants,
            sim_config=self._sim_config,
            undercut_config=self._undercut_config,
            variant=self._variant,
        )
        self.stats.submitted += 1
        await runtime.runner.submit(job)

    # ------------------------------------------------------------ a entrega

    async def _deliver(self, tag, computation: Computation | None) -> None:
        """Chamado pelo executor, no event loop, quando um cálculo termina e ainda é o mais novo."""
        year, round_number, epoch, lap = tag
        runtime = self._races.get((year, round_number))
        if computation is None:
            self._count("skipped")
            return
        if runtime is None or runtime.tracker.epoch != epoch or self._lost.is_set():
            self._count("dropped")  # de um replay que passou, ou calculado depois de perder o lock
            return
        trigger = computation.trigger
        alerts = runtime.alerts.update_all(computation.assessments)
        payload = build_payload(
            year=year,
            round_number=round_number,
            epoch=epoch,
            lap=lap,
            total_laps=trigger.total_laps,
            forecast=computation.forecast,
            sim_input=computation.sim_input,
            leader=computation.leader,
            assessments=computation.assessments,
            alerts=alerts,
            warnings=computation.warnings,
            cars_known=trigger.cars_known,
            late_laps=trigger.late_laps,
            duplicates=trigger.duplicates,
            wait_s=trigger.wait_s,
            computed_at_ms=int(self._wall_clock() * 1000),
        )
        outcome = await self._store.put(
            year, round_number, epoch=epoch, lap=lap, payload=dumps(payload), version=SCHEMA_VERSION
        )
        self._count(outcome.value)
        if outcome is PutOutcome.WRITTEN:
            metrics.mark_delivered()

    def _count(self, outcome: str) -> None:
        metrics.DELIVERIES.labels(outcome).inc()
        setattr(self.stats, outcome, getattr(self.stats, outcome) + 1)
