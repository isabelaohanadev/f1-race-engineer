"""O laço do worker: XREADGROUP -> aplicar (Lua) -> XACK, e XAUTOCLAIM das pendências abandonadas.

POR QUE CONSUMER GROUP (e compartilhado)
* Repartição: cada entrada vai para UM worker do grupo, então N workers dividem a carga (a 16x são
  ~1.760 eventos/s) sem processar o mesmo evento N vezes.
* Memória do progresso: o Redis guarda até onde o grupo leu (`last-delivered-id`) e quais entradas
  foram entregues e ainda não confirmadas (o PEL, "pending entries list"). Se o worker reinicia, o
  progresso não se perde.
* Confirmação explícita (XACK): só depois de GRAVAR o estado. Um worker que morre entre ler e
  gravar deixa a entrada no PEL: nada se perde (a garantia é "pelo menos uma vez").

O QUE ACONTECE SE UM WORKER MORRER COM MENSAGENS PENDENTES
As entradas que ele leu e não confirmou ficam no PEL, em nome dele, e o Redis NÃO as redistribui
sozinho. Por isso todo worker roda, de tempos em tempos, XAUTOCLAIM: "me dê as entradas pendentes há
mais de X ms ociosas, de qualquer consumidor". X (`claim_min_idle_ms`) tem de ser bem MAIOR que o
tempo normal de processar um lote (alguns ms), para não roubar mensagens de um worker saudável que
só está no meio do trabalho; mas curto o bastante para a perda de um worker não atrasar o estado
por muito tempo. Se o worker "morto" na verdade só estava lento e terminar depois, a entrada é
aplicada duas vezes, o que é inofensivo: a atualização é idempotente (replay/state.py).

POR QUE `>` E NÃO O HISTÓRICO
`XREADGROUP ... >` pede só as entradas NUNCA entregues ao grupo. Um worker que reinicia ganha um
nome novo (host-pid-uuid), então não reler "o próprio" PEL: as pendências do antecessor são
resgatadas pelo XAUTOCLAIM de qualquer worker vivo.
"""

import asyncio
import contextlib
import logging
import os
import re
import socket
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from app.redis_client import MAX_BLOCK_MS
from replay.events import MalformedEvent, id_ms, parse_entry
from replay.latency import seconds_since
from replay.state import STREAMS_REGISTRY_KEY, WORKER_GROUP, RaceStateStore
from worker import metrics

logger = logging.getLogger(__name__)

_STREAM_RE = re.compile(r"^race:(\d+):(\d+)$")


@dataclass(slots=True)
class WorkerStats:
    """Contadores do worker (as métricas Prometheus da etapa seguinte leem daqui)."""

    batches: int = 0
    applied: int = 0  # gravou no estado
    ignored: int = 0  # velho, repetido ou de replay antigo (a trava 1 em ação)
    malformed: int = 0  # entrada ilegível: confirmada e descartada
    claimed: int = 0  # resgatadas de um worker que sumiu (XAUTOCLAIM)


def default_consumer_name() -> str:
    # Único por INSTÂNCIA: dois processos no mesmo host (ou o mesmo reiniciado) nunca colidem.
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


class StateWorker:
    def __init__(
        self,
        redis: Redis,
        *,
        consumer: str | None = None,
        batch_size: int = 500,
        block_ms: int = 1_000,
        claim_min_idle_ms: int = 10_000,
        claim_interval_s: float = 5.0,
        discover_interval_s: float = 1.0,
    ) -> None:
        if block_ms >= MAX_BLOCK_MS:
            raise ValueError(
                f"block_ms deve ser < {MAX_BLOCK_MS} (o BLOCK precisa vencer antes do timeout)"
            )
        self._redis = redis
        self._store = RaceStateStore(redis)
        self.consumer = consumer or default_consumer_name()
        self._batch_size = batch_size
        self._block_ms = block_ms
        self._claim_min_idle_ms = claim_min_idle_ms
        self._claim_interval_s = claim_interval_s
        self._discover_interval_s = discover_interval_s
        self._races: dict[str, tuple[int, int]] = {}  # stream -> (year, round)
        self.stats = WorkerStats()

    # ------------------------------------------------------------ descoberta e grupos

    async def _discover(self) -> None:
        members = await self._redis.smembers(STREAMS_REGISTRY_KEY)  # type: ignore[misc]
        races: dict[str, tuple[int, int]] = {}
        for stream in sorted(members):
            match = _STREAM_RE.match(stream)
            if match:
                races[stream] = (int(match.group(1)), int(match.group(2)))
        self._races = races
        await self._ensure_groups()

    async def _ensure_groups(self) -> None:
        """Garante o grupo em cada stream, lendo do INÍCIO (id "0"): o grupo recriado depois de um
        novo replay (o DEL do stream apaga os grupos) tem de ver TODAS as entradas do stream novo,
        e o `$` perderia as publicadas entre o DEL e a recriação. MKSTREAM cria o stream vazio se
        ele ainda não existe (o grupo nasce antes do 1º evento)."""
        for stream in self._races:
            try:
                await self._redis.xgroup_create(stream, WORKER_GROUP, id="0", mkstream=True)
            except ResponseError as exc:
                if "BUSYGROUP" not in str(exc):  # BUSYGROUP = já existe: é o caso normal
                    raise

    # ------------------------------------------------------------ processamento

    async def _process(
        self, stream: str, messages: list[tuple[str, dict[str, str]]], *, claimed: bool = False
    ) -> None:
        """Aplica um lote num único ida-e-volta (pipeline) e confirma. A confirmação vem NO FIM:
        se a conexão cair no meio, as entradas ficam pendentes e serão reentregues."""
        if not messages:
            return
        year, round_number = self._races[stream]
        ack_ids: list[str] = []
        applied_ids: list[
            str
        ] = []  # as que entraram no pipeline (para medir a latência de cada uma)
        queued = 0
        async with self._redis.pipeline(transaction=False) as pipe:
            for entry_id, fields in messages:
                ack_ids.append(entry_id)
                try:
                    entry = parse_entry(entry_id, fields)
                except MalformedEvent:
                    # Sem confirmar, ela voltaria a cada XAUTOCLAIM para sempre (mensagem "veneno").
                    logger.warning("Entrada ilegível descartada: %s %s", entry_id, fields)
                    self.stats.malformed += 1
                    metrics.EVENTS.labels("malformed").inc()
                    continue
                await self._store.queue_apply(pipe, year, round_number, entry)
                applied_ids.append(entry_id)
                queued += 1
            pipe.xack(stream, WORKER_GROUP, *ack_ids)
            results = await pipe.execute()
        applied = sum(1 for result in results[:queued] if result == 1)
        self.stats.batches += 1
        self.stats.applied += applied
        self.stats.ignored += queued - applied
        metrics.BATCHES.inc()
        metrics.EVENTS.labels("applied").inc(applied)
        metrics.EVENTS.labels("ignored").inc(queued - applied)
        if not claimed:  # o resgatado esperou, por definição, muito: não mede a "frescura" normal
            for entry_id in applied_ids:
                metrics.APPLY_LATENCY.observe(seconds_since(id_ms(entry_id)))

    async def _claim_abandoned(self) -> None:
        for stream in list(self._races):
            cursor = "0-0"
            while True:
                try:
                    cursor, messages, _deleted = await self._redis.xautoclaim(
                        stream,
                        WORKER_GROUP,
                        self.consumer,
                        min_idle_time=self._claim_min_idle_ms,
                        start_id=cursor,
                        count=self._batch_size,
                    )
                except ResponseError as exc:
                    if "NOGROUP" in str(exc):
                        break  # o grupo some quando um replay novo apaga o stream
                    raise
                if messages:
                    logger.info("Resgatadas %d entradas pendentes de %s", len(messages), stream)
                    self.stats.claimed += len(messages)
                    metrics.CLAIMED.inc(len(messages))
                    await self._process(stream, messages, claimed=True)
                if cursor == "0-0":  # varreu o PEL inteiro
                    break

    # ------------------------------------------------------------ o laço

    async def run(self, stop: asyncio.Event) -> None:
        """Roda até `stop` ser sinalizado (SIGTERM) ou a tarefa ser cancelada."""
        loop = asyncio.get_running_loop()
        last_discover = last_claim = float("-inf")
        logger.info("Worker %s iniciado", self.consumer)
        while not stop.is_set():
            try:
                now = loop.time()
                if now - last_discover >= self._discover_interval_s:
                    await self._discover()
                    last_discover = now
                if not self._races:
                    await self._idle(stop)
                    continue
                if now - last_claim >= self._claim_interval_s:
                    await self._claim_abandoned()
                    last_claim = now
                reply = await self._redis.xreadgroup(
                    WORKER_GROUP,
                    self.consumer,
                    {stream: ">" for stream in self._races},
                    count=self._batch_size,
                    block=self._block_ms,
                )
                for stream, messages in reply or []:
                    await self._process(stream, messages)
            except ResponseError as exc:
                if "NOGROUP" not in str(exc):
                    raise
                # Um replay novo apagou o stream (e com ele o grupo), ou o stream foi apagado com o
                # worker bloqueado. Recria o grupo (a partir de "0") e segue.
                await self._ensure_groups()
            except (RedisError, OSError):
                # O cliente já tentou de novo (backoff). Não derruba o worker por uma queda do
                # Redis: espera e tenta outra vez. As pendências ficam seguras no PEL.
                logger.warning("Redis indisponível; tentando de novo", exc_info=True)
                await self._idle(stop, seconds=1.0)
        logger.info("Worker %s parou: %s", self.consumer, self.stats)

    async def _idle(self, stop: asyncio.Event, seconds: float | None = None) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), seconds or self._discover_interval_s)
