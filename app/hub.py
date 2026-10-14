"""Distribui os eventos de um stream de corrida aos clientes WebSocket DESTA instância da API.

O DESENHO
* UM leitor por (instância, corrida), não um por cliente: um `XREAD` (sem consumer group) bloqueado
  numa conexão Redis alimenta N clientes locais. Com 500 clientes são 1 conexão Redis, não 500.
* SEM consumer group: cada instância precisa receber TODOS os eventos (para repassá-los aos seus
  clientes). Um grupo repartiria as entradas entre as instâncias. O grupo é dos workers (que
  dividem a carga de gravar o estado); a API só PUBLICA o que vê.
* Cada cliente tem uma FILA LIMITADA. O hub nunca espera por um cliente: `put_nowait`.

CLIENTE LENTO (a estratégia, e por que):
Um cliente que não consome (celular em rede ruim, aba em segundo plano, TCP parado) acumularia
eventos para sempre. As alternativas, e por que não:
  1. fila ilimitada      -> a memória do servidor cresce sem limite: um cliente ruim derruba a API.
  2. o hub espera o cliente (backpressure) -> TODOS os clientes travam atrás do mais lento.
  3. descartar eventos em silêncio -> o cliente segue "conectado" com o estado CORROMPIDO (uma volta
     perdida deixa a classificação errada para sempre).
  4. FILA LIMITADA + DESCONECTAR quando enche (o que fazemos): a memória é limitada (N eventos por
     cliente), o hub nunca bloqueia, e o cliente é fechado com o código 1013 ("tente de novo"); ao
     reconectar ele recebe um snapshot NOVO e volta a um estado correto. Falhar de forma visível
     e recuperável, em vez de degradar em silêncio.
A fila comporta ~5,7 s de eventos a 16x (~1.760/s) e ~90 s a 1x; um cliente que fica mais que isso
sem consumir está de fato fora do ar. (Evolução possível: "conflação", guardar só a ÚLTIMA posição
de cada piloto e nunca descartar voltas; fica como melhoria, e a política simples já é segura.)
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from replay.events import MalformedEvent, entry_to_json, parse_entry, stream_key

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 10_000
READ_COUNT = 500
READ_BLOCK_MS = 1_000


@dataclass(slots=True)
class HubStats:
    dispatched: int = 0  # entradas lidas do stream e repassadas
    dropped_slow: int = 0  # clientes desconectados por não acompanhar


class Subscription:
    """A fila de UM cliente, mais o ponto do stream a partir do qual ela recebe (`join_id`)."""

    def __init__(self, maxsize: int, join_id: str) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize)
        self.join_id = join_id  # tudo o que vem DEPOIS deste ID entra na fila
        self.overflowed = False

    def offer(self, payload: str) -> bool:
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            return False
        return True

    def close(self, *, overflowed: bool = False) -> None:
        """Encerra a fila: descarta o que havia (o cliente vai ser fechado de qualquer forma) e
        deixa um `None` para o consumidor perceber."""
        self.overflowed = overflowed
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(None)


class RaceHub:
    def __init__(
        self,
        redis: Redis | None,
        year: int,
        round_number: int,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        on_empty: Callable[["RaceHub"], None] | None = None,
    ) -> None:
        self._redis = redis
        self.year, self.round_number = year, round_number
        self._stream = stream_key(year, round_number)
        self._queue_size = queue_size
        self._on_empty = on_empty
        self._subs: set[Subscription] = set()
        self._last_id = "0-0"  # o último ID já repassado aos clientes
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.closed = False
        self.stats = HubStats()

    @property
    def stream(self) -> str:
        return self._stream

    @property
    def last_id(self) -> str:
        return self._last_id

    @property
    def clients(self) -> int:
        return len(self._subs)

    def mark_ready(self, last_id: str) -> None:
        self._last_id = last_id
        self._ready.set()

    # ------------------------------------------------------------ clientes

    async def subscribe(self) -> Subscription:
        """Inscreve um cliente. Entre ler `last_id` e registrar a fila NÃO há `await`: o event loop
        não alterna, então nenhum evento é repassado "no meio". A fila recebe tudo o que vem DEPOIS
        de `join_id`; o que veio antes, o cliente busca no próprio stream (catch-up)."""
        await self._ready.wait()
        sub = Subscription(self._queue_size, self._last_id)
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subs.discard(sub)
        if not self._subs and self._on_empty is not None:
            self._on_empty(self)  # ninguém mais ouvindo: o registro encerra este leitor

    # ------------------------------------------------------------ distribuição

    def dispatch(self, entries: list[tuple[str, dict[str, str]]]) -> None:
        """Repassa entradas a todos os clientes. SÍNCRONO de propósito: não há `await`, então o
        hub nunca espera por cliente algum (e um cliente lento não atrasa os outros)."""
        for entry_id, fields in entries:
            self._last_id = entry_id
            try:
                payload = entry_to_json(parse_entry(entry_id, fields))  # serializa UMA vez...
            except MalformedEvent:
                logger.warning("Entrada ilegível ignorada: %s %s", entry_id, fields)
                continue
            self.stats.dispatched += 1
            for sub in list(self._subs):  # ...e entrega a mesma string a todos
                if not sub.offer(payload):
                    self._subs.discard(sub)  # fila cheia: este cliente não acompanha
                    sub.close(overflowed=True)
                    self.stats.dropped_slow += 1

    # ------------------------------------------------------------ o leitor

    def start(self) -> None:
        self._task = asyncio.create_task(self._read_loop(), name=f"hub-{self._stream}")

    async def _current_last_id(self) -> str:
        assert self._redis is not None
        newest = await self._redis.xrevrange(self._stream, count=1)
        return newest[0][0] if newest else "0-0"

    async def _read_loop(self) -> None:
        assert self._redis is not None
        while True:  # o ponto de partida: o fim do stream AGORA (o que veio antes é catch-up)
            try:
                self.mark_ready(await self._current_last_id())
                break
            except (RedisError, OSError):
                logger.warning("Hub %s: Redis indisponível ao iniciar", self._stream, exc_info=True)
                await asyncio.sleep(1)
        while True:
            try:
                reply = await self._redis.xread(
                    {self._stream: self._last_id}, count=READ_COUNT, block=READ_BLOCK_MS
                )
            except (RedisError, OSError):
                # O cliente Redis já tentou de novo (backoff). Não derruba o hub: espera e tenta.
                logger.warning("Hub %s: falha lendo o stream", self._stream, exc_info=True)
                await asyncio.sleep(1)
                continue
            for _name, entries in reply or []:
                self.dispatch(entries)

    async def close(self) -> None:
        self.closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for sub in list(self._subs):
            sub.close()
        self._subs.clear()


class HubRegistry:
    """Um hub por corrida, criado quando chega o 1º cliente e encerrado quando sai o último."""

    def __init__(self, redis: Redis, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._redis = redis
        self._queue_size = queue_size
        self._hubs: dict[tuple[int, int], RaceHub] = {}

    @property
    def hubs(self) -> list[RaceHub]:
        return list(self._hubs.values())

    @property
    def clients(self) -> int:
        return sum(hub.clients for hub in self._hubs.values())

    def _remove(self, hub: RaceHub) -> None:
        if self._hubs.get((hub.year, hub.round_number)) is hub:
            del self._hubs[(hub.year, hub.round_number)]
        asyncio.get_running_loop().create_task(hub.close())  # cancela o leitor (libera a conexão)

    async def subscribe(self, year: int, round_number: int) -> tuple[RaceHub, Subscription]:
        key = (year, round_number)
        while True:
            hub = self._hubs.get(key)
            if hub is None:
                hub = RaceHub(
                    self._redis,
                    year,
                    round_number,
                    queue_size=self._queue_size,
                    on_empty=self._remove,
                )
                self._hubs[key] = hub
                hub.start()
            try:
                sub = await hub.subscribe()
            except BaseException:
                # Cancelado (cliente caiu) esperando o hub ficar pronto: sem hub órfão.
                if not hub.clients:
                    self._remove(hub)
                raise
            if not hub.closed:
                return hub, sub
            # O hub foi encerrado entre obtê-lo e inscrever (o último cliente saiu nesse instante).

    async def close(self) -> None:
        hubs, self._hubs = list(self._hubs.values()), {}
        await asyncio.gather(*(hub.close() for hub in hubs))
