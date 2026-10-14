"""Dublês de teste do replay: relógio falso, publisher em memória, fonte em lista e lock em memória.

O relógio falso é o que permite testar o RITMO sem dormir de verdade: `wait()` avança o tempo
instantaneamente até o prazo (ou até uma ação agendada / um aviso de wake), então um replay de uma
hora roda em milissegundos e o resultado é determinístico (nada de "dorme 100 ms e torce").
"""

import asyncio
import heapq
import itertools
from collections.abc import AsyncIterator, Callable, Sequence

from redis.exceptions import ConnectionError as RedisConnectionError

from replay.events import PositionEvent, ReplayEvent


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self._actions: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()

    def monotonic(self) -> float:
        return self.now

    def at(self, when: float, action: Callable[[], None]) -> None:
        """Agenda `action` para o instante `when` (relógio falso). Roda DENTRO de uma espera do
        motor, no meio do intervalo, que é exatamente quando um usuário muda a velocidade."""
        heapq.heappush(self._actions, (when, next(self._seq), action))

    async def wait(self, timeout: float, wake: asyncio.Event) -> None:
        deadline = self.now + timeout
        while True:
            if wake.is_set():
                break
            if self._actions and self._actions[0][0] <= deadline:
                when, _, action = heapq.heappop(self._actions)
                self.now = max(self.now, when)
                action()  # pode chamar set_speed/pause/stop, que sinalizam `wake`
                continue
            self.now = deadline
            break
        await asyncio.sleep(0)  # cede o loop (outras tarefas, como o heartbeat, têm a vez)


class FakePublisher:
    """Guarda cada lote e o instante (do relógio falso) em que cada evento foi publicado."""

    def __init__(self, clock: FakeClock, *, fail_on_batch: int | None = None) -> None:
        self._clock = clock
        self._fail_on_batch = fail_on_batch
        self.batches: list[list[ReplayEvent]] = []
        self.published: list[tuple[float, ReplayEvent]] = []
        self.resets = 0

    async def reset(self) -> None:
        self.resets += 1

    async def publish_many(self, events: Sequence[ReplayEvent]) -> None:
        if self._fail_on_batch == len(self.batches) + 1:
            raise ConnectionError("redis caiu no meio do replay")
        self.batches.append(list(events))
        self.published.extend((self._clock.now, event) for event in events)

    @property
    def events(self) -> list[ReplayEvent]:
        return [event for _, event in self.published]


class ListSource:
    """Fonte de eventos em memória. `closed` prova que o motor fecha o gerador ao terminar/parar."""

    def __init__(self, events: list[ReplayEvent]) -> None:
        self._events = events
        self.t0_ms = events[0].t
        self.closed = False

    async def events(self) -> AsyncIterator[ReplayEvent]:
        try:
            for event in self._events:
                yield event
        finally:
            self.closed = True


class FakeLockRegistry:
    """O "Redis" dos locks: compartilhe UMA instância entre dois gerenciadores para simular dois
    processos disputando a mesma corrida."""

    def __init__(self) -> None:
        self.owners: dict[str, str] = {}
        self._ids = itertools.count(1)

    def lock(self, key: str, ttl_ms: int = 30) -> "FakeLock":
        return FakeLock(self, key, f"owner-{next(self._ids)}", ttl_ms)


class FakeLock:
    def __init__(self, registry: FakeLockRegistry, key: str, token: str, ttl_ms: int) -> None:
        self._registry = registry
        self._key = key
        self._token = token
        self.ttl_ms = ttl_ms
        self.renewals = 0
        self.lose_after_renewals: int | None = None  # simula o lock expirando e outro assumindo
        self.renew_errors = 0  # quantas renovações seguintes falham como se o Redis tivesse caído

    async def acquire(self) -> bool:
        return self._registry.owners.setdefault(self._key, self._token) == self._token

    async def renew(self) -> bool:
        self.renewals += 1
        if self.renew_errors > 0:
            self.renew_errors -= 1
            raise RedisConnectionError("redis indisponível")
        if self.lose_after_renewals is not None and self.renewals > self.lose_after_renewals:
            self._registry.owners[self._key] = "someone-else"
        return self._registry.owners.get(self._key) == self._token

    async def release(self) -> None:
        if self._registry.owners.get(self._key) == self._token:
            del self._registry.owners[self._key]


def position(t_ms: int, driver: str = "VER") -> PositionEvent:
    return PositionEvent(driver=driver, x=t_ms, y=-t_ms, t=t_ms)


async def settle(condition: Callable[[], bool], *, max_turns: int = 1000) -> None:
    """Cede o event loop até `condition()` ser verdadeira. Só `sleep(0)`: nenhum tempo real."""
    for _ in range(max_turns):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("a condição não ficou verdadeira")
