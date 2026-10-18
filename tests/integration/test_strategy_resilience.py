"""O consumidor de estratégia nos cantos: entradas ruins, duas corridas, o lock que se perde e
volta, falhas do Redis, três instâncias e um processo congelado que acorda. Redis REAL; as falhas
do Redis são injetadas no cliente (não se derruba o servidor: ver docs/estrategia-ao-vivo.md)."""

import asyncio
import json
import signal
import threading
import time
from collections.abc import Callable

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from replay.events import LapCompletedEvent, laps_key
from replay.state import RaceStateStore
from strategy import metrics
from strategy.service import consumer as consumer_module
from strategy.service.consumer import LOCK_KEY, STRATEGY_GROUP, StrategyConsumer
from strategy.service.results import events_key
from tests.integration.helpers import wait_until
from tests.integration.strategy_helpers import (
    Producer,
    R,
    Y,
    has_result,
    make_consumer,
    running,
    wait_result,
)


def until(predicate: Callable[[], bool]) -> Callable[[], "asyncio.Future[bool]"]:
    async def check() -> bool:
        return predicate()

    return check


def malformed_total() -> float:
    return metrics.LAPS.labels("malformed")._value.get()  # noqa: SLF001


async def pending(redis: Redis, race: tuple[int, int] = (Y, R)) -> int:
    return (await redis.xpending(laps_key(*race), STRATEGY_GROUP))["pending"]


def bad(**fields: str) -> dict[str, str]:
    return {"type": "lap_completed", "driver": "VER", "lap": "3", "t": "1", "epoch": "1"} | fields


BAD_ENTRIES = [
    bad(lap="abc"),  # a volta não é um número
    bad(pit_in="talvez"),  # a flag não é 0 nem 1
    {k: v for k, v in bad().items() if k != "epoch"},  # sem epoch
    {"type": "inventado", "epoch": "1"},  # um tipo que não existe
]


# ================================================================ o stream de voltas


async def test_laps_without_the_session_clock_never_produce_a_calculation(
    redis_client: Redis, race_rows
):
    """Uma corrida ingerida sem `session_time_ms`: sem relógio não há "estado no instante t"."""
    consumer = make_consumer(redis_client)
    publisher = Producer(redis_client, race_rows).publisher
    async with running(consumer):
        await RaceStateStore(redis_client).register_stream(Y, R)
        await publisher.publish_many(
            [
                LapCompletedEvent(
                    r.driver, r.lap_number, 1_700_000_000_000 + i, r.lap_time_ms, r.compound
                )
                for i, r in enumerate(sorted(race_rows, key=lambda r: r.lap_number))
                if r.lap_number <= 12
            ]
        )
        await wait_until(
            until(lambda: bool(consumer._races) and consumer._races[(Y, R)].tracker.rows)
        )
        await asyncio.sleep(0.4)

    assert consumer.stats.submitted == 0 and consumer.active is False
    assert await redis_client.exists(events_key(Y, R)) == 0  # nada calculado, nada publicado


async def test_unreadable_entries_are_acknowledged_counted_and_never_stop_the_consumer(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    before = malformed_total()
    async with running(make_consumer(redis_client)):
        await producer.start()
        for fields in BAD_ENTRIES:
            await redis_client.xadd(laps_key(Y, R), fields)
        for lap in range(5, 8):
            await producer.publish_lap(lap)
        await wait_result(redis_client, 7)

        assert malformed_total() - before == len(BAD_ENTRIES)
        assert await pending(redis_client) == 0  # confirmadas: uma mensagem "veneno" não volta


async def test_a_rebuild_skips_the_unreadable_entries_that_are_already_in_the_stream(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    await producer.start()
    for fields in BAD_ENTRIES:
        await redis_client.xadd(laps_key(Y, R), fields)
    for lap in range(5, 9):
        await producer.publish_lap(lap)

    consumer = make_consumer(redis_client)
    async with running(consumer):
        await wait_result(redis_client, 8)  # a reconstrução leu tudo, ilegíveis incluídos

    assert consumer.stats.rebuilt == 1 and await pending(redis_client) == 0


async def test_a_second_replay_of_the_same_race_starts_the_laps_stream_clean_in_the_real_chain(
    redis_client: Redis, synthetic_race
):
    from replay.engine import ReplayState
    from replay.lock import lock_key
    from replay.manager import ReplayManager
    from tests.integration.helpers import RACE
    from tests.replay.fakes import FakeClock

    manager = ReplayManager(synthetic_race, redis_client, clock=FakeClock(), page_size=100)

    async def play() -> int:
        await manager.start(*RACE, speed=16)

        async def finished() -> bool:
            status = manager.status(*RACE)
            return status is not None and status.state is ReplayState.FINISHED

        await wait_until(finished, timeout_s=10)
        await wait_until(lambda: _free(redis_client, lock_key(*RACE)), timeout_s=5)
        return await redis_client.xlen(laps_key(*RACE))

    first = await play()
    second = await play()

    entries = await redis_client.xrange(laps_key(*RACE))
    assert first == second > 0  # sem o DEL no início do segundo replay seria o dobro
    assert {fields["epoch"] for _, fields in entries} == {"2"}  # e só há entradas do replay 2


async def _free(redis: Redis, key: str) -> bool:
    return await redis.exists(key) == 0


async def test_a_rebuild_that_spans_several_pages_applies_every_entry_once(
    redis_client: Redis, race_rows, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        consumer_module, "REBUILD_PAGE", 37
    )  # não divide o total: o cursor "(" conta
    producer = Producer(redis_client, race_rows)
    await producer.start()
    for lap in range(5, 13):
        await producer.publish_lap(lap)
    published = {
        (fields["driver"], int(fields["lap"]))
        for _, fields in await redis_client.xrange(laps_key(Y, R))
        if fields["type"] == "lap_completed"
    }
    assert await redis_client.xlen(laps_key(Y, R)) > 3 * 37  # várias páginas

    consumer = make_consumer(redis_client)
    async with running(consumer):
        await wait_until(until(lambda: consumer.stats.rebuilt == 1))
        tracker = consumer._races[(Y, R)].tracker

        assert set(tracker.rows) == published and len(tracker.rows) == len(published)
        assert tracker.duplicates == 0  # nenhuma entrada lida duas vezes na virada de página


async def test_two_races_at_once_each_get_their_own_state_runner_and_results(
    redis_client: Redis, race_rows
):
    one = Producer(redis_client, race_rows, round_number=10)
    two = Producer(redis_client, race_rows, round_number=11)
    consumer = make_consumer(redis_client)
    async with running(consumer):
        await one.start()
        await two.start()
        for lap in range(5, 10):
            await one.publish_lap(lap)
            await two.publish_lap(lap)
        await wait_result(redis_client, 9, rnd=10)
        await wait_result(redis_client, 9, rnd=11)

        runtimes = consumer._races
        assert set(runtimes) == {(Y, 10), (Y, 11)}
        assert runtimes[(Y, 10)].runner is not runtimes[(Y, 11)].runner  # um executor por corrida
        assert runtimes[(Y, 10)].tracker is not runtimes[(Y, 11)].tracker

    assert consumer.stats.written >= 2


async def test_a_race_registered_before_any_lap_exists_is_waited_for_not_an_error(
    redis_client: Redis, race_rows
):
    await RaceStateStore(redis_client).register_stream(Y, R)  # o replay anunciou, nada publicado
    consumer = make_consumer(redis_client)
    async with running(consumer):
        await wait_until(until(lambda: consumer.stats.rebuilt == 1))
        assert consumer.active and not consumer._races[(Y, R)].tracker.rows

        producer = Producer(redis_client, race_rows)
        await producer.start()
        for lap in range(5, 8):
            await producer.publish_lap(lap)
        await wait_result(redis_client, 7)


# ================================================================ lock, standby e falhas


class ScriptedLock:
    """Um lock cujo `renew` segue um roteiro: um erro do Redis, `False` (perdeu), ou `True`."""

    def __init__(self, ttl_ms: int, script: list) -> None:
        self.ttl_ms, self._script, self.calls = ttl_ms, script, 0

    async def renew(self) -> bool:
        self.calls += 1
        step = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(step, Exception):
            raise step
        return step


async def heartbeat_for(redis: Redis, script: list, wait_s: float) -> StrategyConsumer:
    consumer = make_consumer(redis, lock_ttl_ms=300)
    consumer._lock = ScriptedLock(300, script)  # type: ignore[assignment]
    consumer._lost = asyncio.Event()
    task = asyncio.create_task(consumer._heartbeat())
    await asyncio.sleep(wait_s)
    task.cancel()
    return consumer


async def test_the_heartbeat_tolerates_redis_errors_inside_the_ttl(redis_client: Redis):
    # renova a cada 0,1 s; 2 falhas = 0,2 s, abaixo do TTL de 0,3 s: o lock ainda é dele
    consumer = await heartbeat_for(redis_client, [RedisError("x")] * 2 + [True], wait_s=0.5)

    assert not consumer._lost.is_set() and consumer.stats.lost_lock == 0


async def test_the_heartbeat_gives_up_when_redis_is_gone_for_longer_than_the_ttl(
    redis_client: Redis,
):
    consumer = await heartbeat_for(redis_client, [RedisError("x")], wait_s=0.8)

    assert consumer._lost.is_set() and consumer.stats.lost_lock == 1  # o lock já pode ter expirado


async def test_the_heartbeat_stops_at_once_when_the_lock_belongs_to_someone_else(
    redis_client: Redis,
):
    consumer = await heartbeat_for(redis_client, [True, False], wait_s=0.4)

    assert consumer._lost.is_set() and consumer.stats.lost_lock == 1


async def test_a_standby_keeps_trying_when_redis_fails_on_the_lock(
    redis_client: Redis, monkeypatch: pytest.MonkeyPatch
):
    consumer = make_consumer(redis_client)
    real, calls = consumer._lock.acquire, []

    async def flaky() -> bool:
        calls.append(1)
        if len(calls) <= 3:
            raise RedisConnectionError("fora do ar")
        return await real()

    monkeypatch.setattr(consumer._lock, "acquire", flaky)
    async with running(consumer):
        await wait_until(until(lambda: consumer.active), timeout_s=5)

    assert len(calls) >= 4  # tentou de novo, sem derrubar o laço


async def test_an_active_consumer_survives_a_failed_read_and_keeps_its_leadership(
    redis_client: Redis, race_rows, monkeypatch: pytest.MonkeyPatch
):
    producer = Producer(redis_client, race_rows)
    consumer = make_consumer(redis_client)
    real, failures = redis_client.xreadgroup, []

    async def flaky(*args, **kwargs):
        if len(failures) < 1:
            failures.append(1)
            raise RedisConnectionError("conexão caiu no meio da leitura")
        return await real(*args, **kwargs)

    async with running(consumer):
        await producer.start()
        await wait_until(until(lambda: consumer.stats.rebuilt == 1))
        monkeypatch.setattr(redis_client, "xreadgroup", flaky)
        for lap in range(5, 8):
            await producer.publish_lap(lap)
        await wait_result(redis_client, 7)  # depois de 1 s de espera, a leitura volta e segue

        assert failures and consumer.active and consumer.stats.leaderships == 1


async def start_all(*consumers: StrategyConsumer):
    stops = [asyncio.Event() for _ in consumers]
    tasks = [asyncio.create_task(c.run(stop)) for c, stop in zip(consumers, stops, strict=True)]
    return stops, tasks


async def stop_all(stops, tasks) -> None:
    for stop in stops:
        stop.set()
    for task in tasks:
        if not task.done():
            await asyncio.wait_for(task, 10)


def crash(consumer: StrategyConsumer) -> None:
    """Um processo que morre não limpa nada: o lock fica para expirar e nada é confirmado."""

    async def nothing() -> None:
        consumer.active = False

    consumer._step_down = nothing  # type: ignore[method-assign]


async def test_with_three_instances_exactly_one_takes_over_after_a_crash_never_two_at_once(
    redis_client: Redis, race_rows
):
    consumers = [make_consumer(redis_client) for _ in range(3)]
    stops, tasks = await start_all(*consumers)
    await wait_until(until(lambda: sum(c.active for c in consumers) == 1))
    leader = next(i for i, c in enumerate(consumers) if c.active)
    crash(consumers[leader])
    tasks[leader].cancel()
    survivors = [c for i, c in enumerate(consumers) if i != leader]

    most_at_once = 0
    deadline = time.monotonic() + 2.5  # TTL de 0,6 s: bastante para a corrida pelo lock acabar
    while time.monotonic() < deadline:
        most_at_once = max(most_at_once, sum(c.active for c in survivors))
        await asyncio.sleep(0.01)
    await stop_all(stops, tasks)

    assert most_at_once == 1  # nunca dois ativos, e alguém assumiu
    assert sum(c.stats.leaderships for c in survivors) == 1


async def test_the_standby_takes_over_within_the_ttl_plus_a_poll_after_a_crash(redis_client: Redis):
    a, b = make_consumer(redis_client), make_consumer(redis_client)
    stops, tasks = await start_all(a)
    await wait_until(until(lambda: a.active))
    stops_b, tasks_b = await start_all(b)

    crash(a)
    tasks[0].cancel()
    crashed_at = time.monotonic()
    await wait_until(until(lambda: b.active), timeout_s=5)
    elapsed = time.monotonic() - crashed_at
    await stop_all(stops + stops_b, tasks_b)

    # TTL de 0,6 s + o intervalo de tentativa de 0,05 s + folga de agendamento
    assert elapsed < 0.6 + 0.05 + 0.6, elapsed


async def test_a_consumer_that_lost_the_lock_comes_back_as_standby_and_leads_again(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    await producer.start()
    consumer = make_consumer(redis_client, lock_ttl_ms=300)
    async with running(consumer):
        await wait_until(until(lambda: consumer.active))
        await redis_client.set(LOCK_KEY, "outro", px=60_000)  # outro processo assumiu
        await wait_until(until(lambda: not consumer.active), timeout_s=3)

        await redis_client.delete(LOCK_KEY)  # e foi embora
        await wait_until(until(lambda: consumer.active), timeout_s=3)
        for lap in range(5, 8):
            await producer.publish_lap(lap)
        await wait_result(redis_client, 7)  # reconstruiu o estado e voltou a entregar

    assert consumer.stats.leaderships == 2 and consumer.stats.lost_lock == 1
    assert consumer.stats.rebuilt == 2  # cada liderança reconstrói do stream


async def test_a_clean_stop_in_the_middle_of_a_simulation_exits_fast_and_delivers_nothing(
    redis_client: Redis, race_rows, monkeypatch: pytest.MonkeyPatch
):
    from strategy import live
    from strategy.montecarlo import SimulationCancelled

    started = threading.Event()

    def slow_compute(trigger, constants, *, should_stop=lambda: False, **kwargs):
        started.set()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:  # uma simulação que não termina sozinha
            if should_stop():
                raise SimulationCancelled
            time.sleep(0.01)

    monkeypatch.setattr(live, "compute", slow_compute)
    producer = Producer(redis_client, race_rows)
    consumer = make_consumer(redis_client)
    stop = asyncio.Event()
    task = asyncio.create_task(consumer.run(stop))
    await producer.start()
    await producer.publish_lap(5)
    await wait_until(until(started.is_set), timeout_s=10)

    stopped_at = time.monotonic()
    stop.set()
    await asyncio.wait_for(task, 5)

    assert time.monotonic() - stopped_at < 2.0  # cancelou o que estava em voo
    assert await redis_client.exists(LOCK_KEY) == 0  # e liberou o lock
    assert not await has_result(redis_client, 5) and await redis_client.xlen(events_key(Y, R)) == 0


# ============================================================ entradas levadas por outro consumidor


async def test_entries_taken_by_another_consumer_of_the_group_are_noticed_and_the_state_rebuilt(
    redis_client: Redis, race_rows, monkeypatch: pytest.MonkeyPatch
):
    """Um zumbi que acorda e lê do grupo leva entradas que a ativa nunca vê: o estado dela ficaria
    com um buraco, em silêncio. O lote seguinte denuncia a lacuna e o estado é refeito do stream."""
    producer = Producer(redis_client, race_rows)
    consumer = make_consumer(redis_client)
    real, gate = redis_client.xreadgroup, asyncio.Event()
    gate.set()

    async def gated(*args, **kwargs):  # segura a leitura da ativa enquanto o "zumbi" lê
        await gate.wait()
        return await real(*args, **kwargs)

    monkeypatch.setattr(redis_client, "xreadgroup", gated)
    async with running(consumer):
        await producer.start()
        await wait_until(until(lambda: consumer.stats.rebuilt == 1))
        await producer.publish_lap(5)
        await wait_result(redis_client, 5)

        gate.clear()
        await asyncio.sleep(0.2)  # a ativa chega ao portão
        await producer.publish_lap(6)
        taken = await real(STRATEGY_GROUP, "zumbi", {laps_key(Y, R): ">"}, count=4)
        stolen = sum(len(entries) for _, entries in taken)
        gate.set()
        await producer.publish_lap(7)  # a próxima leitura da ativa já salta o que o zumbi levou
        await wait_result(redis_client, 7)

        tracker = consumer._races[(Y, R)].tracker
        published = {
            (f["driver"], int(f["lap"]))
            for _, f in await redis_client.xrange(laps_key(Y, R))
            if f["type"] == "lap_completed"
        }

    assert stolen == 4 and consumer.stats.gap_rebuilds == 1
    assert set(tracker.rows) == published  # nenhuma volta ficou de fora do estado


# ================================================================ o processo congelado que acorda


async def test_a_frozen_active_process_that_wakes_up_does_not_split_the_brain(
    redis_client: Redis, race_rows, consumer_subprocess
):
    """SIGSTOP além do TTL, a standby assume, SIGCONT. O zumbi não pode retomar o lock nem
    publicar uma volta duas vezes, e o estado de quem lidera tem de ficar completo mesmo que o zumbi
    leia do grupo ao acordar."""
    producer = Producer(redis_client, race_rows)
    await producer.start()
    await wait_until(until_lock(redis_client), timeout_s=30)  # o subprocesso é o ativo
    for lap in range(5, 8):
        await producer.publish_lap(lap)
        await wait_result(redis_client, lap)

    standby = make_consumer(redis_client, lock_ttl_ms=800)
    async with running(standby):
        consumer_subprocess.send_signal(signal.SIGSTOP)  # congela: nem renova o lock
        await wait_until(until(lambda: standby.active), timeout_s=10)
        for lap in range(8, 11):
            await producer.publish_lap(lap)
        await wait_result(redis_client, 10)

        consumer_subprocess.send_signal(signal.SIGCONT)  # o zumbi acorda
        for lap in range(11, 14):
            await producer.publish_lap(lap)
        await wait_result(redis_client, 13)
        await asyncio.sleep(1.5)  # tempo de o zumbi tentar renovar e perceber

        holder = await redis_client.get(LOCK_KEY)
        laps_published = [
            int(json.loads(f["data"])["lap"])
            for _, f in await redis_client.xrange(events_key(Y, R))
        ]
        tracker = standby._races[(Y, R)].tracker
        expected = {
            (f["driver"], int(f["lap"]))
            for _, f in await redis_client.xrange(laps_key(Y, R))
            if f["type"] == "lap_completed"
        }
        assert holder == standby._lock._token  # só a standby segura o lock
        assert standby.active

    assert len(laps_published) == len(set(laps_published))  # nenhuma volta publicada duas vezes
    assert laps_published[-1] == 13
    assert set(tracker.rows) == expected  # o estado de quem lidera ficou completo


def until_lock(redis: Redis):
    async def held() -> bool:
        return await redis.exists(LOCK_KEY) == 1

    return held
