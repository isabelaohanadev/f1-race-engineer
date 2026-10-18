"""O consumidor de estratégia contra o Redis REAL: o ponto a ponto (laps -> resultado -> armazém), a
instância única com lock e standby, a queda (inclusive um SIGKILL de verdade), o epoch de um replay
novo, a duplicata e o embaralhamento."""

import asyncio
import contextlib
import json
import random
import signal
import subprocess
from collections.abc import Callable

import pytest
from redis.asyncio import Redis

from replay.events import laps_key
from replay.state import RaceStateStore
from strategy import metrics
from strategy.montecarlo import prepare, simulate
from strategy.payload import SCHEMA_VERSION, stable
from strategy.service.consumer import LOCK_KEY, STRATEGY_GROUP, StrategyConsumer
from strategy.service.results import events_key, result_key
from tests.integration.helpers import wait_until
from tests.integration.strategy_helpers import (
    CONFIG,
    CONSTANTS,
    TOTAL,
    FakeConstants,
    Producer,
    R,
    Y,
    has_result,
    make_consumer,
    running,
    store_of,
    stored,
    wait_result,
)

# ================================================================ ponta a ponta


async def test_each_leader_lap_ends_as_a_versioned_result_in_the_store_and_the_events_stream(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    async with running(make_consumer(redis_client)):
        await producer.start()
        for lap in range(5, 11):
            await producer.publish_lap(lap)
            await wait_result(redis_client, lap)

    entries = await redis_client.xrange(events_key(Y, R))
    assert [int(fields["lap"]) for _, fields in entries] == [5, 6, 7, 8, 9, 10]  # uma vez cada
    groups = {g["name"]: g for g in await redis_client.xinfo_groups(laps_key(Y, R))}
    assert groups[STRATEGY_GROUP]["pending"] == 0  # tudo o que foi lido foi confirmado
    assert {fields["v"] for _, fields in entries} == {str(SCHEMA_VERSION)}
    latest = await store_of(redis_client).get(Y, R)
    assert latest is not None and latest.latest == 10 and latest.epoch == 1
    payload = json.loads(latest.payload)
    assert (payload["v"], payload["lap"], payload["laps_left"]) == (SCHEMA_VERSION, 10, TOTAL - 10)
    assert json.loads(entries[-1][1]["data"]) == payload  # o evento é o mesmo documento do hash


async def test_the_stored_result_is_the_backtest_result_for_the_same_lap(
    redis_client: Redis, race_rows
):
    """Pelo Redis, com consumer group, janela e thread: o número é o do backtest."""
    producer = Producer(redis_client, race_rows)
    async with running(make_consumer(redis_client)):
        await producer.start()
        for lap in range(5, 9):
            await producer.publish_lap(lap)
            await wait_result(redis_client, lap)

    ordered = sorted(race_rows, key=lambda r: (r.driver, r.lap_number))
    inp = prepare(
        ordered, year=Y, round=R, upto_lap=8, total_laps=TOTAL,
        sim_priors=CONSTANTS.sim_priors, priors=CONSTANTS.priors, snapshot="tempo",
    )  # fmt: skip
    expected = {d.driver: round(d.win, 6) for d in simulate(inp, CONFIG).drivers}

    payload = await stored(redis_client, 8)
    assert {d["driver"]: d["win"] for d in payload["forecast"]["drivers"]} == expected


async def test_late_and_duplicate_entries_never_change_what_was_published(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    async with running(make_consumer(redis_client)):
        await producer.start()
        for lap in range(5, 8):
            await producer.publish_lap(lap)
            await wait_result(redis_client, lap)
        before = await stored(redis_client, 7)

        await producer.publish_lap(6)  # a volta 6 inteira outra vez: tudo duplicata
        await producer.publish_lap(7)
        await asyncio.sleep(0.4)

    assert await stored(redis_client, 7) == before  # o resultado de uma volta é calculado uma vez
    assert await redis_client.xlen(events_key(Y, R)) == 3  # e publicado uma vez


async def test_events_shuffled_inside_the_window_store_the_same_numbers(
    redis_client: Redis, race_rows
):
    """A mesma corrida duas vezes (a seed do sorteio é (ano, etapa, volta): tem de ser a mesma
    corrida), a segunda com as entradas de cada volta embaralhadas no stream, todas publicadas num
    lote, ou seja, dentro da janela. O que fica guardado é o mesmo."""

    async def run(shuffle: random.Random | None) -> dict[int, dict]:
        await redis_client.flushdb()
        producer = Producer(redis_client, race_rows)
        async with running(make_consumer(redis_client, wait_s=0.2)):
            await producer.start()
            for lap in range(5, 11):
                await producer.publish_lap(lap, shuffle=shuffle)
                await wait_result(redis_client, lap)
        return {lap: await stored(redis_client, lap) for lap in range(5, 11)}

    in_order = await run(None)
    shuffled = await run(random.Random(7))

    for lap in range(5, 11):
        a, b = stable(in_order[lap]), stable(shuffled[lap])
        for key in ("forecast", "alerts", "warnings", "baseline", "as_of_ms", "reliability"):
            assert a[key] == b[key], f"volta {lap}: {key} difere"


# ================================================================ instância única, lock e standby


async def test_two_consumers_one_active_and_only_the_active_calculates(
    redis_client: Redis, race_rows
):
    a, b = make_consumer(redis_client), make_consumer(redis_client)
    producer = Producer(redis_client, race_rows)
    async with running(a), running(b):
        await wait_until(lambda: _is(await_count(a, b), 1))
        await producer.start()
        for lap in range(5, 9):
            await producer.publish_lap(lap)
            await wait_result(redis_client, lap)
        standby = b if a.active else a

    assert standby.stats.submitted == 0 and standby.stats.leaderships == 0
    assert await redis_client.xlen(events_key(Y, R)) == 4


def await_count(*consumers: StrategyConsumer) -> Callable[[], int]:
    return lambda: sum(c.active for c in consumers)


async def _is(count: Callable[[], int], expected: int) -> bool:
    return count() == expected


async def test_a_clean_stop_releases_the_lock_and_the_standby_takes_over_at_once(
    redis_client: Redis,
):
    a, b = (
        make_consumer(redis_client, lock_ttl_ms=30_000),
        make_consumer(redis_client, lock_ttl_ms=30_000),
    )
    stop_a = asyncio.Event()
    task_a = asyncio.create_task(a.run(stop_a))
    await wait_until(lambda: _is(lambda: a.active, True))
    async with running(b):
        assert not b.active  # o TTL é de 30 s: sem liberar, b esperaria tudo isso

        stop_a.set()
        await asyncio.wait_for(task_a, 5)

        await wait_until(lambda: _is(lambda: b.active, True), timeout_s=3)
    assert a.stats.lost_lock == 0


async def test_the_standby_takes_over_after_a_crash_and_calculates_the_latest_lap(
    redis_client: Redis, race_rows, monkeypatch: pytest.MonkeyPatch
):
    producer = Producer(redis_client, race_rows)
    a = make_consumer(redis_client)
    b = make_consumer(redis_client)
    stop_a = asyncio.Event()
    task_a = asyncio.create_task(a.run(stop_a))
    await producer.start()
    await wait_until(lambda: _is(lambda: a.active, True))
    for lap in range(5, 9):
        await producer.publish_lap(lap)
        await wait_result(redis_client, lap)
    published_by_a = await redis_client.xlen(events_key(Y, R))

    async def crash(self) -> None:  # um processo que morre não limpa nada: o lock expira sozinho
        self.active = False

    monkeypatch.setattr(StrategyConsumer, "_step_down", crash)
    task_a.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task_a
    assert await redis_client.exists(LOCK_KEY) == 1  # a morte deixou o lock para trás
    monkeypatch.undo()

    # enquanto ninguém calcula, a corrida anda: as voltas 9 a 12 chegam ao stream
    for lap in range(9, 13):
        await producer.publish_lap(lap)
    # e a que morreu tinha lido algumas entradas sem confirmá-las (a janela entre ler e confirmar)
    ghost = await redis_client.xreadgroup(STRATEGY_GROUP, "morto", {laps_key(Y, R): ">"}, count=5)
    assert ghost and (await redis_client.xpending(laps_key(Y, R), STRATEGY_GROUP))["pending"] == 5
    async with running(b):
        await wait_result(redis_client, 12)  # b assumiu (o TTL expirou) e calculou a ÚLTIMA volta
        assert b._races[(Y, R)].tracker.duplicates == 0  # nada foi entregue duas vezes

    assert b.stats.leaderships == 1 and b.stats.rebuilt == 1
    assert not await has_result(redis_client, 10)  # as do meio não se refazem: ninguém as espera
    assert await redis_client.xlen(events_key(Y, R)) == published_by_a + 1
    groups = {g["name"]: g for g in await redis_client.xinfo_groups(laps_key(Y, R))}
    assert groups[STRATEGY_GROUP]["pending"] == 0  # as pendências de quem morreu foram confirmadas
    # o grupo foi posicionado no fim do que a reconstrução leu: nada é entregue duas vezes
    info = await redis_client.xinfo_stream(laps_key(Y, R))
    assert groups[STRATEGY_GROUP]["last-delivered-id"] == info["last-generated-id"]


async def test_after_taking_over_the_new_active_keeps_consuming_new_laps(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    await producer.start()
    for lap in range(5, 9):
        await producer.publish_lap(lap)

    async with running(make_consumer(redis_client)):  # chega com a corrida já em andamento
        await wait_result(redis_client, 8)
        await producer.publish_lap(9)
        await producer.publish_lap(10)
        await wait_result(redis_client, 10)

    assert not await has_result(redis_client, 6)  # só a última volta da reconstrução
    assert await has_result(redis_client, 10)


async def test_a_restart_does_not_recalculate_a_lap_that_already_has_its_result(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    first = make_consumer(redis_client)
    async with running(first):
        await producer.start()
        for lap in range(5, 9):
            await producer.publish_lap(lap)
            await wait_result(redis_client, lap)

    second = make_consumer(redis_client)  # reinicia: reconstrói e acha o resultado da volta 8
    async with running(second):
        await wait_until(lambda: _is(lambda: second.stats.rebuilt == 1, True))
        await asyncio.sleep(0.3)

    assert second.stats.submitted == 0
    assert await redis_client.xlen(events_key(Y, R)) == 4


async def test_a_race_that_already_finished_is_not_calculated_again_when_a_consumer_arrives(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    await producer.start()
    for lap in range(5, 12):
        await producer.publish_lap(lap)
    await producer.finish()

    consumer = make_consumer(redis_client)
    async with running(consumer):
        await wait_until(lambda: _is(lambda: consumer.stats.rebuilt == 1, True))
        await asyncio.sleep(0.3)

    assert consumer.stats.submitted == 0 and await redis_client.exists(events_key(Y, R)) == 0


async def test_a_result_of_another_epoch_is_dropped_at_delivery(redis_client: Redis, race_rows):
    from strategy.live import RaceTracker, compute
    from tests.strategy.synthetic import stream_entries

    consumer = make_consumer(redis_client)
    async with running(consumer):
        await wait_until(lambda: _is(lambda: consumer.active, True))
        producer = Producer(redis_client, race_rows, epoch=2)
        await producer.start()
        await wait_until(lambda: _is(lambda: consumer.stats.rebuilt == 1, True))
        tracker = RaceTracker(Y, R)
        for entry in stream_entries(race_rows, total_laps=TOTAL, epoch=1):
            tracker.apply(entry, 0.0)
        (trigger, *_) = [t for t in tracker.due(99.0) if t.lap == 8]
        computation = compute(trigger, CONSTANTS, sim_config=CONFIG)

        await consumer._deliver((Y, R, 1, 8), computation)  # o consumidor já está no epoch 2

    assert consumer.stats.dropped == 1 and not await has_result(redis_client, 8, epoch=1)


async def test_a_consumer_that_loses_the_lock_stops_calculating_and_delivering(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    a = make_consumer(redis_client, lock_ttl_ms=300)
    async with running(a):
        await producer.start()
        await producer.publish_lap(5)
        await wait_result(redis_client, 5)

        await redis_client.set(LOCK_KEY, "de-outro", px=60_000)  # outro processo assumiu
        await wait_until(lambda: _is(lambda: a.active, False), timeout_s=3)

        await producer.publish_lap(6)
        await asyncio.sleep(0.5)

    assert a.stats.lost_lock == 1
    assert not await has_result(redis_client, 6)  # perdeu o lock: não entrega mais nada


async def test_a_result_calculated_after_losing_the_lock_is_dropped_at_delivery(
    redis_client: Redis, race_rows
):
    from strategy.live import RaceTracker, compute
    from tests.strategy.synthetic import stream_entries

    consumer = make_consumer(redis_client)
    async with running(consumer):
        producer = Producer(redis_client, race_rows)
        await producer.start()
        await wait_until(lambda: _is(lambda: consumer.stats.rebuilt == 1, True))
        tracker = RaceTracker(Y, R)
        for entry in stream_entries(race_rows, total_laps=TOTAL):
            tracker.apply(entry, 0.0)
        (trigger, *_) = [t for t in tracker.due(99.0) if t.lap == 8]
        computation = compute(trigger, CONSTANTS, sim_config=CONFIG)

        consumer._lost.set()  # perdeu o lock com o cálculo em voo
        await consumer._deliver((Y, R, 1, 8), computation)

    assert consumer.stats.dropped == 1 and not await has_result(redis_client, 8)


async def test_a_calculation_with_too_few_cars_is_counted_as_skipped_and_publishes_nothing(
    redis_client: Redis, race_rows
):
    two_cars = [r for r in race_rows if r.driver in ("R", "A")]
    producer = Producer(redis_client, two_cars)
    consumer = make_consumer(redis_client)
    async with running(consumer):
        await producer.start()
        for lap in range(5, 8):
            await producer.publish_lap(lap)
        await wait_until(lambda: _is(lambda: consumer.stats.skipped >= 1, True))

    assert consumer.stats.written == 0 and await redis_client.exists(events_key(Y, R)) == 0


async def test_a_duplicate_does_not_refresh_the_age_of_the_last_result(
    redis_client: Redis, race_rows
):
    from strategy.live import RaceTracker, compute
    from tests.strategy.synthetic import stream_entries

    consumer = make_consumer(redis_client)
    async with running(consumer):
        await Producer(redis_client, race_rows).start()
        await wait_until(lambda: _is(lambda: consumer.stats.rebuilt == 1, True))
        tracker = RaceTracker(Y, R)
        for entry in stream_entries(race_rows, total_laps=TOTAL):
            tracker.apply(entry, 0.0)
        (trigger, *_) = [t for t in tracker.due(99.0) if t.lap == 8]
        computation = compute(trigger, CONSTANTS, sim_config=CONFIG)

        await consumer._deliver((Y, R, 1, 8), computation)
        metrics._last_delivered_at = 1.0  # marca o instante do último resultado novo
        await consumer._deliver((Y, R, 1, 8), computation)  # o mesmo, outra vez

    assert (consumer.stats.written, consumer.stats.exists) == (1, 1)
    assert metrics._last_delivered_at == 1.0  # uma repetição não é um resultado novo


async def test_the_read_never_blocks_forever_when_a_window_has_already_closed(redis_client: Redis):
    """BLOCK 0 é "para sempre": se a janela já passou (a carga das constantes demorou), a leitura
    tem de acordar logo, não esperar uma entrada nova que pode nunca vir."""
    from strategy.live import RaceTracker
    from strategy.service.consumer import RaceRuntime

    clock = [100.0]
    consumer = StrategyConsumer(redis_client, FakeConstants(), clock=lambda: clock[0], block_ms=500)
    tracker = RaceTracker(Y, R, wait_s=1.0)
    tracker._pending[8] = 99.0  # venceu há 1 s
    consumer._races[(Y, R)] = RaceRuntime(Y, R, "laps", tracker, None, None)  # type: ignore[arg-type]

    assert consumer._block_timeout_ms() == 1
    tracker._pending[8] = 100.2
    assert consumer._block_timeout_ms() == 201  # até o fim da janela
    tracker._pending.clear()
    assert consumer._block_timeout_ms() == 500


# ================================================================ replay novo (epoch)


async def test_a_new_replay_resets_the_results_and_the_old_epoch_is_fenced(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    state = RaceStateStore(redis_client)
    consumer = make_consumer(redis_client)
    async with running(consumer):
        await producer.start()
        await state.new_epoch(Y, R)  # epoch 1, como o `start` do replay faz
        for lap in range(5, 9):
            await producer.publish_lap(lap)
            await wait_result(redis_client, lap)

        runtime = consumer._races[(Y, R)]
        runtime.alerts._on.add(("A", "R"))  # um alerta ligado no replay 1
        old_runner = runtime.runner

        # o replay recomeça: o epoch sobe, o stream de voltas é apagado e republicado
        epoch = await state.new_epoch(Y, R)
        second = Producer(redis_client, race_rows, epoch=epoch)
        await second.publisher.reset()
        await second.start()
        for lap in range(5, 7):
            await second.publish_lap(lap)
            await wait_result(redis_client, lap, epoch=epoch)

    assert runtime.alerts.active() == set()  # a histerese de um replay não vale no seguinte
    assert runtime.runner is not old_runner  # e o que estava calculando foi cancelado
    result = await store_of(redis_client).get(Y, R)
    assert result is not None and result.epoch == 2 and result.latest == 6
    assert (
        await store_of(redis_client).get(Y, R, 8) is None
    )  # a volta 8 do replay 1 não existe mais
    assert not await has_result(redis_client, 8, epoch=1)


async def test_an_entry_of_an_old_replay_is_ignored_by_the_consumer(redis_client: Redis, race_rows):
    producer = Producer(redis_client, race_rows, epoch=2)
    stale = Producer(redis_client, race_rows, epoch=1)
    consumer = make_consumer(redis_client)
    async with running(consumer):
        await producer.start()
        await producer.publish_lap(5)
        await wait_result(redis_client, 5, epoch=2)

        await stale.publish_lap(6)  # um evento atrasado de um replay antigo
        await asyncio.sleep(0.4)

    assert not await has_result(redis_client, 6, epoch=1)
    assert not await has_result(redis_client, 6, epoch=2)


# ================================================================ métricas


async def test_the_active_gauge_and_the_result_age_follow_the_consumer(
    redis_client: Redis, race_rows
):
    producer = Producer(redis_client, race_rows)
    consumer = make_consumer(redis_client)
    async with running(consumer):
        await wait_until(lambda: _is(lambda: consumer.active, True))
        assert metrics.CONSUMER_ACTIVE._value.get() == 1  # noqa: SLF001
        await producer.start()
        await producer.publish_lap(5)
        await wait_result(redis_client, 5)
        await wait_until(
            lambda: _is(lambda: metrics.last_result_age() == metrics.last_result_age(), True)
        )
        assert metrics.last_result_age() < 5.0  # o resultado acabou de ser gravado
    assert metrics.CONSUMER_ACTIVE._value.get() == 0  # noqa: SLF001


# ================================================================ a queda de verdade: SIGKILL


async def test_a_real_sigkill_of_the_active_process_is_recovered_by_the_standby(
    redis_client: Redis, race_rows, consumer_subprocess: subprocess.Popen
):
    producer = Producer(redis_client, race_rows)
    await producer.start()
    standby = make_consumer(redis_client, lock_ttl_ms=800)
    await wait_until(lambda: _lock_held(redis_client), timeout_s=30)  # o subprocesso é o ativo
    for lap in range(5, 8):
        await producer.publish_lap(lap)
        await wait_result(
            redis_client, lap
        )  # o processo de verdade calculou (banco vazio: padrões)

    async with running(standby):
        await asyncio.sleep(0.3)
        assert not standby.active  # o lock é do subprocesso

        consumer_subprocess.send_signal(signal.SIGKILL)  # a queda: nada é liberado
        consumer_subprocess.wait(10)
        for lap in range(8, 11):
            await producer.publish_lap(lap)
        await wait_result(redis_client, 10)  # a standby assumiu pelo TTL e calculou a última volta

    assert standby.stats.leaderships == 1
    assert [json.loads(f["data"])["lap"] for _, f in await redis_client.xrange(events_key(Y, R))][
        -1
    ] == 10


async def _lock_held(redis: Redis) -> bool:
    return await redis.exists(LOCK_KEY) == 1


def test_the_hash_key_is_the_one_the_api_reads():
    assert result_key(Y, R) == "strategy:result:2024:10"
