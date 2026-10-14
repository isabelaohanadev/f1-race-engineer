"""Testes do motor: ordem, ritmo (1x/4x/16x), troca de velocidade, pausa/retomada, parada e falha.

Todos usam o FakeClock: o tempo avança instantaneamente, então "uma hora de corrida a 1x" roda em
milissegundos e os instantes de publicação são exatos (nenhum `sleep` real, nenhum teste flaky).
"""

import asyncio

import pytest

from replay.engine import ReplayEngine, ReplayState
from replay.events import LapCompletedEvent, RaceStatusEvent
from tests.replay.fakes import FakeClock, FakePublisher, ListSource, position, settle

# Tolerância dos instantes: 1 µs (o piso do motor). Com lookahead=0 a publicação é exata.
EXACT = pytest.approx


def make_engine(events, clock, *, speed=1.0, lookahead_s=0.0, max_batch=500, publisher=None):
    publisher = publisher or FakePublisher(clock)
    source = ListSource(events)
    engine = ReplayEngine(
        source, publisher, clock=clock, speed=speed, lookahead_s=lookahead_s, max_batch=max_batch
    )
    return engine, publisher, source


def offsets(publisher: FakePublisher, start: float) -> list[float]:
    """Quando (em segundos desde o início do replay) cada evento foi publicado."""
    return [when - start for when, _ in publisher.published]


# ---------------------------------------------------------------- ordem


async def test_events_are_published_in_the_order_the_source_yields_them():
    clock = FakeClock()
    events = [position(0), position(10, "NOR"), LapCompletedEvent("VER", 1, 10), position(2000)]
    engine, publisher, _ = make_engine(events, clock)

    await engine.run()

    assert publisher.events == events  # mesma ordem, nenhum perdido, nenhum duplicado
    assert engine.state is ReplayState.FINISHED


async def test_source_generator_is_closed_when_the_replay_ends():
    clock = FakeClock()
    engine, _, source = make_engine([position(0), position(5)], clock)

    await engine.run()

    assert source.closed  # senão a leitura do banco (prefetch) ficaria órfã


# ---------------------------------------------------------------- ritmo


@pytest.mark.parametrize("speed", [1.0, 4.0, 16.0])
async def test_each_event_is_published_at_its_race_time_divided_by_speed(speed):
    clock = FakeClock()
    start = clock.now
    race_offsets_ms = [0, 1000, 2500, 4000, 10_000]
    engine, publisher, _ = make_engine([position(t) for t in race_offsets_ms], clock, speed=speed)

    await engine.run()

    assert offsets(publisher, start) == EXACT([t / 1000 / speed for t in race_offsets_ms], abs=1e-6)


async def test_changing_speed_midway_does_not_accumulate_drift():
    """1 evento por segundo de corrida; aos 10 s de relógio o usuário passa de 1x para 16x.
    Cada evento tem de sair EXATAMENTE onde a fórmula manda, sem atraso acumulado."""
    clock = FakeClock()
    start = clock.now
    events = [position(t * 1000) for t in range(41)]
    engine, publisher, _ = make_engine(events, clock, speed=1.0)
    clock.at(start + 10, lambda: engine.set_speed(16.0))

    await engine.run()

    expected = [
        t if t <= 10 else 10 + (t - 10) / 16  # até 10 s a 1x; depois cada s de corrida = 1/16 s
        for t in range(41)
    ]
    assert offsets(publisher, start) == EXACT(expected, abs=1e-6)


async def test_changing_speed_interrupts_a_wait_already_in_progress():
    """O próximo evento está a 60 s de corrida. Aos 5 s o usuário pede 16x: a espera em andamento
    (calculada a 1x) tem de ser interrompida e recalculada, não terminar 55 s depois."""
    clock = FakeClock()
    start = clock.now
    engine, publisher, _ = make_engine([position(0), position(60_000)], clock, speed=1.0)
    clock.at(start + 5, lambda: engine.set_speed(16.0))

    await engine.run()

    assert offsets(publisher, start) == EXACT([0, 5 + 55 / 16], abs=1e-6)


async def test_a_late_wakeup_does_not_delay_the_following_events():
    """Se uma espera estoura em 300 ms, os eventos seguintes NÃO herdam o atraso: o motor publica o
    que já venceu e volta ao horário certo. Simulado com um relógio que "dorme demais" uma vez."""

    class OversleepingClock(FakeClock):
        slept_too_long = False

        async def wait(self, timeout, wake):
            await super().wait(timeout, wake)
            if not self.slept_too_long:
                self.slept_too_long = True
                self.now += 0.3  # o agendador do SO demorou 300 ms para nos acordar

    clock = OversleepingClock()
    start = clock.now
    engine, publisher, _ = make_engine([position(t * 1000) for t in range(4)], clock)

    await engine.run()

    # t=1 s foi publicado com 300 ms de atraso; t=2 s e t=3 s voltam exatamente ao horário.
    assert offsets(publisher, start) == EXACT([0, 1.3, 2.0, 3.0], abs=1e-6)


async def test_events_within_the_lookahead_go_out_in_a_single_batch():
    clock = FakeClock()
    events = [position(t) for t in (0, 2, 4, 6, 1000)]  # ms
    engine, publisher, _ = make_engine(events, clock, lookahead_s=0.005)

    await engine.run()

    # 0, 2 e 4 ms vencem dentro de 5 ms: um lote só. O de 6 ms e o de 1 s saem sozinhos.
    assert [[e.t for e in batch] for batch in publisher.batches] == [[0, 2, 4], [6], [1000]]


async def test_batch_size_is_capped():
    clock = FakeClock()
    engine, publisher, _ = make_engine([position(0)] * 1200, clock, max_batch=500)

    await engine.run()

    assert [len(b) for b in publisher.batches] == [500, 500, 200]


async def test_nothing_is_held_back_while_waiting_for_the_next_event():
    """O evento que já venceu é publicado ANTES de esperar o próximo (não fica retido no lote)."""
    clock = FakeClock()
    start = clock.now
    engine, publisher, _ = make_engine([position(0), position(30_000)], clock)

    await engine.run()

    assert offsets(publisher, start)[0] == EXACT(0, abs=1e-6)  # saiu na hora, não aos 30 s


# ---------------------------------------------------------------- pausa e retomada


async def test_pause_and_resume_keep_the_position_without_restarting_or_skipping():
    clock = FakeClock()
    start = clock.now
    events = [position(t * 1000) for t in range(11)]  # 0..10 s
    engine, publisher, _ = make_engine(events, clock)
    clock.at(start + 3.5, engine.pause)  # pausa no meio do intervalo entre o evento 3 e o 4

    task = asyncio.create_task(engine.run())
    await settle(lambda: engine.state is ReplayState.PAUSED)

    assert [e.t for e in publisher.events] == [0, 1000, 2000, 3000]  # só até a pausa
    assert engine.status().race_time_ms == EXACT(events[0].t + 3500, abs=1)

    clock.now += 100  # 100 s de relógio passam enquanto está pausado...
    await settle(lambda: True)
    assert [e.t for e in publisher.events] == [0, 1000, 2000, 3000]  # ...e NADA é publicado
    engine.resume()
    await task

    assert [e.t for e in publisher.events] == [t * 1000 for t in range(11)]  # todos, 1 vez cada
    # O evento de 4 s tem de sair 0,5 s depois de retomar (faltavam 0,5 s de corrida), nem antes
    # (rajada de "recuperação") nem depois (reinício).
    resumed_at = start + 103.5
    assert [when for when, e in publisher.published if e.t >= 4000] == EXACT(
        [resumed_at + (t - 3.5) for t in range(4, 11)], abs=1e-6
    )
    assert engine.state is ReplayState.FINISHED


async def test_pause_is_effective_even_in_the_middle_of_a_long_wait():
    clock = FakeClock()
    start = clock.now
    engine, publisher, _ = make_engine([position(0), position(60_000)], clock)
    clock.at(start + 1, engine.pause)

    task = asyncio.create_task(engine.run())
    await settle(lambda: engine.state is ReplayState.PAUSED)

    assert len(publisher.events) == 1  # o evento de 60 s NÃO foi publicado "no vácuo"
    engine.resume()
    await task
    assert len(publisher.events) == 2


async def test_pause_requested_before_run_starts_is_honoured():
    clock = FakeClock()
    engine, publisher, _ = make_engine([position(0), position(1000)], clock)
    engine.pause()  # ex.: o POST /pause chegou logo depois do /start, antes do motor iniciar

    task = asyncio.create_task(engine.run())
    await settle(lambda: engine.state is ReplayState.PAUSED)
    assert publisher.events == []

    engine.resume()
    await task
    assert len(publisher.events) == 2


async def test_speed_changed_while_paused_applies_on_resume():
    clock = FakeClock()
    start = clock.now
    engine, publisher, _ = make_engine([position(0), position(16_000)], clock)
    clock.at(start + 1, engine.pause)
    task = asyncio.create_task(engine.run())
    await settle(lambda: engine.state is ReplayState.PAUSED)

    engine.set_speed(16.0)
    assert engine.state is ReplayState.PAUSED  # trocar a velocidade não retoma
    engine.resume()
    await task

    assert offsets(publisher, start)[1] == EXACT(1 + 15 / 16, abs=1e-6)  # 15 s de corrida a 16x


# ---------------------------------------------------------------- parada e falhas


async def test_stop_ends_the_replay_and_closes_the_source():
    clock = FakeClock()
    start = clock.now
    engine, publisher, source = make_engine([position(0), position(60_000)], clock)
    clock.at(start + 2, engine.stop)

    await engine.run()

    assert engine.state is ReplayState.STOPPED
    assert len(publisher.events) == 1  # o de 60 s nunca saiu
    assert source.closed


async def test_stop_works_while_paused():
    clock = FakeClock()
    start = clock.now
    engine, _, source = make_engine([position(0), position(5000)], clock)
    clock.at(start + 1, engine.pause)
    task = asyncio.create_task(engine.run())
    await settle(lambda: engine.state is ReplayState.PAUSED)

    engine.stop()
    await task

    assert engine.state is ReplayState.STOPPED and source.closed


async def test_cancellation_marks_the_replay_stopped_and_propagates():
    clock = FakeClock()
    start = clock.now
    engine, _, source = make_engine([position(0), position(5000)], clock)
    clock.at(start + 1, engine.pause)
    task = asyncio.create_task(engine.run())
    await settle(lambda: engine.state is ReplayState.PAUSED)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.state is ReplayState.STOPPED and source.closed


async def test_a_publish_failure_becomes_a_failed_state_not_an_exception():
    clock = FakeClock()
    publisher = FakePublisher(clock, fail_on_batch=2)
    engine, _, source = make_engine(
        [position(0), position(1000), position(2000)], clock, publisher=publisher
    )

    await engine.run()  # não levanta: o supervisor decide o que fazer olhando o estado

    status = engine.status()
    assert status.state is ReplayState.FAILED
    assert "redis caiu" in (status.error or "")
    assert status.events_published == 1  # só o 1º lote chegou a ser publicado
    assert source.closed


async def test_abort_marks_the_replay_failed_with_the_reason():
    clock = FakeClock()
    start = clock.now
    engine, _, _ = make_engine([position(0), position(60_000)], clock)
    clock.at(start + 1, lambda: engine.abort("lock perdido"))

    await engine.run()

    assert engine.state is ReplayState.FAILED
    assert engine.status().error == "lock perdido"


# ---------------------------------------------------------------- status e validação


async def test_status_tracks_progress_lap_and_counts():
    clock = FakeClock()
    events = [
        RaceStatusEvent("started", 1, 0, 3),
        position(10),
        LapCompletedEvent("VER", 1, 20),
        RaceStatusEvent("running", 2, 20, 3),
        position(30),
    ]
    engine, _, _ = make_engine(events, clock)
    assert engine.status().state is ReplayState.PENDING

    await engine.run()

    status = engine.status()
    assert (status.state, status.lap, status.total_laps) == (ReplayState.FINISHED, 2, 3)
    assert status.events_published == 5 and status.error is None


@pytest.mark.parametrize("speed", [0, -4, float("inf"), float("nan")])
def test_engine_rejects_invalid_speeds(speed):
    clock = FakeClock()
    with pytest.raises(ValueError):
        make_engine([position(0)], clock, speed=speed)

    engine, _, _ = make_engine([position(0)], clock)
    with pytest.raises(ValueError):
        engine.set_speed(speed)
