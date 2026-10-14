"""A Timeline é matemática pura: testada com números, sem asyncio e sem relógio."""

import math

import pytest

from replay.timeline import Timeline


def test_event_delay_is_race_time_difference_divided_by_speed():
    timeline = Timeline(origin_ms=10_000, now=500.0, speed=4.0)

    assert timeline.delay_s(10_000, now=500.0) == pytest.approx(0)
    assert timeline.delay_s(14_000, now=500.0) == pytest.approx(1.0)  # 4 s de corrida a 4x = 1 s
    assert timeline.delay_s(14_000, now=500.5) == pytest.approx(0.5)  # já passou meio segundo


def test_a_late_event_has_negative_delay_and_does_not_shift_the_next_one():
    """Evento atrasado não "empurra" os seguintes: o horário do próximo continua sendo calculado a
    partir da âncora. É a diferença entre isto e somar sleeps."""
    timeline = Timeline(origin_ms=0, now=0.0, speed=1.0)

    assert timeline.delay_s(1000, now=1.3) == pytest.approx(-0.3)  # chegou 300 ms atrasado...
    assert timeline.delay_s(2000, now=1.3) == pytest.approx(0.7)  # ...e o seguinte segue pontual


def test_changing_speed_reanchors_so_the_race_time_never_jumps():
    timeline = Timeline(origin_ms=0, now=0.0, speed=1.0)
    assert timeline.race_ms(now=10.0) == pytest.approx(10_000)

    timeline.set_speed(16.0, now=10.0)

    assert timeline.race_ms(now=10.0) == pytest.approx(10_000)  # continua onde estava
    assert timeline.race_ms(now=11.0) == pytest.approx(10_000 + 16_000)  # e agora anda 16x


def test_many_speed_changes_accumulate_no_drift():
    """Ida e volta entre velocidades 1.000 vezes: o tempo de corrida é a SOMA exata de
    (duração na velocidade x velocidade), sem erro acumulado em nenhuma troca."""
    timeline = Timeline(origin_ms=0, now=0.0, speed=1.0)
    now, expected_ms = 0.0, 0.0
    for i in range(1000):
        speed = (1.0, 4.0, 16.0)[i % 3]
        timeline.set_speed(speed, now)
        now += 0.25
        expected_ms += 0.25 * 1000 * speed

    assert timeline.race_ms(now) == pytest.approx(expected_ms, rel=1e-9)


def test_pause_freezes_race_time_and_resume_continues_from_the_same_point():
    timeline = Timeline(origin_ms=0, now=0.0, speed=1.0)

    timeline.pause(now=3.0)
    assert timeline.race_ms(now=3.0) == pytest.approx(3000)
    assert timeline.race_ms(now=500.0) == pytest.approx(3000)  # 497 s depois: continua parado
    assert timeline.delay_s(5000, now=500.0) is None  # pausado: nenhum evento vence

    timeline.resume(now=500.0)
    assert timeline.race_ms(now=500.0) == pytest.approx(3000)  # retomou do MESMO ponto
    assert timeline.delay_s(5000, now=500.0) == pytest.approx(2.0)


def test_pause_and_resume_are_idempotent():
    timeline = Timeline(origin_ms=0, now=0.0)

    timeline.pause(now=2.0)
    timeline.pause(now=9.0)  # a 2ª pausa não pode mover o ponto congelado
    timeline.resume(now=10.0)
    timeline.resume(now=99.0)  # a 2ª retomada não pode mover a âncora

    assert timeline.race_ms(now=11.0) == pytest.approx(2000 + 1000)


def test_speed_can_be_changed_while_paused_and_applies_on_resume():
    timeline = Timeline(origin_ms=0, now=0.0, speed=1.0)
    timeline.pause(now=1.0)

    timeline.set_speed(16.0, now=50.0)
    assert timeline.paused and timeline.race_ms(now=50.0) == pytest.approx(1000)

    timeline.resume(now=60.0)
    assert timeline.race_ms(now=61.0) == pytest.approx(1000 + 16_000)


@pytest.mark.parametrize("speed", [0, -1, math.inf, math.nan])
def test_invalid_speeds_are_rejected(speed):
    with pytest.raises(ValueError):
        Timeline(origin_ms=0, now=0.0, speed=speed)

    timeline = Timeline(origin_ms=0, now=0.0)
    with pytest.raises(ValueError):
        timeline.set_speed(speed, now=1.0)
