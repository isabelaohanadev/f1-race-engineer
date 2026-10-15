"""O estado da corrida até o instante t (strategy/snapshot.py): o que um sistema ao vivo saberia."""

from dataclasses import replace

import pytest

from strategy.fuel import physical_coefficient
from strategy.snapshot import (
    car_states,
    crossing_ms,
    driver_crossing_ms,
    rows_until,
    state_at_driver_lap,
    state_at_leader_lap,
    state_until,
)
from tests.strategy.synthetic import Plan, World, build_race, chase_field, clock_at

TOTAL = 60
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)
N = 30


@pytest.fixture(scope="module")
def rows():
    return chase_field(gap=5.0, world=FLAT)  # R lidera, A a 5 s, os lentos longe atrás


def test_the_leader_crossing_is_the_smallest_clock_on_the_lap(rows):
    assert crossing_ms(rows, N) == round(clock_at(rows, "R", N) * 1000)
    assert crossing_ms(rows, 999) is None


def test_a_drivers_crossing_is_his_own_clock_on_the_lap(rows):
    assert driver_crossing_ms(rows, "A", N) == round(clock_at(rows, "A", N) * 1000)
    assert driver_crossing_ms(rows, "A", 999) is None and driver_crossing_ms(rows, "ZZ", N) is None


def test_only_laps_completed_by_the_instant_are_visible_and_the_instant_is_inclusive(rows):
    t = crossing_ms(rows, N)

    visible = rows_until(rows, t)

    assert all(r.session_time_ms <= t for r in visible)
    assert any(r.driver == "R" and r.lap_number == N for r in visible)  # inclusive
    assert not any(r.driver == "A" and r.lap_number == N for r in visible)  # A cruza 5 s depois
    assert rows_until(rows, t - 1) != visible


def test_laps_without_a_clock_are_never_visible(rows):
    without = [replace(r, session_time_ms=None) if r.lap_number == 10 else r for r in rows]

    assert not any(r.lap_number == 10 for r in rows_until(without, 10**9))


def test_at_the_instant_the_leader_completes_lap_n_everybody_else_is_on_an_earlier_lap(rows):
    """O ponto do módulo: em t o líder completou a N, e todos os outros, ainda não."""
    state = state_at_leader_lap(rows, N)

    assert state.leader_lap == N
    assert state.cars["R"].laps_done == N
    assert state.cars["A"].laps_done == N - 1
    assert state.cars["A"].clock_s == pytest.approx(clock_at(rows, "A", N - 1))
    assert state.laps_down["A"] == 1 and state.laps_down["R"] == 0
    assert all(c.laps_done < N for d, c in state.cars.items() if d != "R")


def test_the_old_lap_count_snapshot_gives_everyone_the_lap_n_row(rows):
    """O desenho antigo: A na volta N dele, que só é completada DEPOIS do instante do líder."""
    legacy = car_states(rows, N)
    state = state_at_leader_lap(rows, N)

    assert legacy["A"].laps_done == N
    assert legacy["A"].clock_s > state.cars["R"].clock_s  # depois do instante t
    assert legacy["A"].clock_s == pytest.approx(clock_at(rows, "A", N))
    assert state.cars["A"].clock_s < legacy["A"].clock_s


def test_nothing_after_the_instant_changes_the_state(rows):
    """Estragar tudo o que acontece DEPOIS de t (tempo, composto, parada, status: o relógio, que é
    futuro, fica) não muda o estado nem uma casa."""
    t = crossing_ms(rows, N)
    spoiled = [
        replace(
            r,
            lap_time_ms=1_000,
            compound="SOFT",
            tyre_life=99,
            stint=9,
            track_status="4",
            pit_in=True,
            pit_out=True,
        )
        if r.session_time_ms > t
        else r
        for r in rows
    ]

    assert state_until(spoiled, t) == state_until(rows, t)
    assert any(r != s for r, s in zip(rows, spoiled, strict=True))  # a premissa: estragou algo


def test_a_lapped_car_is_still_in_the_race_one_lap_down(rows):
    """Um carro 130 s/volta (40 s mais lento) é lapado, mas cruza a linha uma vez por volta dele."""
    plans = [Plan("R", base=90.0), Plan("L", base=130.0, grid_gap=20.0)]
    two = build_race(plans, replace(FLAT, slopes={"SOFT": 0, "MEDIUM": 0, "HARD": 0}))

    state = state_at_leader_lap(two, 40)

    assert "L" in state.cars and state.laps_down["L"] >= 10  # com 40 s por volta, 13 voltas atrás
    assert state.cars["L"].laps_done < 40


def test_a_car_that_stopped_crossing_the_line_has_left_the_race(rows):
    gone = [r for r in rows if not (r.driver == "D05" and r.lap_number > 20)]

    state = state_at_leader_lap(gone, 40)

    assert "D05" not in state.cars  # parou há 20 voltas, 1.800 s
    assert "D04" in state.cars


def test_the_silence_limit_is_in_the_typical_laps_of_the_driver(rows):
    """Um carro que não cruza há 1,5 volta típica ainda está na corrida (parada longa, SC); há 3,
    não."""
    t = crossing_ms(rows, 40)
    lap = 90.0  # a volta típica de R e A

    def last_lap_of_a(silence_laps: float):
        cutoff = t - round(silence_laps * lap * 1000)
        trimmed = [r for r in rows if not (r.driver == "A" and r.session_time_ms > cutoff)]
        return state_until(trimmed, t)

    assert "A" in last_lap_of_a(1.5).cars
    assert "A" not in last_lap_of_a(3.0).cars
    assert "A" in state_until(rows, t, max_silence_laps=10).cars


def test_a_lap_never_completed_has_no_state(rows):
    assert state_at_leader_lap(rows, 999) is None
    assert state_at_driver_lap(rows, "A", 999) is None
    assert state_at_driver_lap(rows, "ZZ", N) is None


def test_the_driver_instant_is_when_that_driver_completes_the_lap(rows):
    state = state_at_driver_lap(rows, "A", N)

    assert state.t_ms == driver_crossing_ms(rows, "A", N)
    assert state.cars["A"].laps_done == N and state.cars["R"].laps_done == N  # R cruzou antes
    assert state.cars["D00"].laps_done < N  # os lentos ainda não


def test_the_state_carries_what_the_last_lap_says_about_the_car(rows):
    state = state_at_leader_lap(rows, 23)  # a volta 23: D11 (grid 11) está na de saída dos boxes

    car = state.cars["R"]
    assert (car.compound, car.age, car.stint) == ("MEDIUM", 23, 1)
    assert car.status == "1" and not car.pit_in and not car.pit_out
    pits = [c for c in state.cars.values() if c.pit_in or c.pit_out]
    assert pits, "algum carro estava nos boxes ou saindo deles nessa volta"
