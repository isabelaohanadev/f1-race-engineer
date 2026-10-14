"""Testes da perda no pit (strategy/pitloss.py) com corridas sintéticas de perda plantada.

Com inclinação e combustível zerados, a perda de uma parada é EXATAMENTE `in_extra + out_extra`.
"""

from dataclasses import replace

import numpy as np
import pytest

from strategy.laps import LapRow
from strategy.pitloss import (
    MIN_REFERENCE_LAPS,
    MIN_STOPS,
    SD_FLOOR,
    PitLossPrior,
    estimate_pit_loss,
    find_pit_stops,
)
from tests.strategy.synthetic import Plan, World, build_race

FLAT = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
NO_OFFSET = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}


def flat_world(**kwargs) -> World:
    """Sem inclinação, sem combustível, sem ruído: a perda medida é a plantada, ao milésimo."""
    base = {
        "slopes": FLAT,
        "offsets": NO_OFFSET,
        "kappa": 0.0,
        "noise": 0.0,
        "total_laps": 50,
        "in_extra": 12.0,
        "out_extra": 9.0,
    }
    return World(**(base | kwargs))


def race_with_stops(laps: list[int], **world) -> list[LapRow]:
    plans = [
        Plan(f"D{i:02d}", base=90 + 0.1 * i, stops=((lap, "HARD"),)) for i, lap in enumerate(laps)
    ]
    return build_race(plans, flat_world(**world))


# ------------------------------------------------------------------ medir a parada


def test_the_planted_loss_is_recovered_exactly():
    stops = find_pit_stops(race_with_stops([20]), kappa=0.0, upto_lap=50)

    assert len(stops) == 1
    assert stops[0].loss == pytest.approx(21.0, abs=0.01)
    assert (stops[0].in_lap, stops[0].out_lap, stops[0].green) == (20, 21, True)


def test_the_loss_is_the_sum_of_the_in_lap_and_the_out_lap_delta():
    stops = find_pit_stops(
        race_with_stops([20], in_extra=15.0, out_extra=4.0), kappa=0.0, upto_lap=50
    )

    assert stops[0].loss == pytest.approx(19.0, abs=0.01)


def test_the_loss_does_not_depend_on_the_fuel_coefficient_thanks_to_the_symmetric_window():
    """Com combustível (k = 0,05) o ritmo muda de volta em volta, mas o ritmo normal é a média da
    mediana ANTES e da mediana DEPOIS: uma tendência linear se cancela, centrada na parada. Então a
    perda sai a mesma com o k certo, sem k ou com um k errado (propriedade, não coincidência)."""
    race = race_with_stops([20], kappa=0.05)

    losses = [find_pit_stops(race, kappa=k, upto_lap=50)[0].loss for k in (0.05, 0.0, 0.2)]

    assert losses == [pytest.approx(21.0, abs=0.01)] * 3


def test_a_stop_needs_clean_reference_laps_on_both_sides():
    """A janela de antes é de 3 voltas e a volta 1 é descartada: com a entrada na volta 3 só a volta
    2 serve (1 < 2); na volta 4 servem a 2 e a 3."""
    assert find_pit_stops(race_with_stops([3]), kappa=0.0, upto_lap=50) == []
    assert len(find_pit_stops(race_with_stops([4]), kappa=0.0, upto_lap=50)) == 1


def test_dirty_laps_do_not_serve_as_reference():
    """Amarela nas voltas de referência: elas deixam de valer, e sem 2 de cada lado a parada
    some."""
    yellow = {
        n: "12" for n in (19, 18, 22, 23, 24)
    }  # antes: 17 limpa só; depois: nenhuma limpa em 22-24
    race = race_with_stops([20], status_laps=yellow)

    assert find_pit_stops(race, kappa=0.0, upto_lap=50) == []


# ------------------------------------------------------------------ bandeira verde x safety car


def test_a_stop_under_safety_car_is_not_green():
    race = race_with_stops([20, 30], status_laps={30: "4", 31: "4"})

    stops = {s.in_lap: s for s in find_pit_stops(race, kappa=0.0, upto_lap=50)}

    assert stops[20].green is True and stops[30].green is False


def test_stops_under_a_slow_field_are_kept_out_of_the_green_estimate():
    """Parada sob SC com uma volta de saída muito mais lenta: se entrasse, puxaria a mediana."""
    race = race_with_stops([15, 20, 25, 30, 35])
    # a parada de volta 30 acontece sob safety car e o piloto perde muito mais tempo
    race = [
        replace(r, lap_time_ms=r.lap_time_ms + 15_000, track_status="4")
        if r.driver == "D03" and r.lap_number in (30, 31)
        else r
        for r in race
    ]

    loss = estimate_pit_loss(race, upto_lap=50, kappa=0.0)

    assert loss.n_green == 4 and loss.n_other == 1
    assert loss.median == pytest.approx(21.0, abs=0.01)
    assert loss.other_median == pytest.approx(51.0, abs=0.01)  # 21 + 15 + 15, só informativo


# ------------------------------------------------------------------ sem o futuro


def test_a_stop_only_counts_once_the_pace_after_it_has_been_observed():
    """Entrada na 20, saída na 21; o ritmo depois da saída são as voltas 22, 23, 24. Precisa de
    2."""
    race = race_with_stops([20])

    assert find_pit_stops(race, kappa=0.0, upto_lap=21) == []  # a saída acabou de acontecer
    assert find_pit_stops(race, kappa=0.0, upto_lap=22) == []  # 1 volta depois: não basta
    assert len(find_pit_stops(race, kappa=0.0, upto_lap=22 + MIN_REFERENCE_LAPS - 1)) == 1


def test_the_estimate_at_lap_n_ignores_everything_after_n():
    race = race_with_stops([12, 16, 20, 24, 28, 32])
    n = 30
    corrupted = [
        replace(r, lap_time_ms=r.lap_time_ms + 60_000, pit_in=not r.pit_in)
        if r.lap_number > n
        else r
        for r in race
    ]
    prior = PitLossPrior(22.0, 2.0, 1.4)

    honest = estimate_pit_loss(race, upto_lap=n, kappa=0.0, prior=prior)

    assert estimate_pit_loss(corrupted, upto_lap=n, kappa=0.0, prior=prior) == honest
    assert estimate_pit_loss(race, upto_lap=50, kappa=0.0, prior=prior) != honest


# ------------------------------------------------------------------ a estimativa


def test_with_enough_stops_and_no_prior_the_race_speaks_alone():
    loss = estimate_pit_loss(race_with_stops([12, 16, 20, 24, 28, 32]), upto_lap=50, kappa=0.0)

    assert (loss.source, loss.weight) == ("corrida", 1.0)
    assert loss.median == pytest.approx(21.0, abs=0.01) and loss.n_green == 6
    assert loss.sd == SD_FLOOR  # todas iguais: a dispersão tem piso, nunca zero


def test_with_no_stops_and_no_prior_there_is_no_estimate():
    assert estimate_pit_loss(race_with_stops([]), upto_lap=50, kappa=0.0) is None
    assert estimate_pit_loss(race_with_stops([20]), upto_lap=50, kappa=0.0) is None  # < MIN_STOPS


def test_with_too_few_stops_the_prior_is_the_estimate():
    prior = PitLossPrior(mean=23.5, between_sd=2.0, stop_sd=1.4)

    loss = estimate_pit_loss(race_with_stops([20]), upto_lap=50, kappa=0.0, prior=prior)

    assert MIN_STOPS == 2
    assert (loss.source, loss.weight, loss.median) == ("prior", 0.0, 23.5)
    assert loss.raw_median == pytest.approx(21.0, abs=0.01)  # a única parada, só informativa
    assert loss.sd == 1.4 and loss.se == 2.0


def test_the_prior_pulls_a_race_with_few_stops_and_lets_go_with_many():
    prior = PitLossPrior(mean=25.0, between_sd=2.0, stop_sd=1.4)
    few = estimate_pit_loss(race_with_stops([20, 30]), upto_lap=50, kappa=0.0, prior=prior)
    many = estimate_pit_loss(
        race_with_stops([8, 12, 16, 20, 24, 28, 32, 36, 40, 44]),
        upto_lap=50,
        kappa=0.0,
        prior=prior,
    )

    assert few.source == many.source == "encolhido"
    assert 21.0 < few.median < 25.0  # entre o dado (21) e o prior (25)
    assert many.weight > few.weight
    assert abs(many.median - 21.0) < abs(few.median - 21.0)


def test_the_median_resists_one_botched_stop():
    race = race_with_stops([12, 16, 20, 24, 28, 32])
    race = [
        replace(r, lap_time_ms=r.lap_time_ms + 30_000)
        if (r.driver, r.lap_number) == ("D02", 21)
        else r
        for r in race
    ]

    loss = estimate_pit_loss(race, upto_lap=50, kappa=0.0)

    assert loss.median == pytest.approx(21.0, abs=0.01)  # a mediana não vê a parada de 51 s
    assert loss.sd < 1.0  # nem a dispersão robusta


def test_the_dispersion_between_stops_is_the_planted_crew_variation():
    rng = np.random.default_rng(0)
    plans = [Plan(f"D{i:02d}", base=90 + 0.1 * i, stops=((8 + 2 * i, "HARD"),)) for i in range(18)]
    # cada equipe tem a sua velocidade: a saída soma um desvio N(0, 1,5) à volta de saída
    race = build_race(plans, flat_world(total_laps=50))
    jitter = {p.code: rng.normal(0, 1.5) for p in plans}
    race = [
        replace(r, lap_time_ms=r.lap_time_ms + round(1000 * jitter[r.driver])) if r.pit_out else r
        for r in race
    ]

    loss = estimate_pit_loss(race, upto_lap=50, kappa=0.0)

    assert loss.sd == pytest.approx(1.5, rel=0.4)
    assert loss.se == pytest.approx(1.2533 * loss.sd / np.sqrt(loss.n_green))
