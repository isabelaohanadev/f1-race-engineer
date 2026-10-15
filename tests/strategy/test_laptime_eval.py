"""O erro de previsão do tempo de volta (strategy/laptime_eval.py) com corridas sintéticas de ritmo
conhecido: sem ruído o modelo erra ~0 e bate a persistência; com paradas, SC e voltas sujas as
voltas
certas são excluídas."""

from dataclasses import replace

import numpy as np
import pytest

from strategy.fuel import physical_coefficient
from strategy.laps import LapRow
from strategy.laptime_eval import (
    LapTimeRecord,
    is_clean,
    lap_time_records,
    per_race,
    persistence,
)
from strategy.snapshot import crossing_ms
from tests.strategy.synthetic import Plan, World, build_race, chase_field

TOTAL = 60
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)


def records(rows, laps=(30,), horizons=(1, 2, 3), **kwargs):
    return lap_time_records(
        rows, race="X", total_laps=TOTAL, laps=laps, horizons=horizons, **kwargs
    )


@pytest.fixture(scope="module")
def clean_rows():
    return chase_field(gap=5.0, world=FLAT)


# ------------------------------------------------------------------ o que é uma volta limpa


def make(**over) -> LapRow:
    base = dict(
        driver="A", lap_number=10, lap_time_ms=90_000, compound="MEDIUM", tyre_life=10, stint=1,
        position=1, session_time_ms=900_000, pit_in=False, pit_out=False, track_status="1",
    )  # fmt: skip
    return LapRow(**(base | over))


@pytest.mark.parametrize(
    ("over", "clean"),
    [
        ({}, True),
        ({"lap_time_ms": None}, False),
        ({"track_status": "4"}, False),
        ({"track_status": "12"}, False),
        ({"pit_in": True}, False),
        ({"pit_out": True}, False),
        ({"lap_number": 1}, False),
        ({"compound": "INTERMEDIATE"}, False),
        ({"compound": None}, False),
    ],
)
def test_only_a_green_dry_lap_without_a_pit_visit_is_clean(over, clean):
    assert is_clean(make(**over)) is clean


def test_the_persistence_is_the_median_of_the_last_three_clean_laps_up_to_the_lap():
    rows = [
        make(lap_number=n, lap_time_ms=ms)
        for n, ms in [(5, 91_000), (6, 95_000), (7, 90_000), (8, 92_000)]
    ]
    rows.append(make(lap_number=9, lap_time_ms=200_000, track_status="4"))  # suja: fora

    assert persistence(rows, "A", 9) == pytest.approx(92.0)  # mediana de 95, 90 e 92... = 92
    assert persistence(rows, "A", 7) == pytest.approx(91.0)  # mediana de 91, 95, 90
    assert persistence(rows, "ZZ", 9) is None


# ------------------------------------------------------------------ o modelo contra a verdade
# plantada


def test_without_noise_the_model_predicts_the_next_laps_exactly_and_beats_persistence(clean_rows):
    """No mundo plano os tempos são exatos: viés e erro ~0, intervalo cobrindo tudo, e a
    persistência
    (que não conhece a degradação) erra por voltas e voltas."""
    out = records(clean_rows, horizons=(1, 2, 3, 4, 5))

    assert {r.horizon for r in out} == {1, 2, 3, 4, 5}
    for r in out:
        assert abs(r.bias) < 0.05 and r.mae < 0.05, r
        assert r.coverage == 1.0
        assert r.baseline_mae is not None and r.baseline_mae > 5 * max(r.mae, 0.01)
    # a persistência piora com o horizonte (a degradação acumula)
    baseline = {r.horizon: r.baseline_mae for r in out}
    assert baseline[5] > baseline[1]


def test_a_noisy_race_has_a_small_error_and_a_coverage_near_nominal():
    rows = chase_field(gap=5.0, world=replace(FLAT, noise=0.15, seed=3))

    out = records(rows, laps=(20, 30, 40), horizons=(1, 3, 5))

    assert out
    for r in out:
        assert abs(r.bias) < 0.4 and r.mae < 0.5
    covered = np.mean([r.coverage for r in out])
    assert 0.8 < covered < 0.99  # o nominal é 90%: sem o ruído de volta no desvio cairia a ~30%


def test_the_laps_with_a_stop_the_safety_car_or_in_another_stint_are_not_predicted():
    """R para na 33 (a 33 é de entrada, a 34 de saída e o stint muda na 35): a partir da 30 com
    horizonte 3, o alvo dele (33) sai. Com SC na 32, o alvo de horizonte 2 do líder (a 32) sai."""
    full = {r.horizon: r.n for r in records(chase_field(gap=5.0, world=FLAT), horizons=(1, 2, 3))}
    stop = {
        r.horizon: r.n
        for r in records(chase_field(r_stop=33, gap=5.0, world=FLAT), horizons=(1, 2, 3))
    }
    sc_rows = chase_field(gap=5.0, world=replace(FLAT, status_laps={32: "4"}))
    safety = {r.horizon: r.n for r in records(sc_rows, horizons=(1, 2, 3))}

    assert stop[1] == full[1] and stop[2] == full[2] and stop[3] == full[3] - 1
    assert safety[1] == full[1] and safety[2] == full[2] - 1  # o líder (volta 30) + 2 = a 32


def test_a_driver_without_a_level_in_his_current_stint_is_not_predicted():
    """Quem acabou de parar (stint novo, menos de 3 voltas limpas) não tem nível: fica de fora."""
    rows = chase_field(a_stop=27, gap=5.0, world=FLAT)
    base = records(chase_field(gap=5.0, world=FLAT), horizons=(1,))[0].n

    (with_stop,) = records(rows, horizons=(1,))

    assert with_stop.n == base - 1  # o A não entra


def test_the_trailing_cars_are_predicted_from_their_own_last_completed_lap():
    """No instante em que o líder completa a 30, o A está na 29 (5 s atrás): o horizonte 1 dele é a
    volta 30 dele, que está no futuro de t e serve de gabarito. O líder tem a 31."""
    rows = chase_field(gap=5.0, world=FLAT)

    instant = {r.horizon: r.n for r in records(rows, horizons=(1,))}
    by_laps = {r.horizon: r.n for r in records(rows, horizons=(1,), snapshot="volta")}

    assert instant[1] == by_laps[1]  # os mesmos pilotos entram, cada um na sua próxima volta


def test_the_model_prediction_does_not_depend_on_the_laps_after_the_instant():
    """Os tempos futuros são o GABARITO, nunca insumo: trocar todos por 1.000 s (limpos) muda o erro
    medido exatamente pela diferença do observado, e o previsto continua o mesmo."""
    plans = [Plan("R", base=88.0), Plan("A", base=88.0, grid_gap=5.0)]
    rows = build_race(plans, FLAT)
    t = crossing_ms(rows, 30)
    spoiled = [replace(r, lap_time_ms=1_000_000) if r.session_time_ms > t else r for r in rows]

    (clean,) = records(rows, horizons=(1,))
    (dirty,) = records(spoiled, horizons=(1,))

    # cada piloto é previsto na SUA próxima volta: R (na 30) na 31, A (na 29) na 30
    observed = np.mean(
        [
            next(x for x in rows if (x.driver, x.lap_number) == key).lap_time_ms / 1000
            for key in (("R", 31), ("A", 30))
        ]
    )
    assert clean.n == dirty.n == 2
    assert clean.bias + observed == pytest.approx(dirty.bias + 1000.0, abs=1e-3)  # o mesmo previsto


# ------------------------------------------------------------------ da volta à corrida


def record(race, lap, horizon, n, mae, baseline=None, baseline_n=None) -> LapTimeRecord:
    return LapTimeRecord(
        race, lap, horizon, n, 0.0, mae, 1.0, 0.0 if baseline is not None else None, baseline,
        n if baseline_n is None else baseline_n,
    )  # fmt: skip


def test_the_race_mean_weights_each_lap_by_the_drivers_it_had_and_races_stand_alone():
    recs = [
        record("A", 10, 1, 10, 1.0),
        record("A", 20, 1, 30, 2.0),  # (10 x 1 + 30 x 2) / 40 = 1,75
        record("B", 10, 1, 5, 4.0),
        record("A", 10, 2, 10, 9.0),  # outro horizonte: fora
    ]

    assert per_race(recs, 1, "mae") == {"A": pytest.approx(1.75), "B": pytest.approx(4.0)}
    assert per_race(recs, 3, "mae") == {}


def test_the_baseline_is_weighted_by_its_own_count_and_skipped_when_absent():
    recs = [
        record("A", 10, 1, 10, 1.0, baseline=3.0, baseline_n=10),
        record("A", 20, 1, 10, 1.0, baseline=5.0, baseline_n=30),
        record("B", 10, 1, 10, 1.0),  # sem persistência
    ]

    assert per_race(recs, 1, "baseline_mae") == {"A": pytest.approx(4.5)}
