"""Testes do nível de ritmo (strategy/pace.py) com corridas sintéticas de nível e diferença
conhecidos."""

from dataclasses import replace

import numpy as np
import pytest

from strategy.laps import LapRow
from strategy.pace import MIN_LEVEL_LAPS, MIN_PAIRS, _moderated_sigma, fit_pace
from tests.strategy.synthetic import OFFSETS, Plan, World, build_race

TOTAL = 60


def field(world: World | None = None, *, drivers: int = 20, spread: float = 0.15) -> list[LapRow]:
    """20 pilotos de ritmos diferentes: metade larga de MEDIUM, metade de SOFT, e todos trocam para
    HARD numa volta diferente. O nível plantado de cada stint é `base + deslocamento[composto]`."""
    plans = [
        Plan(
            f"D{i:02d}",
            base=90 + spread * i,
            compound="MEDIUM" if i % 2 else "SOFT",
            stops=((15 + i, "HARD"),),
        )
        for i in range(drivers)
    ]
    return build_race(plans, world or World(total_laps=TOTAL, noise=0.15, seed=1))


@pytest.fixture(scope="module")
def race():
    return field()


@pytest.fixture(scope="module")
def pace(race):
    return fit_pace(race, upto_lap=TOTAL, total_laps=TOTAL)


# ------------------------------------------------------------------ o nível plantado


def test_each_stint_level_is_the_planted_pace_at_age_zero(pace):
    """Corrigido do combustível e na idade zero, o tempo do stint é `base + deslocamento`."""
    for (driver, _), level in pace.levels.items():
        base = 90 + 0.15 * int(driver[1:])
        assert level.level0 == pytest.approx(base + OFFSETS[level.compound], abs=0.25), (
            driver,
            level.compound,
        )


def test_the_level_follows_the_driver_not_the_compound_average(pace):
    """O piloto mais lento da corrida (D19) tem um nível 2,85 s acima do mais rápido (D00): o nível
    é DELE, não a média do composto, que misturaria os dois."""
    slow = pace.current("D19").level0
    fast = pace.current("D00").level0

    assert slow - fast == pytest.approx(0.15 * 19 + (0.8 - 0.8), abs=0.3)


def test_the_current_level_is_the_latest_stint_with_a_level(pace):
    level = pace.current("D05")

    assert level.stint == 2 and level.compound == "HARD" and level.last_lap == TOTAL
    assert pace.current("NINGUEM") is None


def test_a_level_predicts_the_next_laps_of_the_same_stint():
    """Sem ruído, o nível e a inclinação estimados reproduzem as voltas seguintes do stint."""
    race = field(World(total_laps=TOTAL, noise=0.0, seed=2))
    cut = 40
    pace = fit_pace(race, upto_lap=cut, total_laps=TOTAL)
    level = pace.current("D03")  # no stint 2 (HARD) desde a volta 18
    future = [r for r in race if r.driver == "D03" and cut < r.lap_number <= cut + 5]

    for lap in future:
        predicted = level.predict(lap.tyre_life, lap.lap_number, pace.kappa)
        assert predicted == pytest.approx(lap.lap_time_ms / 1000, abs=0.12)


# ------------------------------------------------------------------ a diferença entre compostos


def test_the_offset_is_measured_inside_each_driver(pace):
    """HARD é 0,4 s mais lento que MEDIUM no nível: medido nos pilotos que rodaram os dois."""
    offset = pace.offset("MEDIUM", "HARD")

    assert offset.source == "pares" and offset.n_pairs >= MIN_PAIRS
    assert offset.delta == pytest.approx(OFFSETS["HARD"] - OFFSETS["MEDIUM"], abs=0.15)
    assert pace.offset("HARD", "MEDIUM").delta == pytest.approx(-offset.delta, abs=0.15)


def test_the_offset_inside_the_driver_cancels_the_drivers_own_pace_where_the_means_do_not():
    """A razão de ser do desenho. Aqui os pilotos que largam de SOFT são 2 s MAIS RÁPIDOS que os de
    MEDIUM: a diferença entre as MÉDIAS dos compostos sairia errada em ~2 s; dentro de cada piloto
    o ritmo dele cancela e a diferença sai certa."""
    plans = [
        Plan(
            f"D{i:02d}",
            base=(88.0 if i % 2 == 0 else 90.0) + 0.02 * i,  # SOFT (par) 2 s mais rápido
            compound="SOFT" if i % 2 == 0 else "MEDIUM",
            stops=((15 + i, "HARD"),),
        )
        for i in range(20)
    ]
    pace = fit_pace(
        build_race(plans, World(total_laps=TOTAL, noise=0.1, seed=3)),
        upto_lap=TOTAL,
        total_laps=TOTAL,
    )

    within = pace.offset("SOFT", "HARD")
    means = pace.compound_levels["HARD"].mean0 - pace.compound_levels["SOFT"].mean0

    truth = OFFSETS["HARD"] - OFFSETS["SOFT"]
    assert within.source == "pares" and abs(within.delta - truth) < 0.2
    assert (
        abs(means - truth) > 0.8
    )  # a diferença entre as médias mistura quem é rápido com quem não


def test_with_too_few_drivers_on_both_compounds_it_falls_back_to_the_compound_means(pace):
    """Ninguém rodou SOFT e MEDIUM juntos (todos vão de um deles para o HARD): sem pares."""
    offset = pace.offset("SOFT", "MEDIUM")

    assert offset.source == "médias" and offset.n_pairs == 0
    assert offset.se > pace.offset("MEDIUM", "HARD").se  # e com mais incerteza


def test_an_offset_to_the_same_compound_is_zero_and_to_an_unknown_one_does_not_exist(pace):
    assert pace.offset("HARD", "HARD").delta == 0.0
    assert pace.offset("HARD", "INTERMEDIATE") is None


# ------------------------------------------------------------------ a evolução da pista cancela


def test_track_evolution_cancels_in_the_difference_between_two_cars():
    """Duas corridas iguais, mas na segunda a pista evolui bem mais rápido (k 0,12 em vez de 0,05).
    O tempo previsto de CADA carro muda muito; a DIFERENÇA entre os dois quase não muda."""

    def gap(kappa: float) -> tuple[float, float]:
        race = field(World(total_laps=TOTAL, noise=0.0, kappa=kappa, seed=4))
        model = fit_pace(race, upto_lap=40, total_laps=TOTAL)
        a, r = model.current("D04"), model.current("D03")
        times = [lvl.predict(lvl.mean_age + 6, 46, model.kappa) for lvl in (a, r)]
        return times[0] - times[1], times[0]

    slow_track, absolute_slow = gap(0.05)
    fast_track, absolute_fast = gap(0.12)

    assert abs(absolute_fast - absolute_slow) > 0.2  # o tempo absoluto previsto mudou
    assert slow_track == pytest.approx(fast_track, abs=0.08)  # a diferença, não


# ------------------------------------------------------------------ encolhimento do nível


def test_a_noisy_short_stint_is_pulled_toward_the_compound_mean_and_a_clean_one_is_not():
    plans = [
        Plan(f"D{i:02d}", base=90 + 0.05 * i, compound="MEDIUM", stops=((14 + i, "HARD"),))
        for i in range(18)
    ]
    # o D99 largou de MEDIUM com 5 s a mais de nível e MUITO ruído, e só tem 4 voltas limpas
    plans.append(Plan("D99", base=95.0, compound="MEDIUM", age=1, pace_noise=2.0))
    race = [r for r in build_race(plans, World(total_laps=TOTAL, noise=0.1, seed=5))]
    race = [r for r in race if r.driver != "D99" or r.lap_number <= 5]  # corrida dele acaba na 5

    pace = fit_pace(race, upto_lap=TOTAL, total_laps=TOTAL)
    short = pace.levels[("D99", 1)]
    clean = pace.levels[("D05", 1)]
    mean = pace.compound_levels["MEDIUM"].mean0

    assert short.n_laps == 4 and short.weight < 0.9 and clean.weight > 0.95
    assert abs(short.level0 - mean) < abs(short.raw_level0 - mean)  # foi puxado para a média


def test_without_enough_stints_the_level_variation_is_not_estimated_and_nothing_is_shrunk():
    plans = [Plan(f"D{i}", base=90 + i, stops=((20, "HARD"),)) for i in range(2)]

    pace = fit_pace(build_race(plans, World(total_laps=40, seed=6)), upto_lap=40, total_laps=40)

    assert not np.isfinite(pace.tau_level)
    assert all(lvl.weight == 1.0 and lvl.level0 == lvl.raw_level0 for lvl in pace.levels.values())


def test_a_stint_with_too_few_clean_laps_has_no_level():
    race = field()
    cut = 15 + 5 + 1 + MIN_LEVEL_LAPS - 1  # D05 saiu na 21: a volta de saída é descartada

    pace = fit_pace(race, upto_lap=cut, total_laps=TOTAL)

    assert ("D05", 2) not in pace.levels  # o stint novo ainda não tem voltas limpas suficientes
    assert pace.current("D05").stint == 1  # o nível "atual" é o do stint anterior: quem usa confere


# ------------------------------------------------------------------ incerteza


def test_the_uncertainty_of_a_sum_is_the_regression_prediction_formula(pace):
    """`(n * se_ref)^2 + (soma(a - idade_media))^2 * sd_inclinacao^2`: o nível no ritmo atual (idade
    média) mais a inclinação, que pesa mais longe da idade média."""
    level = pace.current("D06")
    ages = [level.mean_age + 6 + i for i in range(5)]

    expected = np.hypot(
        5 * level.se_ref, sum(a - level.mean_age for a in ages) * level.slope.predictive_sd
    )

    assert level.sd_of_sum(ages) == pytest.approx(expected)


def test_at_the_mean_age_only_the_level_matters_and_away_from_it_the_slope_does(pace):
    level = pace.current("D06")
    centred = [level.mean_age] * 5
    far = [level.mean_age + 12 + i for i in range(5)]

    assert level.sd_of_sum(centred, slope_sd=9.9) == pytest.approx(
        5 * level.se_ref
    )  # a inclinação some
    assert level.sd_of_sum(far, slope_sd=0.05) > 5 * level.sd_of_sum(far, slope_sd=0.0)


def test_a_larger_slope_uncertainty_gives_a_larger_sum_uncertainty(pace):
    level = pace.current("D06")
    ages = [level.mean_age + 8 + i for i in range(6)]

    assert (
        level.sd_of_sum(ages, slope_sd=0.05)
        > level.sd_of_sum(ages)
        > level.sd_of_sum(ages, slope_sd=0.0)
    )


# ------------------------------------------------------------------ sem o futuro


def test_the_pace_model_at_lap_n_ignores_everything_after_n(race):
    n = 38
    corrupted = [
        replace(r, lap_time_ms=r.lap_time_ms + 60_000, compound="SOFT", tyre_life=999)
        if r.lap_number > n
        else r
        for r in race
    ]

    assert fit_pace(corrupted, upto_lap=n, total_laps=TOTAL) == fit_pace(
        race, upto_lap=n, total_laps=TOTAL
    )


def test_a_wet_race_has_no_pace_model():
    wet = [replace(r, compound="INTERMEDIATE") for r in field()]

    pace = fit_pace(wet, upto_lap=TOTAL, total_laps=TOTAL)

    assert pace.levels == {} and pace.compound_levels == {} and pace.offsets == {}


# ------------------------------------------------------------------ o ruído de um stint curto


def test_the_noise_of_a_short_stint_is_pulled_to_its_own_scatter_but_not_all_the_way():
    noisy = np.array([-2.0, 1.5, 2.2, -1.7])  # 4 voltas com muita dispersão

    moderated = _moderated_sigma(noisy, pooled=0.1)

    assert 0.1 < moderated < float(np.std(noisy, ddof=1))  # entre a global e a do próprio stint


def test_a_long_stint_speaks_for_itself_and_a_stint_without_dof_uses_the_pooled_noise():
    rng = np.random.default_rng(0)
    long = rng.normal(0, 0.5, 60)

    assert _moderated_sigma(long, pooled=0.1) == pytest.approx(
        float(np.std(long, ddof=1)), rel=0.15
    )
    assert _moderated_sigma(np.array([1.0]), pooled=0.3) == 0.3


# ------------------------------------------------------------------ robustez do nível


def test_the_level_resists_a_few_bad_laps_in_the_stint():
    """Tráfego e erros acontecem: 3 voltas de um stint de ~40 levam +6 s. O nível é a MEDIANA dos
    tempos corrigidos, então quase não se mexe (a média subiria ~0,45 s)."""
    clean = field(World(total_laps=TOTAL, noise=0.05, seed=7))
    spoiled = [
        replace(r, lap_time_ms=r.lap_time_ms + 6000)
        if r.driver == "D02" and r.lap_number in (25, 31, 38)
        else r
        for r in clean
    ]

    honest = fit_pace(clean, upto_lap=TOTAL, total_laps=TOTAL).current("D02")
    hurt = fit_pace(spoiled, upto_lap=TOTAL, total_laps=TOTAL).current("D02")

    assert hurt.raw_level0 == pytest.approx(honest.raw_level0, abs=0.12)


# ------------------------------------------------------------------ o viés por corrida da diferença


def test_the_between_race_spread_widens_the_offset_uncertainty_beyond_the_standard_error():
    """O erro-padrão de uma diferença entre compostos é 0,05 a 0,17 s e as corridas se espalham ~0,5
    s: a incerteza que o undercut usa inclui esse espalhamento (de `Priors.offset_sd`)."""
    from dataclasses import replace as dc_replace

    from strategy.priors import DEFAULT_PRIORS

    plain = fit_pace(
        field(), upto_lap=TOTAL, total_laps=TOTAL, priors=dc_replace(DEFAULT_PRIORS, offset_sd=0.0)
    )
    wide = fit_pace(
        field(), upto_lap=TOTAL, total_laps=TOTAL, priors=dc_replace(DEFAULT_PRIORS, offset_sd=0.5)
    )

    a, b = plain.offset("MEDIUM", "HARD"), wide.offset("MEDIUM", "HARD")
    assert a.delta == b.delta and a.se == b.se  # a estimativa não muda
    assert a.predictive_sd < 0.3 and b.predictive_sd == pytest.approx(
        np.hypot(a.predictive_sd, 0.5)
    )
    assert plain.offset("HARD", "HARD").predictive_sd == 0.0  # igual a si mesmo: sem incerteza
