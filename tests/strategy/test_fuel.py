"""Testes do combustível (strategy/fuel.py) com corridas SINTÉTICAS de resposta conhecida.

Cada corrida plantada tem `k` (combustível) e a degradação de cada composto escolhidos por nós: se o
estimador não os recupera, o defeito é dele. As voltas já são "limpas" (status 1, sem box).
"""

from dataclasses import replace

import numpy as np
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from strategy import fuel
from strategy.fuel import (
    MAX_RATIO,
    MIN_COMPOUND_LAPS,
    PRIOR_REL_SD,
    CompoundSlope,
    FuelEstimate,
    FuelVariant,
    choose_fuel_coefficient,
    coefficient_series,
    compound_slopes,
    correct_for_fuel,
    data_weight,
    estimate_fuel_coefficient,
    fuel_coefficient,
    jackknife_se,
    laps_until,
    max_step,
    physical_coefficient,
    pit_stops_seen,
    sensitivity,
    shrink_coefficient,
    weight_drops,
)
from strategy.laps import LapRow

TRUE_KAPPA = 0.05
TRUE_SLOPES = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
OFFSETS = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}  # o composto mais duro é mais lento
TOTAL_LAPS = 60


def lap(driver, n, seconds, compound, age, stint, **overrides) -> LapRow:
    fields = {
        "driver": driver,
        "lap_number": n,
        "lap_time_ms": round(seconds * 1000),
        "compound": compound,
        "tyre_life": age,
        "stint": stint,
        "position": 1,
        "session_time_ms": n * 90_000,
        "pit_in": False,
        "pit_out": False,
        "track_status": "1",
    }
    return LapRow(**(fields | overrides))


def synthetic_race(
    *,
    kappa: float = TRUE_KAPPA,
    slopes: dict[str, float] = TRUE_SLOPES,
    drivers: int = 20,
    seed: int = 0,
    noise: float = 0.15,
    same_strategy: bool = False,
) -> list[LapRow]:
    """Cada piloto faz 2 stints, num composto e numa volta de parada próprios (a variação é o que
    separa o combustível da idade). Tempo = piloto + composto + degradação*idade - k*(volta - 1)."""
    rng = np.random.default_rng(seed)
    compounds = list(slopes)
    rows = []
    for d in range(drivers):
        code = f"D{d:02d}"
        base = 90.0 + 0.2 * d  # o ritmo de cada piloto
        if same_strategy:
            c1, c2, pit, age1, age2 = "MEDIUM", "HARD", 25, 1, 1
        else:
            c1, c2 = rng.choice(compounds, size=2, replace=False)
            pit = int(rng.integers(12, 42))
            age1, age2 = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        # stint 1: voltas 2..pit-1 (a volta 1 é descartada); a volta `pit` é a de ENTRADA nos
        # boxes e a seguinte a de SAÍDA (lentas e descartadas pela limpeza, como nos dados reais,
        # onde são elas que dão a contagem de paradas); stint 2: pit+2..fim.
        for n in range(2, pit):
            age = age1 + (n - 1)
            t = base + OFFSETS[c1] + slopes[c1] * age - kappa * (n - 1) + rng.normal(0, noise)
            rows.append(lap(code, n, t, c1, age, 1))
        rows.append(lap(code, pit, base + 30.0, c1, age1 + pit - 1, 1, pit_in=True))
        rows.append(lap(code, pit + 1, base + 25.0, c2, age2, 2, pit_out=True))
        for n in range(pit + 2, TOTAL_LAPS + 1):
            age = age2 + (n - pit - 2)
            t = base + OFFSETS[c2] + slopes[c2] * age - kappa * (n - 1) + rng.normal(0, noise)
            rows.append(lap(code, n, t, c2, age, 2))
    return rows


def clean_race(**kwargs) -> list[LapRow]:
    """As voltas que o estimador de fato recebe: a corrida sintética depois da limpeza (sem a volta
    de entrada nem a de saída dos boxes)."""
    return fuel.dry_clean_laps(synthetic_race(**kwargs))


# ------------------------------------------------------------------ o coeficiente físico


def test_physical_coefficient_is_thirty_milliseconds_per_kilo_over_the_race():
    assert physical_coefficient(57) == pytest.approx(0.03 * 110 / 57)
    assert physical_coefficient(57) == pytest.approx(0.0579, abs=1e-4)


def test_a_longer_race_burns_fuel_more_slowly_so_each_lap_gains_less():
    assert physical_coefficient(78) < physical_coefficient(57) < physical_coefficient(44)


def test_physical_coefficient_rejects_a_race_with_no_laps():
    with pytest.raises(ValueError, match="total_laps"):
        physical_coefficient(0)


# ------------------------------------------------------------------ a correção


def test_the_correction_leaves_lap_one_alone_and_adds_k_per_lap_after_it():
    assert correct_for_fuel(1, 90.0, 0.05) == pytest.approx(90.0)
    assert correct_for_fuel(11, 90.0, 0.05) == pytest.approx(90.5)


def test_the_correction_accepts_arrays():
    result = correct_for_fuel(np.array([1, 2, 3]), np.array([90.0, 90.0, 90.0]), 0.1)

    np.testing.assert_allclose(result, [90.0, 90.1, 90.2])


def test_the_correction_removes_the_planted_fuel_effect_exactly():
    """Sem ruído e sem degradação, o tempo corrigido com o k verdadeiro é constante."""
    laps = np.arange(1, 50)
    raw = 90.0 - 0.05 * (laps - 1)

    np.testing.assert_allclose(correct_for_fuel(laps, raw, 0.05), 90.0)


# ------------------------------------------------------------------ a estimativa conjunta


def test_the_joint_estimate_recovers_the_planted_fuel_coefficient():
    estimate = estimate_fuel_coefficient(clean_race())

    assert estimate.identified and estimate.note == ""
    assert abs(estimate.kappa - TRUE_KAPPA) < 3 * estimate.se
    assert estimate.se < 0.005  # preciso, com 20 pilotos e stints que começam em voltas diferentes


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_the_estimate_is_stable_across_different_races(seed):
    estimate = estimate_fuel_coefficient(clean_race(seed=seed))

    assert abs(estimate.kappa - TRUE_KAPPA) < 0.01


def test_with_no_fuel_effect_the_estimate_is_near_zero():
    estimate = estimate_fuel_coefficient(clean_race(kappa=0.0, seed=5))

    assert abs(estimate.kappa) < 3 * estimate.se


def test_it_is_not_identified_when_every_stint_starts_on_the_same_lap():
    """Todos param na volta 25 com pneu novo, no mesmo composto: idade e volta andam juntas em
    todo stint, e o combustível não se separa da degradação (o `lstsq` devolveria um número)."""
    estimate = estimate_fuel_coefficient(clean_race(same_strategy=True))

    assert not estimate.identified
    assert estimate.kappa is None and estimate.se is None
    assert "não se separa" in estimate.note


def test_too_little_data_is_reported_instead_of_estimated():
    few = clean_race(drivers=3)

    estimate = estimate_fuel_coefficient(few)

    assert not estimate.identified and "poucos dados" in estimate.note


def test_the_estimate_survives_a_few_gross_outliers():
    rows = clean_race(seed=6)
    late = [i for i, r in enumerate(rows) if r.tyre_life and r.tyre_life > 20]
    for i in late[::25]:  # voltas 8 s mais lentas no fim dos stints (tráfego com pneu gasto)
        rows[i] = replace(rows[i], lap_time_ms=rows[i].lap_time_ms + 8_000)

    estimate = estimate_fuel_coefficient(rows)

    assert abs(estimate.kappa - TRUE_KAPPA) < 0.01


def test_a_dirty_lap_reaching_the_estimator_is_rejected_loudly():
    rows = synthetic_race()
    rows[0] = replace(rows[0], tyre_life=None)

    with pytest.raises(ValueError, match="volta suja"):
        estimate_fuel_coefficient(rows)


# ------------------------------------------------------------------ a escolha


def est(kappa, se=0.003, note=""):
    return FuelEstimate(kappa, se, 800, 40, 20, note)


PHYS = 0.058


def test_a_close_and_precise_estimate_is_used():
    choice = choose_fuel_coefficient(est(0.070), PHYS)

    assert (choice.source, choice.kappa) == ("estimado", 0.070)
    assert choice.ratio == pytest.approx(0.070 / PHYS)


@pytest.mark.parametrize("kappa", [PHYS * MAX_RATIO, PHYS / MAX_RATIO])
def test_the_ratio_limits_themselves_are_accepted(kappa):
    assert choose_fuel_coefficient(est(kappa), PHYS).source == "estimado"


@pytest.mark.parametrize("kappa", [PHYS * 2.01, PHYS / 2.01, 0.0, -0.03])
def test_a_divergent_or_wrong_signed_estimate_falls_back_to_the_physical_one(kappa):
    choice = choose_fuel_coefficient(est(kappa), PHYS)

    assert (choice.source, choice.kappa) == ("físico", PHYS)
    assert "diverge" in choice.reason


def test_an_imprecise_estimate_is_not_used_even_if_it_looks_right():
    choice = choose_fuel_coefficient(est(PHYS, se=0.04), PHYS)  # erro-padrão > metade do físico

    assert choice.source == "físico" and "impreciso" in choice.reason


def test_an_unidentified_estimate_falls_back_and_says_why():
    choice = choose_fuel_coefficient(est(None, None, "poucos dados (10 voltas)"), PHYS)

    assert (choice.source, choice.kappa, choice.ratio) == ("físico", PHYS, None)
    assert "poucos dados" in choice.reason


def test_the_physical_value_is_always_carried_for_comparison():
    assert choose_fuel_coefficient(est(0.07), PHYS).physical == PHYS
    assert choose_fuel_coefficient(est(None, None), PHYS).physical == PHYS


# ------------------------------------------------------------------ degradação por composto


def by_compound(slopes: list[CompoundSlope]) -> dict[str, CompoundSlope]:
    return {s.compound: s for s in slopes}


def test_compound_slopes_recover_the_planted_degradation_at_the_true_k():
    slopes = by_compound(compound_slopes(clean_race(), TRUE_KAPPA))

    for compound, truth in TRUE_SLOPES.items():
        assert abs(slopes[compound].slope - truth) < 3 * slopes[compound].se, compound


def test_without_the_fuel_correction_the_degradation_comes_out_too_low_by_exactly_k():
    """O motivo da correção: o ganho de combustível mascara o desgaste."""
    race = clean_race()
    raw = by_compound(compound_slopes(race, 0.0))
    fixed = by_compound(compound_slopes(race, TRUE_KAPPA))

    for compound in TRUE_SLOPES:
        assert fixed[compound].slope - raw[compound].slope == pytest.approx(TRUE_KAPPA, abs=1e-9)


def test_a_compound_with_too_few_laps_has_no_readable_slope():
    race = clean_race()
    few = [r for r in race if r.compound != "SOFT"] + [r for r in race if r.compound == "SOFT"][
        : MIN_COMPOUND_LAPS - 1
    ]

    slopes = by_compound(compound_slopes(few, TRUE_KAPPA))

    assert slopes["SOFT"].slope is None and slopes["SOFT"].se is None
    assert slopes["HARD"].slope is not None


def test_a_compound_with_a_single_lap_does_not_invent_a_slope():
    """Sem a checagem de identificabilidade o `lstsq` chegou a devolver +32 s/volta (dado real)."""
    race = clean_race()
    one = [r for r in race if r.compound != "SOFT"] + [
        next(r for r in race if r.compound == "SOFT")
    ]

    slopes = by_compound(compound_slopes(one, TRUE_KAPPA))

    assert slopes["SOFT"].slope is None


def test_compound_slopes_of_no_laps_is_empty():
    assert compound_slopes([], 0.05) == []


# ------------------------------------------------------------------ sensibilidade


def test_sensitivity_shifts_every_compounds_degradation_by_exactly_the_change_in_k():
    """Dentro do stint volta = idade + constante: somar k*volta soma k à inclinação da idade."""
    points = sensitivity(clean_race(), PHYS)
    base = points[0]

    assert base.multiplier == 0.0
    for point in points[1:]:
        for compound, slope in point.slopes.items():
            shift = slope.slope - base.slopes[compound].slope
            assert shift == pytest.approx(point.kappa, abs=1e-9), (point.multiplier, compound)


def test_sensitivity_keeps_the_order_between_compounds():
    points = sensitivity(clean_race(), PHYS)

    orders = {tuple(sorted(p.slopes, key=lambda c: p.slopes[c].slope)) for p in points}

    assert len(orders) == 1  # todos os compostos sobem juntos: a ordem não muda com k


def test_sensitivity_covers_the_requested_multipliers_of_the_physical_value():
    points = sensitivity(clean_race(), PHYS, multipliers=(0.5, 1.0, 1.5))

    assert [p.multiplier for p in points] == [0.5, 1.0, 1.5]
    assert [p.kappa for p in points] == pytest.approx([0.5 * PHYS, PHYS, 1.5 * PHYS])


# ------------------------------------------------------------------ sem vazamento do futuro


def test_laps_until_drops_everything_after_the_cut():
    race = synthetic_race()

    cut = laps_until(race, 30)

    assert max(r.lap_number for r in cut) == 30
    assert len(cut) < len(race)


def test_the_decision_at_a_given_point_does_not_depend_on_what_happens_afterwards():
    """O teste de vazamento do combustível: corrompe TODAS as voltas depois de 50% da corrida (+60 s
    em cada uma) e a decisão tomada com dados até 50% não pode mudar nem na última casa."""
    race = synthetic_race(seed=7)
    half = TOTAL_LAPS // 2
    corrupted = [
        replace(r, lap_time_ms=r.lap_time_ms + 60_000) if r.lap_number > half else r for r in race
    ]

    honest = fuel.analyze_race("x", race, TOTAL_LAPS)
    tampered = fuel.analyze_race("x", corrupted, TOTAL_LAPS)

    assert tampered.progress[0.5] == honest.progress[0.5]
    assert tampered.progress[0.25] == honest.progress[0.25]
    assert tampered.shrink_progress[0.5] == honest.shrink_progress[0.5]
    assert tampered.shrink_progress[0.25] == honest.shrink_progress[0.25]
    assert tampered.full != honest.full  # e a corrida inteira, que VÊ o futuro, claro que muda


# ------------------------------------------------------------------ encolhimento contínuo

PHYS_SYN = physical_coefficient(TOTAL_LAPS)  # 0,055 s/volta na corrida sintética de 60 voltas


def test_shrink_without_an_estimate_is_the_physical_value_with_zero_weight():
    for hat, se in ((None, None), (0.09, None), (None, 0.003), (0.09, float("inf"))):
        assert shrink_coefficient(hat, se, PHYS_SYN) == (PHYS_SYN, 0.0)


def test_shrink_weight_is_half_at_the_prior_scale_and_goes_to_the_extremes():
    tau = PRIOR_REL_SD * PHYS_SYN

    assert shrink_coefficient(0.07, tau, PHYS_SYN)[1] == pytest.approx(0.5)
    assert shrink_coefficient(0.07, 1e-9, PHYS_SYN)[1] == pytest.approx(1.0)
    assert shrink_coefficient(0.07, 1e9, PHYS_SYN)[1] == pytest.approx(0.0, abs=1e-12)


def test_shrink_weight_falls_monotonically_as_the_standard_error_grows():
    ses = np.geomspace(1e-4, 1e3, 60)

    weights = [shrink_coefficient(0.07, se, PHYS_SYN)[1] for se in ses]

    assert all(a >= b for a, b in zip(weights, weights[1:], strict=False))


def test_shrink_is_continuous_where_the_estimate_appears():
    """O que a regra binária não tem: sem estimativa é o físico, e uma estimativa com erro-padrão
    enorme é praticamente o físico também. Aparecer não move `k`; o peso sobe de 0."""
    unavailable, _ = shrink_coefficient(None, None, PHYS_SYN)
    barely_there, _ = shrink_coefficient(0.09, 1e6, PHYS_SYN)

    assert barely_there == pytest.approx(unavailable, abs=1e-9)


def test_the_data_weight_is_what_keeps_a_precise_estimate_from_stepping_in():
    """Uma estimativa precisa mas com poucas paradas vistas quase não entra: sem o peso de dados, o
    erro-padrão pequeno daria w = 1 de uma vez (o degrau que a regra binária tem)."""
    few_stops, w_few = shrink_coefficient(0.09, 1e-6, PHYS_SYN, data=0.05)
    many_stops, w_many = shrink_coefficient(0.09, 1e-6, PHYS_SYN, data=1.0)

    assert w_few == pytest.approx(0.05) and few_stops == pytest.approx(PHYS_SYN, abs=0.002)
    assert w_many == pytest.approx(1.0) and many_stops == pytest.approx(0.09)


@pytest.mark.parametrize("hat", [1e6, -1e6, 0.0, -0.05, 20 * PHYS_SYN, PHYS_SYN / 100])
def test_no_estimate_however_absurd_can_move_k_outside_the_bounds(hat):
    """Cedo na corrida o estimador chegou a devolver 19x o físico com erro-padrão pequeno."""
    kappa, weight = shrink_coefficient(hat, 1e-9, PHYS_SYN, data=1.0)

    assert PHYS_SYN / MAX_RATIO - 1e-12 <= kappa <= PHYS_SYN * MAX_RATIO + 1e-12
    assert weight == pytest.approx(1.0)


def test_pit_stops_seen_counts_exits_but_not_a_pit_lane_start_or_unknown_flags():
    rows = [
        lap("A", 1, 90, "SOFT", 1, 1, pit_out=True),  # largou dos boxes: não é parada
        lap("A", 20, 90, "SOFT", 20, 1, pit_in=True),
        lap("A", 21, 90, "HARD", 1, 2, pit_out=True),  # parada
        lap("B", 30, 90, "SOFT", 30, 1, pit_out=None),  # dado antigo: não sei, não conta
        lap("B", 31, 90, "SOFT", 31, 1, pit_out=True),  # parada
    ]

    assert pit_stops_seen(rows) == 2


def test_data_weight_is_the_stops_per_driver_capped_at_one():
    def stops(n_drivers: int, n_stops: int) -> list[LapRow]:
        rows = [lap(f"D{i}", 5, 90, "SOFT", 5, 1) for i in range(n_drivers)]
        return rows + [lap(f"D{i}", 25, 90, "HARD", 1, 2, pit_out=True) for i in range(n_stops)]

    assert data_weight([]) == 0.0
    assert data_weight(stops(20, 0)) == 0.0
    assert data_weight(stops(20, 10)) == pytest.approx(0.5)
    assert data_weight(stops(20, 20)) == 1.0
    assert data_weight(stops(10, 18)) == 1.0  # duas paradas por piloto não passam de 1


def test_data_weight_never_decreases_as_laps_are_added():
    race = synthetic_race(seed=3)

    weights = [data_weight(laps_until(race, n)) for n in range(1, TOTAL_LAPS + 1)]

    assert all(a <= b for a, b in zip(weights, weights[1:], strict=False))
    assert weights[0] == 0.0 and weights[-1] == 1.0


def test_jackknife_needs_enough_drivers_and_is_deterministic():
    laps = fuel.dry_clean_laps(synthetic_race(seed=4))

    se = jackknife_se(laps)

    assert se is not None and se > 0
    assert jackknife_se(laps) == se
    assert jackknife_se([r for r in laps if r.driver in ("D00", "D01", "D02")]) is None


# ------------------------------------------------------------------ as três variantes


def test_the_physical_variant_is_the_physical_value_at_every_lap():
    race = synthetic_race(seed=5)

    for n in (10, 30, TOTAL_LAPS):
        c = fuel_coefficient(race, upto_lap=n, total_laps=TOTAL_LAPS, variant=FuelVariant.FISICO)
        assert (c.kappa, c.weight) == (PHYS_SYN, 0.0)


def test_the_estimated_variant_is_the_binary_rule():
    race = synthetic_race(kappa=1.6 * PHYS_SYN, seed=5)
    laps = fuel.dry_clean_laps(laps_until(race, 45))

    c = fuel_coefficient(race, upto_lap=45, total_laps=TOTAL_LAPS, variant=FuelVariant.ESTIMADO)
    rule = choose_fuel_coefficient(estimate_fuel_coefficient(laps), PHYS_SYN)

    assert c.kappa == rule.kappa and c.weight in (0.0, 1.0)


def test_the_shrinkage_variant_starts_physical_and_ends_at_the_estimate():
    race = synthetic_race(kappa=1.6 * PHYS_SYN, seed=5)
    early = fuel_coefficient(
        race, upto_lap=9, total_laps=TOTAL_LAPS, variant=FuelVariant.ENCOLHIMENTO
    )
    late = fuel_coefficient(
        race, upto_lap=TOTAL_LAPS, total_laps=TOTAL_LAPS, variant=FuelVariant.ENCOLHIMENTO
    )

    assert (early.kappa, early.weight) == (PHYS_SYN, 0.0)
    assert late.weight > 0.95 and late.kappa == pytest.approx(1.6 * PHYS_SYN, rel=0.05)


@pytest.mark.parametrize("variant", list(FuelVariant))
def test_every_variant_ignores_the_future_by_construction(variant):
    """Sem vazamento do futuro: corrompe TODAS as voltas depois de N (tempo +60 s e a flag de saída
    dos boxes invertida) e o k das três variantes na volta N não muda nem na última casa."""
    race = synthetic_race(kappa=1.6 * PHYS_SYN, seed=6)
    n = 35
    corrupted = [
        replace(r, lap_time_ms=r.lap_time_ms + 60_000, pit_out=not r.pit_out)
        if r.lap_number > n
        else r
        for r in race
    ]

    honest = fuel_coefficient(race, upto_lap=n, total_laps=TOTAL_LAPS, variant=variant)
    tampered = fuel_coefficient(corrupted, upto_lap=n, total_laps=TOTAL_LAPS, variant=variant)

    assert tampered == honest


def test_a_large_jackknife_error_pulls_k_back_to_the_physical_value(monkeypatch):
    """O erro-padrão por cluster sai pequeno cedo na corrida mesmo quando a estimativa depende de
    poucos pilotos; o do jackknife não. O `k` usa o MAIOR dos dois: aqui o jackknife manda."""
    race = synthetic_race(kappa=1.6 * PHYS_SYN, seed=5)
    kwargs = {"upto_lap": TOTAL_LAPS, "total_laps": TOTAL_LAPS, "variant": FuelVariant.ENCOLHIMENTO}

    trusting = fuel_coefficient(race, **kwargs)
    monkeypatch.setattr(fuel, "jackknife_se", lambda laps: 1.0)  # a estimativa depende de 1 piloto
    doubtful = fuel_coefficient(race, **kwargs)

    assert trusting.weight > 0.95
    assert doubtful.weight < 0.01
    assert doubtful.kappa == pytest.approx(PHYS_SYN, abs=0.001)


def test_a_missing_jackknife_means_physical(monkeypatch):
    race = synthetic_race(kappa=1.6 * PHYS_SYN, seed=5)
    monkeypatch.setattr(fuel, "jackknife_se", lambda laps: None)

    c = fuel_coefficient(
        race, upto_lap=TOTAL_LAPS, total_laps=TOTAL_LAPS, variant=FuelVariant.ENCOLHIMENTO
    )

    assert (c.kappa, c.weight) == (PHYS_SYN, 0.0) and "poucos pilotos" in c.reason


@pytest.fixture(scope="module")
def series_16x():
    """As séries volta a volta (regra e encolhimento) de uma corrida com k = 1,6x o físico."""
    race = synthetic_race(kappa=1.6 * PHYS_SYN, seed=0)
    return {
        variant: coefficient_series(race, total_laps=TOTAL_LAPS, variant=variant)
        for variant in (FuelVariant.ESTIMADO, FuelVariant.ENCOLHIMENTO)
    }


def test_the_shrinkage_has_no_step_where_the_rule_switches_source(series_16x):
    """O ponto do encolhimento. A regra troca de físico para estimado num degrau (k real 1,6x: o
    salto é de ~0,03 s/volta); o encolhimento chega ao mesmo valor em passos pequenos."""
    rule_step = max_step(series_16x[FuelVariant.ESTIMADO])
    shrink_step = max_step(series_16x[FuelVariant.ENCOLHIMENTO])

    assert rule_step > 0.02  # premissa: a regra TEM mesmo um degrau aqui
    assert shrink_step < 0.5 * rule_step


def test_the_shrinkage_reaches_the_same_value_as_the_rule_once_the_data_is_rich(series_16x):
    rule, shrunk = series_16x[FuelVariant.ESTIMADO], series_16x[FuelVariant.ENCOLHIMENTO]

    assert shrunk[-1].kappa == pytest.approx(rule[-1].kappa, rel=0.02)
    assert shrunk[0].kappa == pytest.approx(PHYS_SYN)  # e começa no físico, como a regra


def test_the_weight_grows_with_the_data_in_trend(series_16x):
    weights = np.array([c.weight for c in series_16x[FuelVariant.ENCOLHIMENTO]])
    quarter = len(weights) // 4

    assert weights[:quarter].mean() < 0.2 < 0.9 < weights[-quarter:].mean()
    assert weights.min() >= 0.0 and weights.max() <= 1.0


def test_the_weight_is_not_claimed_to_be_monotone_lap_by_lap(series_16x):
    """Afirma-se a tendência: as quedas materiais de `w` (> 0,01) são contadas, e são poucas."""
    series = series_16x[FuelVariant.ENCOLHIMENTO]

    assert weight_drops(series) <= len(series) // 4
    assert weight_drops(series, tolerance=-1.0) >= weight_drops(
        series
    )  # tolerância menor, mais quedas


def test_max_step_and_weight_drops_of_a_trivial_series_are_zero():
    assert max_step([]) == 0 and weight_drops([]) == 0


# ------------------------------------------------------------------ uma corrida inteira


def test_analyze_race_ties_the_pieces_together():
    race = synthetic_race(seed=8)

    report = fuel.analyze_race("2024 R99 Sintético", race, TOTAL_LAPS)

    assert report.physical == pytest.approx(physical_coefficient(TOTAL_LAPS))
    assert report.full.source in ("estimado", "físico")
    assert set(report.progress) == set(fuel.PROGRESS_FRACTIONS)
    assert len(report.sensitivity) == len(fuel.SENSITIVITY_MULTIPLIERS)
    assert set(report.chosen) == set(TRUE_SLOPES)
    assert report.n_laps == len(fuel.dry_clean_laps(race))


def test_dry_clean_laps_drop_wet_compounds_and_dirty_laps():
    rows = [
        lap("A", 5, 90, "MEDIUM", 5, 1),
        lap("A", 6, 90, "INTERMEDIATE", 6, 1),  # molhado
        lap("A", 7, 90, "MEDIUM", 7, 1, track_status="12"),  # amarela
        lap("A", 8, 90, "MEDIUM", 8, 1, pit_in=True),  # box
    ]

    assert [r.lap_number for r in fuel.dry_clean_laps(rows)] == [5]


# ------------------------------------------------------------------ relatório e CLI


def reports_for(*seeds):
    return [
        fuel.analyze_race(f"2024 R{s:02d} Sintético", synthetic_race(seed=s), TOTAL_LAPS)
        for s in seeds
    ]


def test_summary_numbers_are_computed_from_the_reports():
    text = fuel.build_report(reports_for(1, 2), [], markdown=False)

    assert "2 corridas secas analisadas" in text
    assert "identificável em 2 de 2" in text
    assert "Sensibilidade: somar `k * volta`" in text
    assert "Linhas com degradação NEGATIVA" in text


def test_markdown_report_explains_the_method_and_has_all_three_tables():
    from datetime import date

    text = fuel.build_report(
        reports_for(1), ["2024 R21 São Paulo"], markdown=True, generated_on=date(2026, 10, 14)
    )

    assert text.startswith("# Combustível")
    assert "em 2026-10-14" in text and "2024 R21 São Paulo" in text
    for section in (
        "## Método",
        "## Estimado x físico",
        "## Regra e encolhimento",
        "## Sensibilidade",
    ):
        assert section in text
    assert "fixado antes de olhar os resultados" in text


def test_the_tables_have_one_row_per_race(tmp_path):
    reports = reports_for(1, 2)

    assert len(fuel.full_table(reports)) == 3
    assert len(fuel.progress_table(reports)) == 3
    assert all(len(row) == len(fuel.full_table(reports)[0]) for row in fuel.full_table(reports))


def test_load_reports_skips_races_that_are_not_dry(seeded: Engine):
    with sessionmaker(seeded)() as session:
        reports, skipped = fuel.load_reports(session, year=2024)

    assert reports == []
    assert skipped == ["2024 R21 São Paulo: pista molhada (0% seca)"]


def test_main_without_dry_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(fuel, "load_reports", lambda session, year=None, **kw: ([], []))
    monkeypatch.setattr(fuel, "get_session_factory", lambda: sessionmaker())

    assert fuel.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    reports = reports_for(1)
    monkeypatch.setattr(fuel, "load_reports", lambda session, year=None, **kw: (reports, []))
    monkeypatch.setattr(fuel, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "combustivel.md"

    assert fuel.main(["--markdown", str(target)]) == 0

    assert "2024 R01 Sintético" in capsys.readouterr().out
    assert "| 2024 R01 Sintético |" in target.read_text(encoding="utf-8")
