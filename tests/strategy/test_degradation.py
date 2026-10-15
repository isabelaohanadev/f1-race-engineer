"""Testes do modelo de degradação (strategy/degradation.py) com corridas SINTÉTICAS em que a
inclinação de cada stint, a variação entre stints (tau) e o ruído de volta são plantados."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from strategy import degradation
from strategy.degradation import (
    BETWEEN_COMPOUNDS_FLOOR,
    MIN_UNIT_LAPS,
    TAU_FALLBACK,
    coverage,
    estimate_tau,
    fit_degradation,
    future_stints,
    random_effects_mean,
    shrink_slope,
)
from strategy.fuel import FuelVariant, laps_until, physical_coefficient
from strategy.laps import LapRow

TOTAL = 60
PHYS = physical_coefficient(TOTAL)
SLOPES = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
OFFSETS = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}


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


def make_race(
    *,
    drivers: int = 20,
    tau: float = 0.02,
    noise: float = 0.2,
    kappa: float = PHYS,
    seed: int = 0,
    slopes: dict[str, float] = SLOPES,
    compounds: tuple[str, ...] | None = None,
) -> list[LapRow]:
    """Cada piloto faz 2 stints, cada um com a SUA inclinação (a do composto + N(0, tau)): é a
    variação entre stints que o modelo precisa enxergar. As voltas de entrada e de saída dos boxes
    existem (lentas, com as flags), como nos dados reais."""
    rng = np.random.default_rng(seed)
    names = list(compounds or slopes)
    rows = []
    for d in range(drivers):
        code = f"D{d:02d}"
        base = 90.0 + 0.2 * d
        c1, c2 = (rng.choice(names, size=2, replace=len(names) < 2)).tolist()
        pit = int(rng.integers(14, 40))
        age1, age2 = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        s1, s2 = (slopes[c] + rng.normal(0, tau) for c in (c1, c2))
        for n in range(2, pit):
            age = age1 + (n - 1)
            t = base + OFFSETS[c1] + s1 * age - kappa * (n - 1) + rng.normal(0, noise)
            rows.append(lap(code, n, t, c1, age, 1))
        rows.append(lap(code, pit, base + 30, c1, age1 + pit - 1, 1, pit_in=True))
        rows.append(lap(code, pit + 1, base + 25, c2, age2, 2, pit_out=True))
        for n in range(pit + 2, TOTAL + 1):
            age = age2 + (n - pit - 2)
            t = base + OFFSETS[c2] + s2 * age - kappa * (n - 1) + rng.normal(0, noise)
            rows.append(lap(code, n, t, c2, age, 2))
    return rows


def fit(rows, *, upto=TOTAL, variant=FuelVariant.ENCOLHIMENTO):
    return fit_degradation(rows, upto_lap=upto, total_laps=TOTAL, variant=variant)


@pytest.fixture(scope="module")
def race():
    return make_race(seed=1)


@pytest.fixture(scope="module")
def model(race):
    return fit(race)


# ------------------------------------------------------------------ as peças puras


def test_random_effects_mean_gives_more_weight_to_the_precise_unit():
    mean, se = random_effects_mean(np.array([0.05, 0.15]), np.array([0.001, 0.05]), tau=0.0)

    assert mean == pytest.approx(0.05, abs=0.001)
    assert se == pytest.approx(1 / np.sqrt(1 / 0.001**2 + 1 / 0.05**2))


def test_random_effects_mean_flattens_the_weights_as_tau_grows():
    """Com muita variação real entre unidades, nenhuma deve mandar sozinha."""
    slopes, ses = np.array([0.05, 0.15]), np.array([0.001, 0.05])

    precise, _ = random_effects_mean(slopes, ses, tau=0.0)
    flat, _ = random_effects_mean(slopes, ses, tau=1.0)

    assert flat == pytest.approx(0.10, abs=0.001) and flat > precise


def test_tau_is_zero_when_the_observed_differences_are_only_sampling_noise():
    slopes = np.array([0.070, 0.071, 0.069, 0.070, 0.072, 0.068])
    ses = np.full(6, 0.02)  # o erro de cada um é 10x maior que a dispersão entre eles

    tau, source = estimate_tau(slopes, ses, ["MEDIUM"] * 6)

    assert tau == 0.0 and source == "estimado"


def test_tau_recovers_the_planted_variation_between_units():
    rng = np.random.default_rng(1)
    slopes = 0.07 + rng.normal(0, 0.03, 200)  # variação real de 0,03
    ses = np.full(200, 0.01)

    tau, _ = estimate_tau(slopes, ses, ["MEDIUM"] * 200)

    assert tau == pytest.approx(0.03, abs=0.006)


def test_tau_does_not_mistake_different_compounds_for_heterogeneity():
    """HARD e SOFT têm degradações diferentes: isso é diferença entre compostos, não variação entre
    pilotos. As médias são por composto, então dois grupos homogêneos dão tau ~ 0."""
    rng = np.random.default_rng(2)
    slopes = np.concatenate([0.04 + rng.normal(0, 0.002, 50), 0.12 + rng.normal(0, 0.002, 50)])
    strata = ["HARD"] * 50 + ["SOFT"] * 50

    tau, _ = estimate_tau(slopes, np.full(100, 0.01), strata)

    assert tau < 0.005


def test_tau_falls_back_to_the_default_when_there_are_too_few_units():
    tau, source = estimate_tau(np.array([0.05, 0.09]), np.array([0.01, 0.01]), ["HARD", "HARD"])

    assert (tau, source) == (TAU_FALLBACK, "padrão")


def test_shrink_is_halfway_when_the_standard_error_equals_tau():
    slope, se, weight = shrink_slope(0.20, se=0.02, prior_mean=0.06, prior_se=0.005, tau=0.02)

    assert weight == pytest.approx(0.5)
    assert slope == pytest.approx(0.13)
    assert se == pytest.approx(np.sqrt(0.5 * 0.02**2 + 0.25 * 0.005**2))


def test_shrink_keeps_the_prior_when_the_observation_agrees_with_it():
    slope, _, _ = shrink_slope(0.06, se=0.05, prior_mean=0.06, prior_se=0.005, tau=0.02)

    assert slope == pytest.approx(0.06)


def test_a_driver_with_very_noisy_data_is_pulled_to_the_prior_and_a_precise_one_is_not():
    noisy, _, w_noisy = shrink_slope(0.20, se=0.20, prior_mean=0.06, prior_se=0.005, tau=0.02)
    precise, _, w_precise = shrink_slope(0.20, se=0.001, prior_mean=0.06, prior_se=0.005, tau=0.02)

    assert w_noisy < 0.02 and noisy == pytest.approx(0.06, abs=0.003)
    assert w_precise > 0.99 and precise == pytest.approx(0.20, abs=0.002)


# ------------------------------------------------------------------ recuperar o que foi plantado


def test_the_compound_slopes_recover_the_planted_degradation(model):
    for compound, truth in SLOPES.items():
        prior = model.compounds[compound]
        assert abs(prior.mean - truth) < 3 * prior.se + 0.004, compound  # +0,004: o erro do k


def test_the_compounds_come_out_in_the_planted_order(model):
    means = {c: p.mean for c, p in model.compounds.items()}

    assert means["SOFT"] > means["MEDIUM"] > means["HARD"]


def test_the_lap_noise_is_the_planted_noise(race):
    """Ruído de volta plantado 0,2 s; o modelo mede o desvio das voltas em torno da reta."""
    assert fit(race).sigma_lap == pytest.approx(0.2, rel=0.2)


def test_the_between_stint_variation_is_the_planted_one():
    estimated = [fit(make_race(tau=0.03, seed=s)).tau for s in range(5)]
    none = [fit(make_race(tau=0.0, seed=s)).tau for s in range(5)]

    assert np.median(estimated) == pytest.approx(0.03, abs=0.012)
    assert np.median(none) < 0.012
    assert np.median(estimated) > 2 * np.median(none)


def test_tau_is_reported_as_estimated_when_there_are_data_for_it(model):
    assert model.tau_source == "estimado"


# ------------------------------------------------------------------ as duas incertezas, separadas


def test_the_parameter_error_shrinks_with_more_data_but_the_lap_noise_does_not():
    """O ponto da separação: mais dados tornam a inclinação mais certa (se cai), mas o ruído de UMA
    volta continua o mesmo (só o medimos melhor). Misturar os dois é o erro clássico."""
    small = fit(make_race(drivers=8, seed=3))
    large = fit(make_race(drivers=40, seed=3))

    se_small = small.compounds["MEDIUM"].se
    se_large = large.compounds["MEDIUM"].se
    assert se_large < 0.75 * se_small
    assert small.sigma_lap == pytest.approx(large.sigma_lap, rel=0.15)


def test_the_predictive_error_adds_the_between_stint_variation_to_the_parameter_error(model):
    for estimate in model.units.values():
        assert estimate.predictive_sd == pytest.approx(np.hypot(estimate.se, estimate.tau))
        assert estimate.predictive_sd >= max(estimate.se, estimate.tau)


def test_a_slope_estimate_carries_all_three_numbers_with_their_own_names(model):
    estimate = model.slope("D00", "MEDIUM")

    assert estimate is not None
    assert estimate.se > 0 and estimate.tau == model.tau and estimate.sigma_lap > 0
    assert estimate.sigma_lap != estimate.se  # não são o mesmo número com dois nomes


def test_the_standard_error_alone_is_overconfident_for_the_next_stint():
    """Por que a variação entre stints existe: com tau real de 0,03 e erro-padrão ~0,01, o intervalo
    só com o erro-padrão cobre pouco do stint futuro; com tau, cobre perto do nominal."""
    stints = []
    for seed in range(25):
        stints += future_stints(
            f"S{seed}",
            make_race(tau=0.03, seed=seed, kappa=PHYS),
            total_laps=TOTAL,
            variant=FuelVariant.FISICO,
        )

    only_se, with_tau = coverage(stints, "param"), coverage(stints, "predictive")

    assert len(stints) > 150
    assert only_se < 0.80  # confiante demais
    assert 0.88 <= with_tau <= 0.99  # perto de 95%
    assert with_tau - only_se > 0.15


# ------------------------------------------------------------------ encolhimento para o composto


def extreme_driver(laps: int, seed: int = 0) -> list[LapRow]:
    """O D99: inclinação ENORME (0,40) num stint de `laps` voltas, com o ruído de volta realista
    (0,2) das demais (sem ruído o ajuste seria perfeito e o erro-padrão iria a zero)."""
    rng = np.random.default_rng(seed)
    return [
        lap("D99", n, 90 + 0.40 * (n + 3) - PHYS * (n - 1) + rng.normal(0, 0.2), "MEDIUM", n + 3, 1)
        for n in range(2, 2 + laps)
    ]


def race_with_a_short_extreme_driver(laps: int = 8) -> list[LapRow]:
    return make_race(tau=0.01, seed=4) + extreme_driver(laps)


def test_a_driver_with_few_laps_is_pulled_toward_the_compound_mean():
    """O encolhimento faz o que promete: a inclinação de 8 voltas (planta-se 0,40, enorme) vem para
    perto do prior, e o dado dele pesa menos do que pesaria com muitas voltas."""
    model = fit(race_with_a_short_extreme_driver())

    estimate = model.units[("D99", "MEDIUM")]
    prior = model.compounds["MEDIUM"]

    assert estimate.n_laps == 8
    assert 0 < estimate.weight < 0.9  # não é o dado dele sozinho
    assert prior.mean < estimate.slope < 0.38  # entre o prior e o valor bruto (~0,40)


def test_a_driver_with_many_laps_keeps_most_of_his_own_estimate(model):
    own = max(model.units.values(), key=lambda e: e.n_laps)

    assert own.weight > 0.5 and own.source == "piloto"


def test_the_weight_of_the_drivers_own_data_grows_with_his_laps():
    short = fit(race_with_a_short_extreme_driver(8)).units[("D99", "MEDIUM")]
    long = fit(race_with_a_short_extreme_driver(38)).units[("D99", "MEDIUM")]

    assert long.n_laps > short.n_laps
    assert long.weight > short.weight + 0.15
    assert long.se < short.se


def test_a_driver_with_fewer_laps_than_the_minimum_gets_no_estimate_of_his_own():
    model = fit(race_with_a_short_extreme_driver(MIN_UNIT_LAPS - 1))

    assert ("D99", "MEDIUM") not in model.units
    assert model.slope("D99", "MEDIUM").source == "composto"


# ------------------------------------------------------------------ a cadeia de recursos


def test_a_driver_who_never_used_the_compound_gets_the_compound_mean(model):
    estimate = model.slope("NOVO", "MEDIUM")

    assert estimate.source == "composto" and estimate.weight == 0.0 and estimate.n_laps == 0
    assert estimate.slope == model.compounds["MEDIUM"].mean
    assert estimate.predictive_sd == pytest.approx(np.hypot(estimate.se, model.tau))


def test_a_compound_nobody_ran_falls_back_to_the_others_with_a_wider_interval():
    race = make_race(seed=5, compounds=("MEDIUM", "HARD"))  # ninguém roda SOFT

    model = fit(race)
    asked = model.slope("D00", "SOFT")
    known = model.slope("NOVO", "MEDIUM")

    assert asked.source == "todos os compostos" and "SOFT" not in model.compounds
    assert asked.predictive_sd > known.predictive_sd  # a diferença entre compostos entra
    assert asked.se >= BETWEEN_COMPOUNDS_FLOOR  # com piso: dois compostos parecidos enganam


def test_a_wet_race_has_no_dry_degradation_model():
    wet = [replace(r, compound="INTERMEDIATE") for r in make_race(seed=6)]

    model = fit(wet)

    assert model.compounds == {} and model.units == {} and model.pooled is None
    assert model.slope("D00", "INTERMEDIATE") is None
    assert model.tau_source == "padrão"


def test_before_any_data_there_is_no_model():
    model = fit(make_race(seed=7), upto=1)

    assert model.units == {} and model.slope("D00", "MEDIUM") is None


# ------------------------------------------------------------------ sem vazamento do futuro


def test_the_model_at_lap_n_ignores_everything_after_n(race):
    """O teste explícito de vazamento: corrompe TODAS as voltas depois de N (+60 s, o composto e a
    idade trocados, a flag de saída dos boxes invertida) e o modelo da volta N não muda nem na
    última casa decimal."""
    n = 35
    corrupted = [
        replace(
            r,
            lap_time_ms=r.lap_time_ms + 60_000,
            compound="SOFT",
            tyre_life=999,
            pit_out=not r.pit_out,
        )
        if r.lap_number > n
        else r
        for r in race
    ]

    for variant in FuelVariant:
        honest = fit(race, upto=n, variant=variant)
        tampered = fit(corrupted, upto=n, variant=variant)
        assert tampered == honest, variant


def test_the_model_at_lap_n_equals_the_model_of_a_race_that_stopped_at_n(race):
    n = 35

    assert fit(race, upto=n) == fit(laps_until(race, n), upto=n)


def test_a_later_cut_sees_more_and_so_changes_the_model(race):
    """Sanidade do teste anterior: o corte importa (não é que o modelo ignora tudo)."""
    assert fit(race, upto=30) != fit(race, upto=TOTAL)


# ------------------------------------------------------------------ o k muda a degradação 1 para 1


def test_the_degradation_shifts_by_exactly_the_change_in_k_and_the_uncertainties_do_not(race):
    physical = fit(race, variant=FuelVariant.FISICO)
    shrunk = fit(race, variant=FuelVariant.ENCOLHIMENTO)
    delta_k = shrunk.fuel.kappa - physical.fuel.kappa

    assert abs(delta_k) > 0.001  # premissa: as duas variantes diferem
    for compound, prior in physical.compounds.items():
        assert shrunk.compounds[compound].mean - prior.mean == pytest.approx(delta_k, abs=1e-9)
        assert shrunk.compounds[compound].se == pytest.approx(prior.se)
    for key, estimate in physical.units.items():
        assert shrunk.units[key].slope - estimate.slope == pytest.approx(delta_k, abs=1e-9)
        assert shrunk.units[key].se == pytest.approx(estimate.se)
    assert shrunk.tau == pytest.approx(physical.tau, rel=1e-6)
    assert shrunk.sigma_lap == pytest.approx(physical.sigma_lap, rel=1e-6)


def test_a_forced_k_is_used_as_given_and_skips_the_estimate_of_k(race):
    """`kappa=` força o `k` (o undercut o desloca em +- sd para medir a dependência): o modelo usa
    exatamente esse valor, e o resultado é o mesmo qualquer que seja a variante pedida, porque não
    há nada a estimar (e o jackknife, ~20 ajustes, não é refeito)."""
    forced = {
        v: fit_degradation(race, upto_lap=TOTAL, total_laps=TOTAL, variant=v, kappa=0.07)
        for v in FuelVariant
    }

    assert {m.fuel.kappa for m in forced.values()} == {0.07}
    assert forced[FuelVariant.FISICO] == forced[FuelVariant.ENCOLHIMENTO]
    assert forced[FuelVariant.FISICO] == forced[FuelVariant.ESTIMADO]
    assert forced[FuelVariant.ENCOLHIMENTO].fuel.se is None  # sem jackknife


def test_forcing_k_to_its_own_value_changes_nothing_but_the_bookkeeping(race, model):
    """Forçar o `k` que o próprio modelo escolheu devolve as mesmas inclinações e incertezas."""
    again = fit_degradation(
        race,
        upto_lap=TOTAL,
        total_laps=TOTAL,
        variant=FuelVariant.ENCOLHIMENTO,
        kappa=model.fuel.kappa,
    )

    assert again.compounds == model.compounds
    assert again.units == model.units
    assert (again.tau, again.sigma_lap) == (model.tau, model.sigma_lap)


# ------------------------------------------------------------------ stints futuros


def test_future_stints_are_only_those_that_start_after_the_cut():
    race = make_race(seed=8)
    stints = future_stints("x", race, total_laps=TOTAL, fraction=0.5, variant=FuelVariant.FISICO)

    first_laps = {
        (r.driver, r.stint): min(
            x.lap_number for x in race if (x.driver, x.stint) == (r.driver, r.stint)
        )
        for r in race
    }
    assert stints
    for s in stints:
        started = [first for (driver, _), first in first_laps.items() if driver == s.driver]
        assert max(started) > TOTAL // 2  # o piloto tem um stint que começou depois do corte


def test_coverage_of_no_stints_is_none():
    assert coverage([], "param") is None and coverage([], "predictive") is None


# ------------------------------------------------------------------ relatório e CLI


@pytest.fixture(scope="module")
def reports():
    return [degradation.analyze_race("2024 R99 Sintético", make_race(seed=9), TOTAL)]


def test_the_race_table_has_the_three_uncertainties_and_the_compounds(reports):
    table = degradation.race_table(reports)

    assert "ruído de volta σ" in table[0] and "variação entre pilotos/stints τ" in table[0]
    assert len(table) == 2 and all(len(row) == len(table[0]) for row in table)
    assert "±" in table[1][5]  # a inclinação sai com o erro-padrão ao lado


def test_the_variant_table_has_one_row_per_variant(reports):
    table = degradation.variant_table(reports)

    assert [row[0] for row in table[1:]] == [v.value for v in FuelVariant]


def test_the_calibration_table_compares_the_two_intervals(reports):
    table = degradation.calibration_table(reports)

    assert "cobertura só com o erro-padrão" in table[0] and "cobertura com a variação τ" in table[0]
    assert len(table) >= 3


def test_the_markdown_report_states_the_three_uncertainties_and_the_separate_races(reports):
    text = degradation.build_report(
        reports,
        reports,
        ["2024 R12 British: pista mista (76% seca)"],
        markdown=True,
        generated_on=date(2026, 10, 14),
    )

    assert text.startswith("# Modelo de degradação do pneu")
    assert "em 2026-10-14" in text
    for section in (
        "## Método",
        "## Calibração",
        "## O `k` muda a degradação",
        "## Reportadas à parte",
    ):
        assert section in text
    assert "**não medem a qualidade do modelo**" in text and "2024 R12 British" in text
    assert "predictive_sd = sqrt(se² + τ²)" in text


def test_load_reports_puts_a_wet_race_apart_and_keeps_it_out_of_the_main_list(seeded: Engine):
    with sessionmaker(seeded)() as session:
        main, separate, skipped = degradation.load_reports(session, year=2024)

    assert main == [] and skipped == ["2024 R21 São Paulo: pista molhada (0% seca)"]
    assert [r.label for r in separate] == ["2024 R21 São Paulo"]


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(degradation, "load_reports", lambda session, year=None: ([], [], []))
    monkeypatch.setattr(degradation, "get_session_factory", lambda: sessionmaker())

    assert degradation.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path, reports):
    monkeypatch.setattr(degradation, "load_reports", lambda session, year=None: (reports, [], []))
    monkeypatch.setattr(degradation, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "degradacao.md"

    assert degradation.main(["--markdown", str(target)]) == 0

    assert "2024 R99 Sintético" in capsys.readouterr().out
    assert "| 2024 R99 Sintético |" in target.read_text(encoding="utf-8")
