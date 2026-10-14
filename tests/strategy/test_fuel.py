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
    CompoundSlope,
    FuelEstimate,
    choose_fuel_coefficient,
    compound_slopes,
    correct_for_fuel,
    estimate_fuel_coefficient,
    laps_until,
    physical_coefficient,
    sensitivity,
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
        # stint 1: voltas 2..pit (a volta 1 é descartada); stint 2: pit+2..fim (a de saída também)
        for n in range(2, pit + 1):
            age = age1 + (n - 1)
            t = base + OFFSETS[c1] + slopes[c1] * age - kappa * (n - 1) + rng.normal(0, noise)
            rows.append(lap(code, n, t, c1, age, 1))
        for n in range(pit + 2, TOTAL_LAPS + 1):
            age = age2 + (n - pit - 2)
            t = base + OFFSETS[c2] + slopes[c2] * age - kappa * (n - 1) + rng.normal(0, noise)
            rows.append(lap(code, n, t, c2, age, 2))
    return rows


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
    estimate = estimate_fuel_coefficient(synthetic_race())

    assert estimate.identified and estimate.note == ""
    assert abs(estimate.kappa - TRUE_KAPPA) < 3 * estimate.se
    assert estimate.se < 0.005  # preciso, com 20 pilotos e stints que começam em voltas diferentes


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_the_estimate_is_stable_across_different_races(seed):
    estimate = estimate_fuel_coefficient(synthetic_race(seed=seed))

    assert abs(estimate.kappa - TRUE_KAPPA) < 0.01


def test_with_no_fuel_effect_the_estimate_is_near_zero():
    estimate = estimate_fuel_coefficient(synthetic_race(kappa=0.0, seed=5))

    assert abs(estimate.kappa) < 3 * estimate.se


def test_it_is_not_identified_when_every_stint_starts_on_the_same_lap():
    """Todos param na volta 25 com pneu novo, no mesmo composto: idade e volta andam juntas em
    todo stint, e o combustível não se separa da degradação (o `lstsq` devolveria um número)."""
    estimate = estimate_fuel_coefficient(synthetic_race(same_strategy=True))

    assert not estimate.identified
    assert estimate.kappa is None and estimate.se is None
    assert "não se separa" in estimate.note


def test_too_little_data_is_reported_instead_of_estimated():
    few = synthetic_race(drivers=3)

    estimate = estimate_fuel_coefficient(few)

    assert not estimate.identified and "poucos dados" in estimate.note


def test_the_estimate_survives_a_few_gross_outliers():
    rows = synthetic_race(seed=6)
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
    slopes = by_compound(compound_slopes(synthetic_race(), TRUE_KAPPA))

    for compound, truth in TRUE_SLOPES.items():
        assert abs(slopes[compound].slope - truth) < 3 * slopes[compound].se, compound


def test_without_the_fuel_correction_the_degradation_comes_out_too_low_by_exactly_k():
    """O motivo da correção: o ganho de combustível mascara o desgaste."""
    race = synthetic_race()
    raw = by_compound(compound_slopes(race, 0.0))
    fixed = by_compound(compound_slopes(race, TRUE_KAPPA))

    for compound in TRUE_SLOPES:
        assert fixed[compound].slope - raw[compound].slope == pytest.approx(TRUE_KAPPA, abs=1e-9)


def test_a_compound_with_too_few_laps_has_no_readable_slope():
    race = synthetic_race()
    few = [r for r in race if r.compound != "SOFT"] + [r for r in race if r.compound == "SOFT"][
        : MIN_COMPOUND_LAPS - 1
    ]

    slopes = by_compound(compound_slopes(few, TRUE_KAPPA))

    assert slopes["SOFT"].slope is None and slopes["SOFT"].se is None
    assert slopes["HARD"].slope is not None


def test_a_compound_with_a_single_lap_does_not_invent_a_slope():
    """Sem a checagem de identificabilidade o `lstsq` chegou a devolver +32 s/volta (dado real)."""
    race = synthetic_race()
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
    points = sensitivity(synthetic_race(), PHYS)
    base = points[0]

    assert base.multiplier == 0.0
    for point in points[1:]:
        for compound, slope in point.slopes.items():
            shift = slope.slope - base.slopes[compound].slope
            assert shift == pytest.approx(point.kappa, abs=1e-9), (point.multiplier, compound)


def test_sensitivity_keeps_the_order_between_compounds():
    points = sensitivity(synthetic_race(), PHYS)

    orders = {tuple(sorted(p.slopes, key=lambda c: p.slopes[c].slope)) for p in points}

    assert len(orders) == 1  # todos os compostos sobem juntos: a ordem não muda com k


def test_sensitivity_covers_the_requested_multipliers_of_the_physical_value():
    points = sensitivity(synthetic_race(), PHYS, multipliers=(0.5, 1.0, 1.5))

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
    assert tampered.full != honest.full  # e a corrida inteira, que VÊ o futuro, claro que muda


# ------------------------------------------------------------------ uma corrida inteira


def test_analyze_race_ties_the_pieces_together():
    race = synthetic_race(seed=8)

    report = fuel.analyze_race("2024 R99 Sintético", race, TOTAL_LAPS)

    assert report.physical == pytest.approx(physical_coefficient(TOTAL_LAPS))
    assert report.full.source in ("estimado", "físico")
    assert set(report.progress) == set(fuel.PROGRESS_FRACTIONS)
    assert len(report.sensitivity) == len(fuel.SENSITIVITY_MULTIPLIERS)
    assert set(report.chosen) == set(TRUE_SLOPES)
    assert report.n_laps == len(race)


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
        "## Decisão com dados",
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
    monkeypatch.setattr(fuel, "load_reports", lambda session, year=None: ([], []))
    monkeypatch.setattr(fuel, "get_session_factory", lambda: sessionmaker())

    assert fuel.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    reports = reports_for(1)
    monkeypatch.setattr(fuel, "load_reports", lambda session, year=None: (reports, []))
    monkeypatch.setattr(fuel, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "combustivel.md"

    assert fuel.main(["--markdown", str(target)]) == 0

    assert "2024 R01 Sintético" in capsys.readouterr().out
    assert "| 2024 R01 Sintético |" in target.read_text(encoding="utf-8")
