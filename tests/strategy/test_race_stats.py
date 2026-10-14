"""Testes das estatísticas por corrida e do relatório das constantes (strategy/race_stats.py)."""

from dataclasses import replace
from datetime import date

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from strategy import race_stats
from strategy.population import RaceKind, RaceRef
from strategy.priors import RaceStats, leave_one_out
from strategy.race_stats import compute_race_stats
from tests.strategy.synthetic import Plan, World, build_race


def synthetic_race():
    plans = [
        Plan(
            f"D{i:02d}",
            base=90 + 0.1 * i,
            compound="MEDIUM" if i % 2 else "SOFT",
            stops=((10 + i, "HARD"),),
        )
        for i in range(18)
    ]
    return build_race(plans, World(total_laps=50, noise=0.1, seed=1))


REF = RaceRef(1, "2024 R99 Sintético", 50, RaceKind.SECA, 1.0, 2024, 99, "Sakhir")


def test_a_race_reports_its_own_pit_loss_and_the_number_of_green_stops():
    stats = compute_race_stats(REF, synthetic_race())

    assert stats.label == "2024 R99 Sintético" and stats.circuit == "Sakhir"
    assert (stats.year, stats.round) == (2024, 99)
    assert stats.pit_loss == pytest.approx(21.0, abs=1.0)  # 12 + 9 plantados, mais termos de pneu
    assert stats.n_green_stops == len(stats.green_losses) == 18
    assert stats.other_losses == ()


def test_a_race_with_too_few_green_stops_has_no_loss_of_its_own():
    plans = [Plan("D00", stops=((20, "HARD"),)), Plan("D01", base=90.5)]

    stats = compute_race_stats(REF, build_race(plans, World(total_laps=50, seed=2)))

    assert stats.pit_loss is None and stats.pit_loss_sd is None and stats.n_green_stops == 1


def test_stops_under_safety_car_are_recorded_apart_from_the_green_ones():
    plans = [Plan(f"D{i:02d}", base=90 + 0.1 * i, stops=((10 + 2 * i, "HARD"),)) for i in range(10)]
    world = World(total_laps=50, seed=3, status_laps={20: "4", 21: "4"})  # D05 para na 20

    stats = compute_race_stats(REF, build_race(plans, world))

    assert len(stats.other_losses) >= 1
    assert stats.n_green_stops == len(stats.green_losses) < 10


def test_between_compounds_only_counts_compounds_with_enough_units():
    class Prior:
        def __init__(self, mean, n):
            self.mean, self.n_units = mean, n

    class Model:
        compounds = {"HARD": Prior(0.04, 10), "MEDIUM": Prior(0.07, 12), "SOFT": Prior(0.5, 1)}

    assert race_stats._between_compounds(Model) == pytest.approx(0.0212, abs=1e-3)  # sem o SOFT
    Model.compounds = {"HARD": Prior(0.04, 10), "SOFT": Prior(0.5, 2)}
    assert race_stats._between_compounds(Model) is None  # só um composto com unidades suficientes


# ------------------------------------------------------------------ relatório


def stats_list() -> list[RaceStats]:
    return [
        RaceStats(
            f"R{i}", f"C{i}", 2024, i, 0.02, 0.015, 20.0 + 2 * (i - 1), 1.4, 10,
            green_losses=(20.0 + 2 * (i - 1),) * 10, other_losses=(30.0,) if i == 1 else (),
        )
        for i in range(1, 6)
    ]  # fmt: skip


def test_the_table_shows_the_reserve_computed_without_each_race():
    stats = stats_list()

    table = race_stats.stats_table(stats)

    assert table[0][7] == "reserva da perda SEM esta corrida (s)"
    for row, s in zip(table[1:], stats, strict=True):
        assert float(row[7]) == pytest.approx(leave_one_out(stats, s.label).pit_loss.mean, abs=0.05)


def test_the_reserve_of_a_race_does_not_change_when_the_race_itself_changes():
    stats = stats_list()
    before = race_stats.stats_table(stats)[3][7]
    stats[2] = replace(stats[2], pit_loss=999.0)

    assert race_stats.stats_table(stats)[3][7] == before


def test_the_summary_is_computed_from_the_stops():
    text = race_stats.build_report(stats_list(), markdown=False)

    assert "5 corridas; 50 paradas em bandeira verde" in text
    assert "mediana 30.0 s contra 24.0 s das verdes" in text  # 1 parada de 30 contra as verdes
    assert "Valores de `DEFAULT_PRIORS`" in text


def test_the_markdown_report_explains_the_green_rule_and_the_missing_circuit_history():
    text = race_stats.build_report(stats_list(), markdown=True, generated_on=date(2026, 10, 14))

    assert text.startswith("# Perda no pit e constantes por corrida")
    assert "em 2026-10-14" in text
    assert "Paradas verdes" in text and "não tem o que devolver" in text
    assert "relativa ao pelotão" in text  # a ressalva sobre a safety car


def test_load_race_stats_has_nothing_for_a_wet_race(seeded: Engine):
    with sessionmaker(seeded)() as session:
        assert race_stats.load_race_stats(session, year=2024) == []


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(race_stats, "load_race_stats", lambda session, year=None: [])
    monkeypatch.setattr(race_stats, "get_session_factory", lambda: sessionmaker())

    assert race_stats.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(race_stats, "load_race_stats", lambda session, year=None: stats_list())
    monkeypatch.setattr(race_stats, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "perda.md"

    assert race_stats.main(["--markdown", str(target)]) == 0

    assert "R3" in capsys.readouterr().out
    assert "| R3 |" in target.read_text(encoding="utf-8")


def test_a_race_reports_the_compound_offsets_measured_inside_the_drivers():
    stats = compute_race_stats(REF, synthetic_race())

    assert stats.offsets  # há pilotos suficientes com SOFT/MEDIUM e HARD
    assert all(isinstance(x, float) for x in stats.offsets)


def test_the_summary_reports_the_between_race_spread_of_the_offsets():
    stats = [replace(s, offsets=(0.5, -0.5, 0.1)) for s in stats_list()]

    text = race_stats.build_report(stats, markdown=False)

    assert "15 diferenças" in text and "desvio ENTRE corridas" in text
