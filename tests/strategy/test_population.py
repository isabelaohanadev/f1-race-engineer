"""Testes da população do backtest (strategy/population.py): seca entra, o resto fica à parte."""

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from strategy import population
from strategy.laps import LapRow
from strategy.population import RaceKind, RaceRef, classify_race, dry_race_share, split_population


def lap(n: int, compound: str | None) -> LapRow:
    return LapRow("VER", n, 90_000, compound, n, 1, 1, n * 90_000, False, False, "1")


def race(dry: int, wet: int) -> list[LapRow]:
    return [lap(n, "MEDIUM") for n in range(dry)] + [
        lap(100 + n, "INTERMEDIATE") for n in range(wet)
    ]


def test_dry_race_share_counts_only_slick_compounds_among_known_ones():
    rows = race(7, 3) + [lap(900, None)]  # a volta sem composto não entra na conta

    assert dry_race_share(rows) == pytest.approx(0.7)
    assert dry_race_share([]) == 0.0


@pytest.mark.parametrize(
    ("dry", "wet", "kind"),
    [
        (100, 0, RaceKind.SECA),
        (99, 1, RaceKind.SECA),  # exatamente 99%: entra
        (98, 2, RaceKind.MISTA),  # logo abaixo: sai do principal
        (76, 24, RaceKind.MISTA),  # o GP da Grã-Bretanha de 2024
        (50, 50, RaceKind.MISTA),  # exatamente a metade: ainda mista
        (49, 51, RaceKind.MOLHADA),
        (0, 100, RaceKind.MOLHADA),  # o GP do Brasil de 2024
    ],
)
def test_races_are_classified_by_how_dry_they_were(dry, wet, kind):
    assert classify_race(race(dry, wet)) is kind


def test_only_dry_races_are_in_the_main_backtest():
    dry = RaceRef(1, "A", 50, RaceKind.SECA, 1.0)
    mixed = RaceRef(2, "B", 50, RaceKind.MISTA, 0.76)
    wet = RaceRef(3, "C", 50, RaceKind.MOLHADA, 0.0)

    assert [r.is_main for r in (dry, mixed, wet)] == [True, False, False]
    assert dry.reason == ""
    assert mixed.reason == "pista mista (76% seca)"
    assert wet.reason == "pista molhada (0% seca)"


def test_split_population_separates_the_wet_race_from_the_main_ones(seeded: Engine):
    with sessionmaker(seeded)() as session:
        main, separate = split_population(session)

    # 2023/1 (1 volta SOFT) é seca; 2024/21 (INTERMEDIATE e WET) é molhada.
    assert [r.label for r in main] == ["2023 R01 Bahrain"]
    assert [r.label for r in separate] == ["2024 R21 São Paulo"]
    assert separate[0].reason == "pista molhada (0% seca)"


def test_the_report_lists_both_groups_apart():
    main = [RaceRef(1, "2024 R10 Spanish", 66, RaceKind.SECA, 1.0)]
    separate = [RaceRef(2, "2024 R12 British", 52, RaceKind.MISTA, 0.76)]

    text = population.build_report(main, separate)

    assert "BACKTEST PRINCIPAL (1 corridas secas)" in text and "2024 R10 Spanish" in text
    assert "REPORTADAS À PARTE (1), fora da nota principal" in text
    assert "2024 R12 British" in text and "pista mista (76% seca)" in text


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(population, "split_population", lambda session, year=None: ([], []))
    monkeypatch.setattr(population, "get_session_factory", lambda: sessionmaker())

    assert population.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_the_split(monkeypatch, capsys):
    refs = ([RaceRef(1, "A", 50, RaceKind.SECA, 1.0)], [RaceRef(2, "B", 50, RaceKind.MISTA, 0.8)])
    monkeypatch.setattr(population, "split_population", lambda session, year=None: refs)
    monkeypatch.setattr(population, "get_session_factory", lambda: sessionmaker())

    assert population.main([]) == 0
    assert "REPORTADAS À PARTE" in capsys.readouterr().out
