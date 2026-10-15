"""Testes da "verdade" do backtest (strategy/results.py): a ordem na pista e o oficial ao lado."""

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from strategy import results
from strategy.laps import EntryRow, LapRow
from strategy.results import compare_results, official_order, on_track_order


def lap(driver: str, n: int, session_ms: int | None) -> LapRow:
    return LapRow(driver, n, 90_000, "MEDIUM", n, 1, 1, session_ms, False, False, "1")


def finish(*drivers: str, laps: int = 5, gap_ms: int = 1_000) -> list[LapRow]:
    """Todos terminam `laps` voltas, na ordem dada (cada um `gap_ms` depois do anterior)."""
    return [lap(d, laps, 450_000 + i * gap_ms) for i, d in enumerate(drivers)]


def entry(driver: str, position: int | None, status: str | None = "Finished") -> EntryRow:
    return EntryRow(driver, position, status)


# ------------------------------------------------------------------ a ordem na pista


def test_who_crossed_the_line_first_wins_when_the_laps_are_equal():
    assert on_track_order(finish("NOR", "VER", "LEC")) == ["NOR", "VER", "LEC"]


def test_more_laps_beat_an_earlier_crossing():
    """O lapped cruza a linha ANTES do vencedor na última volta dele, mas tem menos voltas."""
    rows = [lap("VER", 5, 450_000), lap("HAM", 4, 360_000)]

    assert on_track_order(rows) == ["VER", "HAM"]


def test_a_retirement_is_ranked_by_the_laps_it_completed():
    rows = finish("VER", "NOR") + [lap("ALB", 2, 180_000)]  # abandonou na volta 2

    assert on_track_order(rows) == ["VER", "NOR", "ALB"]


def test_only_the_last_lap_with_a_clock_counts_for_each_driver():
    rows = [
        lap("VER", 1, 90_000),
        lap("VER", 2, 180_000),
        lap("VER", 3, None),
        lap("NOR", 2, 181_000),
    ]

    assert on_track_order(rows) == ["VER", "NOR"]  # VER: volta 2 (a 3 não tem relógio)


def test_a_driver_without_any_clock_is_left_out_of_the_on_track_order():
    rows = finish("VER", "NOR") + [lap("ALB", 1, None)]

    assert on_track_order(rows) == ["VER", "NOR"]


def test_a_tie_on_laps_and_time_is_broken_by_driver_code_deterministically():
    rows = [lap("NOR", 5, 450_000), lap("HAM", 5, 450_000)]

    assert on_track_order(rows) == ["HAM", "NOR"]


def test_on_track_order_of_no_laps_is_empty():
    assert on_track_order([]) == []


# ------------------------------------------------------------------ a ordem oficial


def test_the_official_order_follows_finish_position_and_skips_unclassified_drivers():
    entries = [entry("NOR", 2), entry("VER", 1), entry("ALB", None, "Did not start")]

    assert official_order(entries) == ["VER", "NOR"]


# ------------------------------------------------------------------ a comparação


def test_when_track_and_official_agree_nothing_differs():
    rows = finish("VER", "NOR", "LEC")
    entries = [entry("VER", 1), entry("NOR", 2), entry("LEC", 3)]

    c = compare_results("x", rows, entries)

    assert (c.winner_on_track, c.winner_official) == ("VER", "VER")
    assert c.podium_same and c.positions_differ == 0 and c.compared == 3
    assert c.notes == []


def test_a_post_race_disqualification_of_the_winner_shifts_everyone():
    """O GP da Bélgica 2024, em miniatura: RUS vence na pista e é desqualificado depois."""
    rows = finish("RUS", "HAM", "PIA", "LEC")
    entries = [entry("HAM", 1), entry("PIA", 2), entry("LEC", 3), entry("RUS", 4, "Disqualified")]

    c = compare_results("Bélgica", rows, entries)

    assert (c.winner_on_track, c.winner_official) == ("RUS", "HAM")
    assert not c.podium_same
    assert c.positions_differ == 4  # os 4 mudam de posição
    assert c.notes == ["RUS: Disqualified"]


def test_a_post_race_time_penalty_swapping_two_drivers_differs_in_exactly_two():
    rows = finish("VER", "NOR", "LEC", "PIA")
    entries = [entry("VER", 1), entry("LEC", 2), entry("NOR", 3), entry("PIA", 4)]  # NOR levou +5 s

    c = compare_results("x", rows, entries)

    assert c.winner_on_track == c.winner_official == "VER"
    assert not c.podium_same and c.positions_differ == 2


def test_a_driver_who_did_not_start_does_not_make_everyone_else_differ():
    """Quem não largou só existe no oficial (no fim): a comparação usa quem está nas duas."""
    rows = finish("VER", "NOR")
    entries = [entry("VER", 1), entry("NOR", 2), entry("ALB", 3, "Did not start")]

    c = compare_results("x", rows, entries)

    assert c.positions_differ == 0 and c.compared == 2
    assert c.notes == ["ALB: Did not start"]


def test_routine_statuses_are_not_listed_as_notes():
    rows = finish("VER", "NOR", "HAM")
    entries = [entry("VER", 1, "Finished"), entry("NOR", 2, "Lapped"), entry("HAM", 3, "Retired")]

    assert compare_results("x", rows, entries).notes == []


def test_a_race_with_no_data_has_no_winner_and_does_not_crash():
    c = compare_results("vazia", [], [])

    assert (c.winner_on_track, c.winner_official) == (None, None)
    assert c.podium_same and c.positions_differ == 0 and c.compared == 0


# ------------------------------------------------------------------ relatório e CLI


def two_races():
    agree = compare_results(
        "A", finish("VER", "NOR", "LEC"), [entry("VER", 1), entry("NOR", 2), entry("LEC", 3)]
    )
    differ = compare_results(
        "B",
        finish("RUS", "HAM", "PIA"),
        [entry("HAM", 1), entry("PIA", 2), entry("RUS", 3, "Disqualified")],
    )
    return [agree, differ]


def test_the_table_has_one_row_per_race_and_flags_the_podium_that_differs():
    table = results.results_table(two_races())

    assert [row[0] for row in table] == ["corrida", "A", "B"]
    assert table[1][3] == "sim" and table[2][3] == "NÃO"
    assert table[2][5] == "RUS: Disqualified"


def test_the_summary_is_computed_from_the_comparisons():
    text = results.build_report(two_races(), markdown=False)

    assert "O vencedor é o mesmo em 1 de 2 corridas (difere em: B)." in text
    assert "O pódio (três pilotos, na ordem) é o mesmo em 1 de 2 (difere em: B)." in text


def test_the_markdown_report_states_the_decision_and_the_belgian_case():
    from datetime import date

    text = results.build_report(two_races(), markdown=True, generated_on=date(2026, 10, 14))

    assert text.startswith("# A verdade do backtest")
    assert "ordem na pista" in text and "resultado\noficial é reportado ao lado" in text
    assert "Bélgica 2024" in text and "em 2026-10-14, sobre 2 corridas" in text


def test_load_comparisons_reads_races_in_order(seeded: Engine):
    with sessionmaker(seeded)() as session:
        comparisons = results.load_comparisons(session)

    assert [c.label for c in comparisons] == ["2023 R01 Bahrain", "2024 R21 São Paulo"]
    brazil = comparisons[1]
    assert brazil.winner_official == "VER"  # o oficial vem de race_entries
    assert brazil.on_track == []  # as voltas semeadas não têm relógio da sessão


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(results, "load_comparisons", lambda session, year=None: [])
    monkeypatch.setattr(results, "get_session_factory", lambda: sessionmaker())

    assert results.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    comparisons = two_races()
    monkeypatch.setattr(results, "load_comparisons", lambda session, year=None: comparisons)
    monkeypatch.setattr(results, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "verdade.md"

    assert results.main(["--markdown", str(target)]) == 0

    assert "ORDEM NA PISTA x RESULTADO OFICIAL" in capsys.readouterr().out
    assert "| B |" in target.read_text(encoding="utf-8")
