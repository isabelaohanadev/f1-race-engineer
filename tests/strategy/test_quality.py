"""Testes da inspeção de qualidade das voltas (strategy/quality.py).

Cada verificação é exercitada com dados SINTÉTICOS em que a resposta certa é conhecida: um dado com
exatamente um defeito plantado tem de acusar exatamente um defeito, e um dado limpo, nenhum.
"""

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from strategy import quality
from strategy.quality import EntryRow, LapRow

LAP_MS = 90_000


def lap(driver: str = "VER", n: int = 1, **overrides) -> LapRow:
    """Uma volta limpa (pista livre, sem box, relógio coerente) com o defeito de `overrides`."""
    fields = {
        "driver": driver,
        "lap_number": n,
        "lap_time_ms": LAP_MS,
        "compound": "MEDIUM",
        "tyre_life": n,
        "stint": 1,
        "position": 1,
        "session_time_ms": n * LAP_MS,
        "pit_in": False,
        "pit_out": False,
        "track_status": "1",
    }
    return LapRow(**(fields | overrides))


def clean_race(driver: str = "VER", laps: int = 5) -> list[LapRow]:
    return [lap(driver, n) for n in range(1, laps + 1)]


def by_driver(rows):
    return quality._by_driver(rows)


# ------------------------------------------------------------------ cada verificação


def test_clean_data_has_no_anomalies():
    result = quality.inspect_laps("limpa", clean_race("VER") + clean_race("NOR"))

    assert set(result.anomalies.values()) == {0}


def test_lap_number_gaps_count_missing_middle_laps_but_not_retirements():
    rows = [lap("VER", n) for n in (1, 2, 3, 5)] + [lap("NOR", n) for n in (1, 2)]  # NOR abandonou

    assert quality.count_lap_number_gaps(by_driver(rows)) == 1  # só a volta 4 do VER


def test_tyre_life_jump_inside_a_stint_is_counted():
    rows = [lap(n=1, tyre_life=1), lap(n=2, tyre_life=2), lap(n=3, tyre_life=4)]  # pulou o 3

    assert quality.count_tyre_life_jumps(by_driver(rows)) == 1


def test_tyre_life_reset_at_a_new_stint_is_not_a_jump():
    rows = [lap(n=1, tyre_life=9), lap(n=2, tyre_life=10), lap(n=3, tyre_life=1, stint=2)]

    assert quality.count_tyre_life_jumps(by_driver(rows)) == 0


def test_tyre_life_across_a_missing_lap_is_not_judged():
    rows = [lap(n=1, tyre_life=1), lap(n=3, tyre_life=3)]  # a volta 2 não existe: o pulo é esperado

    assert quality.count_tyre_life_jumps(by_driver(rows)) == 0


def test_compound_change_inside_a_stint_is_counted_but_not_across_stints():
    inside = [lap(n=1), lap(n=2, compound="HARD")]
    across = [lap(n=1), lap(n=2, compound="HARD", stint=2)]

    assert quality.count_compound_changes_inside_stint(by_driver(inside)) == 1
    assert quality.count_compound_changes_inside_stint(by_driver(across)) == 0


def test_session_clock_must_grow_lap_after_lap():
    rows = [lap(n=1), lap(n=2, session_time_ms=1_000), lap(n=3)]  # a 2 volta no tempo

    assert quality.count_session_time_not_increasing(by_driver(rows)) == 1


def test_clock_mismatch_ignores_rounding_noise_but_catches_a_real_gap():
    noise = [lap(n=1), lap(n=2, session_time_ms=2 * LAP_MS + quality.CLOCK_TOLERANCE_MS)]
    real = [lap(n=1), lap(n=2, session_time_ms=2 * LAP_MS + 500)]

    assert quality.count_clock_mismatches(by_driver(noise)) == 0
    assert quality.count_clock_mismatches(by_driver(real)) == 1


def test_clock_mismatch_only_judges_consecutive_laps_with_all_three_values():
    rows = [
        lap(n=1),
        lap(n=3, session_time_ms=9_999_999),  # volta 2 ausente: não há com o que comparar
        lap(n=4, lap_time_ms=None, session_time_ms=1),  # sem tempo de volta
    ]

    assert quality.count_clock_mismatches(by_driver(rows)) == 0


def test_position_order_mismatch_counts_laps_where_position_and_arrival_disagree():
    rows = [
        lap("VER", 1, position=1, session_time_ms=100),
        lap("NOR", 1, position=2, session_time_ms=105),  # volta 1: concordam
        lap("VER", 2, position=1, session_time_ms=200),
        lap("NOR", 2, position=2, session_time_ms=190),  # volta 2: o P2 cruzou ANTES do P1
    ]

    assert quality.count_position_order_mismatches(rows) == 1


def test_slow_laps_are_measured_against_the_race_median():
    times = [90_000, 91_000, 92_000, 93_000, 200_000]  # mediana 92 s, limite 115 s
    rows = [lap(n=i + 1, lap_time_ms=t) for i, t in enumerate(times)]

    assert quality.count_slow_laps(rows) == 1


def test_slow_laps_without_any_time_is_zero():
    assert quality.count_slow_laps([lap(lap_time_ms=None)]) == 0


# ------------------------------------------------------------------ resultado oficial


def finish_line(*positions: int | None, status: str | None = "Finished") -> list[EntryRow]:
    return [EntryRow(f"D{i:02d}", p, status) for i, p in enumerate(positions, start=1)]


def test_a_complete_finishing_order_has_no_result_problems():
    rows = [lap("D01", 5, position=1), lap("D02", 5, position=2), lap("D03", 5, position=3)]

    checks = quality.result_checks(finish_line(1, 2, 3), rows)

    assert checks == {
        "sem_posicao_final": 0,
        "sem_status": 0,
        "posicao_final_repetida": 0,
        "posicao_final_com_buraco": 0,
        "final_vs_ultima_volta": 0,
    }


def test_missing_repeated_and_skipped_finishing_positions_are_each_detected():
    # D04 sem posição; 2 aparece duas vezes; a 4 não existe (buraco entre 3 e 5).
    checks = quality.result_checks(finish_line(1, 2, 2, None, 3, 5), [])

    assert checks["sem_posicao_final"] == 1
    assert checks["posicao_final_repetida"] == 1
    assert checks["posicao_final_com_buraco"] == 1  # 5 posições distintas até 5: falta uma


def test_a_complete_order_without_status_is_the_signature_of_a_calculated_result():
    """Ordem 1..N perfeita, mas sem status: foi calculada das voltas, não é a oficial."""
    rows = [lap("D01", 5, position=1), lap("D02", 5, position=2)]

    checks = quality.result_checks(finish_line(1, 2, status=None), rows)

    assert checks["sem_status"] == 2
    assert (checks["sem_posicao_final"], checks["posicao_final_com_buraco"]) == (0, 0)


def test_a_retirement_is_a_difference_from_the_last_lap_not_a_defect():
    """Quem abandona na volta 12 com a posição 7 é classificado em 20º: o resultado oficial e a
    última volta discordam, o que é esperado e por isso só é MEDIDO."""
    rows = [lap("D01", 5, position=1), lap("D02", 12, position=7)]

    checks = quality.result_checks(finish_line(1, 20), rows)

    assert checks["final_vs_ultima_volta"] == 1
    assert checks["sem_posicao_final"] == 0


def test_result_checks_ignore_drivers_without_laps_or_last_lap_position():
    rows = [lap("D01", 5, position=None)]  # a última volta não tem posição

    checks = quality.result_checks(finish_line(1, 2), rows)  # D02 nem tem voltas

    assert checks["final_vs_ultima_volta"] == 0


def test_result_checks_without_any_entries_do_not_crash():
    assert set(quality.result_checks([], []).values()) == {0}


def test_inspect_laps_includes_the_result_checks_in_the_report():
    rows = [lap("D01", 5, position=1)]

    q = quality.inspect_laps("x", rows, entries=finish_line(None))

    assert q.results["sem_posicao_final"] == 1
    header = quality.anomaly_table([q])[0]
    assert "sem_posicao_final" in header and "relogio_nao_cresce" in header


# ------------------------------------------------------------------ o relatório de uma corrida


def test_inspect_laps_counts_the_pit_lap_trap_missing_values_and_statuses():
    rows = [
        lap(n=1, pit_out=True, lap_time_ms=None),  # saída dos boxes sem tempo
        lap(n=2, pit_in=True),  # ENTRADA COM tempo de volta: a armadilha
        lap(n=3, pit_out=True),  # SAÍDA COM tempo de volta: a armadilha
        lap(n=4, track_status="124"),  # amarela + SC
        lap(n=5, track_status="671"),  # VSC + fim do VSC + livre
        lap(n=6, track_status="5"),  # bandeira vermelha
        lap(n=7, track_status=None, pit_in=None, pit_out=None, compound=None),  # dado antigo
    ]

    q = quality.inspect_laps("x", rows, total_laps=70)

    assert (q.pit_in_laps, q.pit_in_with_time) == (1, 1)
    assert (q.pit_out_laps, q.pit_out_with_time) == (2, 1)
    assert q.status_laps == {"2": 1, "4": 1, "5": 1, "6": 1, "7": 1}
    assert q.green_laps == 3  # voltas 1, 2 e 3 (status exatamente "1")
    assert q.missing["track_status"] == 1 and q.missing["pit_in"] == 1
    assert q.missing["lap_time_ms"] == 1 and q.missing["compound"] == 1
    assert (q.laps, q.drivers, q.max_lap, q.total_laps) == (7, 1, 7, 70)


def test_dry_share_counts_only_slick_compounds():
    rows = [
        lap(n=1, compound="SOFT"),
        lap(n=2, compound="HARD"),
        lap(n=3, compound="INTERMEDIATE"),
        lap(n=4, compound="WET"),
    ]

    q = quality.inspect_laps("x", rows)

    assert q.dry_share == 0.5
    assert q.compound_laps == {"SOFT": 1, "HARD": 1, "INTERMEDIATE": 1, "WET": 1}


def test_inspect_laps_of_an_empty_race_does_not_divide_by_zero():
    q = quality.inspect_laps("vazia", [])

    assert (q.laps, q.drivers, q.max_lap, q.dry_share) == (0, 0, 0, 0.0)
    assert quality.composition_table([q])[1][4] == "-"  # sem voltas, sem percentual


# ------------------------------------------------------------------ relatório (texto e markdown)


def test_report_tables_have_one_row_per_race_plus_the_header():
    qualities = [
        quality.inspect_laps("A", clean_race()),
        quality.inspect_laps("B", clean_race(), has_positions=True),
    ]

    comp = quality.composition_table(qualities)
    anom = quality.anomaly_table(qualities)

    assert [row[0] for row in comp] == ["corrida", "A", "B"]
    assert [row[0] for row in anom] == ["corrida", "A", "B"]
    assert [row[3] for row in comp[1:]] == ["não", "sim"]
    assert all(len(row) == len(comp[0]) for row in comp)


def test_markdown_table_has_header_separator_and_aligned_columns():
    text = quality.render_markdown([["a", "b"], ["1", "2"]])

    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_markdown_report_states_what_it_was_generated_from_and_explains_the_columns():
    from datetime import date

    qualities = [quality.inspect_laps("A", clean_race()), quality.inspect_laps("B", clean_race())]

    text = quality.build_report(qualities, markdown=True, generated_on=date(2026, 10, 14))

    assert "em 2026-10-14, sobre 2 corridas" in text
    assert "python -m strategy.quality --markdown" in text
    for column in ("box entrada / saída", "sem_status", "relogio_vs_tempo_de_volta"):
        assert column in text  # toda coluna que o relatório mostra tem de estar explicada


def test_text_table_has_a_separator_under_the_header():
    lines = quality.render_text([["corrida", "n"], ["A", "10"]]).splitlines()

    assert set(lines[1].replace(" ", "")) == {"-"}
    assert len(lines) == 3


# ------------------------------------------------------------------ leitura do banco (SQLite)


def test_load_qualities_reads_every_race_in_order_and_flags_old_laps_as_unknown(seeded: Engine):
    with sessionmaker(seeded)() as session:
        qualities = quality.load_qualities(session)

    assert [q.label for q in qualities] == ["2023 R01 Bahrain", "2024 R21 São Paulo"]
    brazil = qualities[1]
    assert (brazil.laps, brazil.drivers, brazil.total_laps) == (5, 2, 69)
    # As voltas semeadas não têm as colunas da Fase 3: "não sei", e o relatório diz isso.
    assert brazil.missing["pit_in"] == 5 and brazil.missing["track_status"] == 5
    assert brazil.has_positions is False


def test_load_qualities_can_filter_by_season(seeded: Engine):
    with Session(seeded) as session:
        only_2024 = quality.load_qualities(session, year=2024)

    assert [q.label for q in only_2024] == ["2024 R21 São Paulo"]


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(quality, "load_qualities", lambda session, year=None: [])
    monkeypatch.setattr(quality, "get_session_factory", lambda: sessionmaker())

    code = quality.main([])

    assert code == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_the_tables_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    qualities = [quality.inspect_laps("2024 R10 Spanish", clean_race(), total_laps=5)]
    monkeypatch.setattr(quality, "load_qualities", lambda session, year=None: qualities)
    monkeypatch.setattr(quality, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "qualidade.md"

    code = quality.main(["--markdown", str(target)])

    assert code == 0
    assert "2024 R10 Spanish" in capsys.readouterr().out
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Qualidade dos dados de voltas")
    assert "## Como ler" in text and "## Composição e cobertura" in text
    assert "| 2024 R10 Spanish |" in text
