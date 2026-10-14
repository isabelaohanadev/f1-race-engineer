"""Testes da limpeza das voltas (strategy/cleaning.py).

Cada regra é exercitada com UMA volta com exatamente aquele defeito, e a ordem das regras (que
decide em qual coluna do funil a volta cai) com voltas que têm vários defeitos de uma vez.
"""

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from strategy import cleaning
from strategy.cleaning import MAX_LOSS_SHARE, Reason, build_funnel, clean_laps, exclusion_reason
from strategy.laps import LapRow


def lap(**overrides) -> LapRow:
    """Uma volta que ENTRA no modelo; `overrides` planta o defeito."""
    fields = {
        "driver": "VER",
        "lap_number": 5,
        "lap_time_ms": 90_000,
        "compound": "MEDIUM",
        "tyre_life": 5,
        "stint": 1,
        "position": 1,
        "session_time_ms": 450_000,
        "pit_in": False,
        "pit_out": False,
        "track_status": "1",
    }
    return LapRow(**(fields | overrides))


# ------------------------------------------------------------------ uma regra por vez


def test_a_clean_green_lap_is_kept():
    assert exclusion_reason(lap()) is None


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ({"pit_in": None}, Reason.DESCONHECIDO),
        ({"pit_out": None}, Reason.DESCONHECIDO),
        ({"track_status": None}, Reason.DESCONHECIDO),
        ({"lap_time_ms": None}, Reason.SEM_TEMPO),
        ({"lap_number": 1}, Reason.VOLTA_1),
        ({"pit_in": True}, Reason.BOX),
        ({"pit_out": True}, Reason.BOX),
        ({"compound": None}, Reason.SEM_PNEU),
        ({"tyre_life": None}, Reason.SEM_PNEU),
        ({"stint": None}, Reason.SEM_PNEU),
    ],
)
def test_each_rule_catches_its_defect(defect, reason):
    assert exclusion_reason(lap(**defect)) is reason


def test_a_pit_lap_that_HAS_a_lap_time_is_still_removed():
    """A armadilha medida nas corridas secas: 97% das voltas de box têm tempo de volta."""
    assert exclusion_reason(lap(pit_in=True, lap_time_ms=124_967)) is Reason.BOX


# ------------------------------------------------------------------ o status: exatamente "1"


@pytest.mark.parametrize("status", ["12", "21", "2", "212"])
def test_yellow_only_status_is_removed_and_labelled_as_yellow(status):
    assert exclusion_reason(lap(track_status=status)) is Reason.SO_AMARELA


@pytest.mark.parametrize("status", ["26", "124", "4", "5", "6", "7", "671", "45", "2671"])
def test_any_other_status_is_removed_and_not_labelled_as_yellow(status):
    """ "26" (amarela + VSC) NÃO é "só amarela", mesmo contendo um 2."""
    assert exclusion_reason(lap(track_status=status)) is Reason.PISTA_NAO_VERDE


@pytest.mark.parametrize("status", ["9", "19", "3", "8", "0", "X"])
def test_a_status_code_nobody_listed_is_removed_too(status):
    """A regra é "exatamente 1", não "sem os códigos ruins que conheço": um código novo da F1 não
    entra no modelo por distração."""
    assert exclusion_reason(lap(track_status=status)) is not None


def test_only_the_exact_string_one_is_green():
    assert exclusion_reason(lap(track_status="1")) is None
    for almost in ("11", "01", " 1", "1 "):
        assert exclusion_reason(lap(track_status=almost)) is not None


# ------------------------------------------------------------------ a ordem das regras


def test_unknown_data_wins_over_every_other_rule():
    everything_wrong = lap(pit_in=None, lap_time_ms=None, lap_number=1, track_status="4")

    assert exclusion_reason(everything_wrong) is Reason.DESCONHECIDO


def test_a_lap_with_no_time_is_counted_as_no_time_even_on_lap_one():
    assert exclusion_reason(lap(lap_number=1, lap_time_ms=None)) is Reason.SEM_TEMPO


def test_the_yellow_cost_is_marginal_a_yellow_lap_one_is_charged_to_lap_one():
    assert exclusion_reason(lap(lap_number=1, track_status="12")) is Reason.VOLTA_1
    assert exclusion_reason(lap(pit_out=True, track_status="2")) is Reason.BOX
    assert exclusion_reason(lap(track_status="4")) is Reason.PISTA_NAO_VERDE


def test_clean_laps_keeps_only_the_laps_that_pass_every_rule():
    rows = [lap(lap_number=n) for n in range(2, 6)] + [lap(lap_number=9, track_status="12")]

    assert [row.lap_number for row in clean_laps(rows)] == [2, 3, 4, 5]


# ------------------------------------------------------------------ o funil


def mixed_race() -> list[LapRow]:
    """10 voltas, uma de cada defeito e 3 boas: a resposta certa é conhecida volta a volta."""
    return [
        lap(lap_number=1),  # volta_1
        lap(lap_number=2, lap_time_ms=None),  # sem_tempo
        lap(lap_number=3, pit_in=True),  # box
        lap(lap_number=4, pit_out=True),  # box
        lap(lap_number=5, track_status="4"),  # pista_nao_verde
        lap(lap_number=6, track_status="12"),  # so_amarela
        lap(lap_number=7, pit_in=None),  # desconhecido
        lap(lap_number=8),
        lap(lap_number=9),
        lap(lap_number=10),
    ]


def test_funnel_counts_every_rule_and_the_columns_add_up_to_the_total():
    funnel = build_funnel("x", mixed_race())

    assert funnel.removed == {
        Reason.DESCONHECIDO: 1,
        Reason.SEM_TEMPO: 1,
        Reason.VOLTA_1: 1,
        Reason.BOX: 2,
        Reason.SEM_PNEU: 0,
        Reason.PISTA_NAO_VERDE: 1,
        Reason.SO_AMARELA: 1,
    }
    assert funnel.kept == 3
    assert sum(funnel.removed.values()) + funnel.kept == funnel.total == 10


def test_funnel_keeps_the_yellow_cost_apart_from_the_others():
    funnel = build_funnel("x", mixed_race())

    assert funnel.loss_share == pytest.approx(0.7)  # 7 de 10 saem
    # Mantendo a amarela, a volta 6 ficaria: só ELA é o custo da amarela.
    assert funnel.loss_share_if_yellow_kept == pytest.approx(0.6)


def test_funnel_of_an_empty_race_does_not_divide_by_zero():
    funnel = build_funnel("vazia", [])

    assert (funnel.loss_share, funnel.loss_share_if_yellow_kept, funnel.dry_share_of_kept) == (
        0.0,
        0.0,
        0.0,
    )
    assert funnel.warnings == ["nenhuma volta útil"]


# ------------------------------------------------------------------ os alertas


def race_with_loss(lost: int, total: int = 10) -> list[LapRow]:
    """`total` voltas das quais exatamente `lost` saem (por volta 1, sem depender de status)."""
    return [lap(lap_number=1) for _ in range(lost)] + [lap() for _ in range(total - lost)]


def test_a_race_that_loses_more_than_forty_percent_is_flagged():
    funnel = build_funnel("x", race_with_loss(5, 10))  # 50%

    assert any("perde 50%" in w for w in funnel.warnings)


def test_a_race_that_loses_exactly_forty_percent_is_not_flagged():
    """ "Mais de 40%": o limite em si não alerta."""
    funnel = build_funnel("x", race_with_loss(4, 10))

    assert funnel.loss_share == pytest.approx(MAX_LOSS_SHARE)
    assert funnel.warnings == []


def test_a_wet_race_is_flagged_as_having_no_dry_laps():
    wet = [lap(compound="INTERMEDIATE") for _ in range(8)] + [lap(compound="WET") for _ in range(2)]

    assert build_funnel("x", wet).warnings == ["sem pista seca"]


def test_a_mixed_race_is_flagged_and_says_how_wet_it_was():
    mixed = [lap() for _ in range(8)] + [lap(compound="INTERMEDIATE") for _ in range(2)]

    assert build_funnel("x", mixed).warnings == ["pista mista (20% molhado)"]


def test_a_fully_dry_race_has_no_warning():
    assert build_funnel("x", [lap(compound=c) for c in ("SOFT", "MEDIUM", "HARD")]).warnings == []


# ------------------------------------------------------------------ o relatório


def test_funnel_table_has_one_row_per_race_and_one_column_per_rule():
    funnels = [build_funnel("A", mixed_race()), build_funnel("B", race_with_loss(5))]

    table = cleaning.funnel_table(funnels)

    assert table[0][2 : 2 + len(cleaning.FUNNEL_ORDER)] == [r.value for r in cleaning.FUNNEL_ORDER]
    assert [row[0] for row in table] == ["corrida", "A", "B"]
    assert all(len(row) == len(table[0]) for row in table)
    assert "perde 70%" in table[1][-1] and "perde 50%" in table[2][-1]  # as duas passam de 40%


def test_summary_numbers_are_computed_from_the_funnels():
    text = cleaning.build_report([build_funnel("A", mixed_race())], markdown=False)

    assert "Ao todo: 10 voltas, 3 úteis (30%)" in text
    assert "so_amarela 1" in text and "A amarela, sozinha, custa 1 voltas (10% do total)" in text
    assert "Corridas que perdem mais de 40%: A." in text


def test_markdown_report_states_its_origin_and_explains_every_column():
    from datetime import date

    text = cleaning.build_report(
        [build_funnel("A", mixed_race())], markdown=True, generated_on=date(2026, 10, 14)
    )

    assert text.startswith("# Limpeza das voltas")
    assert "em 2026-10-14, sobre 1 corridas" in text
    for reason in cleaning.FUNNEL_ORDER:
        assert f"**{reason.value}**" in text  # toda coluna do funil está explicada
    assert "exatamente `1`" in text


def test_load_funnels_treats_laps_saved_before_the_new_columns_as_unknown(seeded: Engine):
    with sessionmaker(seeded)() as session:
        funnels = cleaning.load_funnels(session, year=2024)

    (brazil,) = funnels
    assert brazil.label == "2024 R21 São Paulo"
    assert brazil.removed[Reason.DESCONHECIDO] == brazil.total == 5  # nenhuma tem pit/status
    assert brazil.kept == 0 and "nenhuma volta útil" in brazil.warnings


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(cleaning, "load_funnels", lambda session, year=None: [])
    monkeypatch.setattr(cleaning, "get_session_factory", lambda: sessionmaker())

    assert cleaning.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_the_funnel_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    funnels = [build_funnel("2024 R10 Spanish", mixed_race())]
    monkeypatch.setattr(cleaning, "load_funnels", lambda session, year=None: funnels)
    monkeypatch.setattr(cleaning, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "limpeza.md"

    assert cleaning.main(["--markdown", str(target)]) == 0

    assert "2024 R10 Spanish" in capsys.readouterr().out
    assert "| 2024 R10 Spanish |" in target.read_text(encoding="utf-8")
