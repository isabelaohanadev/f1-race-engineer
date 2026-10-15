"""O relatório do backtest (strategy/backtest_report.py): a redação segue os números.

Resultados FABRICADOS em que o veredito certo é conhecido (o modelo melhor, igual ou pior que o
comparador), para provar que o texto diz "supera", "não supera" e "é pior" quando deve, inclusive
quando o resultado é ruim.
"""

import dataclasses
from datetime import date

import pytest

from strategy.backtest import LIVE, ORACLE, TEMPO, VOLTA, RaceOutcome
from strategy.backtest_report import (
    FREEZE_COMMIT,
    LIMITATIONS,
    build_report,
    fmt,
    num,
    pct,
    pooled_mae,
    section_apart,
    section_brier,
    section_calibration,
    section_laptime,
    section_leakage,
    section_sensitivity,
    section_summary,
    section_undercut,
    section_variants,
    verdict_summary,
    wilson,
)
from strategy.forecast_eval import Interval, LapScore, ProgressRow
from strategy.laptime_eval import LapTimeRecord
from strategy.protocol import FREEZE_BASE_COMMIT, FROZEN, Protocol

RACES = ("A", "B", "C", "D", "E")
HEADLINE = FROZEN.headline_variant
LAPS = tuple(range(3, 60, 6))  # uma volta em cada faixa de 10% de uma corrida de 60 voltas


def scores_for(race: str, i: int, model: float, leader: float, spread: float = 0.05):
    shift = spread * (i - 2)
    return [
        LapScore(
            race,
            lap,
            60,
            model + shift,
            0.5,
            0.4,
            True,
            leader + shift,
            0.9,
            ((0.8, True), (0.8, False), (0.05, False)),
            ((0.9, True),),
        )  # fmt: skip
        for lap in LAPS
    ]


def records_for(race: str, i: int, mae: float, baseline: float, spread: float = 0.02):
    shift = spread * (i - 2)
    return [
        LapTimeRecord(race, lap, h, 10, 0.01, mae + shift, 0.9, 0.0, baseline + shift, 10)
        for lap in (5, 25, 45)
        for h in FROZEN.horizons
    ]


def outcomes(
    *,
    model=0.2,
    leader=0.6,
    mae=0.1,
    baseline=0.5,
    variant_shift: dict[str, tuple[float, float]] | None = None,
    sc_shift: dict[float, float] | None = None,
    apart: bool = False,
) -> list[RaceOutcome]:
    """5 corridas secas (e, se pedido, 1 de fora) com os números dados. `variant_shift` desloca o
    Brier e o erro de tempo de volta das outras variantes; `sc_shift`, o Brier por multiplicador."""
    variant_shift = variant_shift or {}
    sc_shift = sc_shift or {}
    out = []
    for i, race in enumerate(RACES):
        o = RaceOutcome(race, "seca", True, 60)
        for variant in FROZEN.variants:
            d_brier, d_mae = variant_shift.get(variant, (0.0, 0.0))
            o.laptime[(variant, TEMPO)] = records_for(race, i, mae + d_mae, baseline)
            o.forecast[(variant, TEMPO, 1.0)] = scores_for(race, i, model + d_brier, leader)
        for m in FROZEN.sc_multipliers:
            if m != 1.0:
                o.forecast[(HEADLINE, TEMPO, m)] = scores_for(
                    race, i, model + sc_shift.get(m, 0.0), leader
                )
        o.laptime[(HEADLINE, VOLTA)] = records_for(race, i, mae * 0.5, baseline)
        o.forecast[(HEADLINE, VOLTA, 1.0)] = scores_for(race, i, model * 0.5, leader)
        for key in ((LIVE, TEMPO), (ORACLE, TEMPO), (LIVE, VOLTA)):
            o.undercut[key] = []
        o.skipped = {"rival não parou": 3}
        o.seconds = 1.0
        out.append(o)
    if apart:
        wet = RaceOutcome("W", "molhada", False, 60)
        wet.laptime[(HEADLINE, TEMPO)] = records_for("W", 2, 0.4, 0.3)
        wet.forecast[(HEADLINE, TEMPO, 1.0)] = scores_for("W", 2, 0.9, 0.7)
        wet.undercut[(LIVE, TEMPO)] = []
        out.append(wet)
    return out


# ------------------------------------------------------------------ as peças


def test_the_wilson_interval_brackets_a_proportion_and_is_wide_for_few_alerts():
    low, high = wilson(8, 10)

    assert low == pytest.approx(0.49, abs=0.01) and high == pytest.approx(0.94, abs=0.01)
    assert wilson(0, 0) is None
    assert wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0
    narrow = wilson(800, 1000)
    assert narrow[1] - narrow[0] < 0.06


def test_the_formatters_show_a_point_when_there_is_no_interval():
    assert fmt(Interval(0.5, 0.4, 0.6, 5)) == "0.500 [0.400, 0.600]"
    assert fmt(Interval(0.5, 0.5, 0.5, 1)) == "0.500" and fmt(None) == "-"
    assert pct(Interval(0.9, 0.8, 0.95, 5)) == "90% [80%, 95%]" and pct(None) == "-"
    assert num(None) == "-" and num(1.23456, 2) == "1.23"


def test_the_pooled_error_weights_each_lap_by_its_drivers():
    recs = [
        LapTimeRecord("A", 5, 1, 10, 0, 1.0, 1.0, None, None, 0),
        LapTimeRecord("A", 5, 2, 30, 0, 2.0, 1.0, None, None, 0),  # (10 x 1 + 30 x 2) / 40
        LapTimeRecord("B", 5, 1, 5, 0, 4.0, 1.0, None, None, 0),
    ]

    assert pooled_mae(recs) == {"A": pytest.approx(1.75), "B": pytest.approx(4.0)}


def interval(mean: float, low: float, high: float) -> Interval:
    return Interval(mean, low, high, 21)


def row(low: float, high: float, versus_leader: Interval | None) -> ProgressRow:
    point = Interval(0.3, 0.2, 0.4, 21)
    return ProgressRow(
        low,
        high,
        21 if versus_leader else 0,
        point,
        point,
        point,
        versus_leader,
        versus_leader,
        point,
    )


def test_the_verdict_sentence_lists_the_bins_by_beats_does_not_beat_and_is_worse():
    rows = [
        row(0.0, 0.1, interval(-0.3, -0.4, -0.2)),  # supera
        row(0.1, 0.2, interval(-0.3, -0.4, -0.2)),  # supera
        row(0.2, 0.3, interval(0.0, -0.1, 0.1)),  # não supera
        row(0.3, 0.4, interval(0.3, 0.2, 0.4)),  # é pior
    ]

    text = verdict_summary(rows, "o líder atual vence", lambda r: r.verdict_leader)

    assert "SUPERA em 2 de 4 faixas (0%-10%, 10%-20%)" in text
    assert "NÃO SUPERA em 1 de 4 faixas (20%-30%)" in text
    assert "É PIOR em 1 de 4 faixas (30%-40%)" in text
    assert "inteiro abaixo de zero" in text and "contém o zero" in text


def test_the_verdict_sentence_does_not_mention_a_category_that_does_not_occur():
    rows = [row(0.0, 0.1, interval(-0.3, -0.4, -0.2)), row(0.1, 0.2, None)]

    text = verdict_summary(rows, "todos iguais", lambda r: r.verdict_leader)

    assert "SUPERA em 1 de 1 faixas" in text
    assert "NÃO SUPERA em" not in text and "É PIOR em" not in text
    assert verdict_summary([row(0.0, 0.1, None)], "x", lambda r: r.verdict_leader).startswith(
        "Sem dados"
    )


# ------------------------------------------------------------------ o tempo de volta


def test_the_laptime_section_says_the_model_beats_the_persistence_when_it_does():
    text = section_laptime(outcomes(mae=0.1, baseline=0.5), FROZEN)

    assert "nos 5 horizontes" in text and "erra menos que a persistência" in text
    assert "| encolhimento | 1 |" in text and "| fisico | 5 |" in text


def test_the_laptime_section_says_in_so_many_words_when_the_model_does_not_beat_it():
    text = section_laptime(outcomes(mae=0.5, baseline=0.1), FROZEN)

    assert "NÃO supera a persistência em nenhum horizonte" in text
    assert "nos 5 horizontes" not in text


def test_the_laptime_section_lists_only_the_horizons_where_it_wins():
    wins_early = outcomes(mae=0.1, baseline=0.5)
    for o in wins_early:  # nos horizontes 4 e 5 o modelo passa a ser pior que a persistência
        for variant in FROZEN.variants:
            o.laptime[(variant, TEMPO)] = [
                dataclasses.replace(r, mae=1.0, baseline_mae=0.1) if r.horizon >= 4 else r
                for r in o.laptime[(variant, TEMPO)]
            ]

    text = section_laptime(wins_early, FROZEN)

    assert "só supera a persistência nos horizontes 1, 2, 3" in text


# ------------------------------------------------------------------ o Brier


def test_the_brier_section_says_it_beats_the_leader_in_every_bin_when_it_does():
    text = section_brier(outcomes(model=0.2, leader=0.6), FROZEN)

    assert "SUPERA em 10 de 10 faixas" in text and "NÃO SUPERA em" not in text
    assert "a unidade do intervalo é a **corrida**" in text
    assert text.count("| supera |") >= 10


def test_the_brier_section_says_in_so_many_words_when_it_does_not_beat_the_leader():
    ruim = section_brier(outcomes(model=0.6, leader=0.2), FROZEN)
    igual = section_brier(outcomes(model=0.4, leader=0.4), FROZEN)

    leader = "Contra **o líder atual vence**, o Monte Carlo"
    assert f"{leader} É PIOR em 10 de 10 faixas" in ruim
    assert f"{leader} SUPERA" not in ruim and f"{leader} NÃO SUPERA" not in ruim
    assert f"{leader} NÃO SUPERA em 10 de 10 faixas" in igual


def test_the_brier_section_compares_with_both_comparators_and_reports_the_overall():
    text = section_brier(outcomes(), FROZEN)

    assert "o líder atual vence" in text and "todos iguais" in text
    assert "Monte Carlo menos o líder (pareado)" in text
    assert "O favorito do Monte Carlo ganhou" in text


# ------------------------------------------------------------------ a calibração


def test_the_calibration_section_shows_the_n_of_every_bin_and_warns_about_thin_bins():
    text = section_calibration(outcomes(), FROZEN)

    assert "**Vitória**" in text and "**Pódio**" in text
    assert "pares (piloto, volta)" in text and "| corridas |" in text
    assert "Uma faixa com poucas corridas não diz nada" in text
    assert "| 0.70 a 0.90 | 100 | 5 | 0.800 |" in text  # 2 pares por volta, 10 voltas, 5 corridas


# ------------------------------------------------------------------ as variantes de k


def test_a_variant_better_in_both_measures_is_declared_better():
    text = section_variants(outcomes(variant_shift={"fisico": (-0.1, -0.05)}), FROZEN)

    assert "`fisico`: melhor que o principal" in text


def test_a_variant_worse_in_both_measures_is_declared_worse():
    text = section_variants(outcomes(variant_shift={"estimado": (0.1, 0.05)}), FROZEN)

    assert "`estimado`: pior que o principal" in text


def test_a_variant_better_in_only_one_measure_is_not_declared_anything():
    """A regra pede os dois, no mesmo sentido: melhor no Brier e pior no tempo de volta é "sem
    diferença detectável"."""
    text = section_variants(outcomes(variant_shift={"fisico": (-0.1, 0.05)}), FROZEN)

    assert "`fisico`: sem diferença detectável" in text


def test_identical_variants_are_not_distinguishable_and_the_headline_stays():
    text = section_variants(outcomes(), FROZEN)

    assert (
        "`fisico`: sem diferença detectável" in text
        and "`estimado`: sem diferença detectável" in text
    )
    assert "encolhimento (principal)" in text


# ------------------------------------------------------------------ SC, undercut, vazamento


def test_the_sensitivity_section_has_one_column_per_rate_and_a_paired_difference():
    text = section_sensitivity(outcomes(sc_shift={2.0: 0.05}), FROZEN)

    for m in FROZEN.sc_multipliers:
        assert f"taxa de SC {m:g}x" in text
    assert "taxa 2x menos 1x (Brier, pareado)" in text
    assert "taxa 0.5x menos 1x (Brier, pareado)" in text
    assert "os **mesmos números aleatórios**" in text


def test_the_undercut_section_states_the_selection_bias_before_any_number():
    text = section_undercut(outcomes(), FROZEN)

    assert text.index("ressalva do viés") < text.index("Ao vivo")
    assert "precisão é otimista" in text and "revocação não é mensurável" in text
    assert "condicional" in text
    assert text.index("Ao vivo") < text.index("Oráculo")
    assert "rival não parou: 15" in text  # 5 corridas x 3


def test_the_leakage_section_explains_the_sign_and_shows_both_states():
    text = section_leakage(outcomes(), FROZEN)

    assert "Diferença negativa = o estado antigo é mais otimista" in text
    assert "estado no instante t" in text and "estado por contagem de voltas" in text
    # o estado antigo foi fabricado com erro menor: a diferença (contagem - instante) é negativa
    assert "| Brier da vitória |" in text


def test_the_apart_section_gives_points_without_intervals_and_says_why():
    text = section_apart(outcomes(apart=True)[-1:], FROZEN)

    assert "não entram nas médias das secas" in text and "sem intervalo de confiança" in text
    assert "| W | molhada |" in text
    assert section_apart([], FROZEN).startswith("Nenhuma corrida de fora")


# ------------------------------------------------------------------ o relatório inteiro


def test_the_full_report_has_every_section_in_order_and_names_the_freeze():
    report = build_report(outcomes(apart=True), generated_on=date(2026, 1, 2))

    headings = [
        "## Valores congelados",
        "## 1. Erro de previsão do tempo de volta",
        "## 2. Probabilidade de vitória",
        "## 3. Curva de calibração",
        "## 4. As variantes de k",
        "## 5. Sensibilidade à taxa de Safety Car",
        "## 6. O alerta de undercut nas paradas reais",
        "## 7. Diagnóstico do vazamento",
        "## 8. A britânica, o Canadá e o Brasil, à parte",
        "## Limitações",
        "## Ajustes pós-resultado",
    ]
    positions = [report.index(h) for h in headings]
    assert positions == sorted(positions)
    assert f"`{FREEZE_COMMIT}`" in report and f"`{FREEZE_BASE_COMMIT}`" in report
    assert "2026-01-02" in report and "5 corridas secas" in report and "1 fora delas" in report
    assert "VERIFICAÇÃO RÁPIDA" not in report


def test_the_frozen_table_lists_every_protocol_field_with_its_value():
    report = build_report(outcomes())

    for field in dataclasses.fields(FROZEN):
        assert f"`{field.name}`" in report
    assert "| `alert_threshold` | 0.6 |" in report and "| `undercut_sd_inflation` | 1.3 |" in report


def test_the_report_prints_the_values_of_the_protocol_it_was_given():
    report = build_report(outcomes(), Protocol(alert_threshold=0.75))

    assert "| `alert_threshold` | 0.75 |" in report


def test_no_adjustment_is_said_plainly_and_adjustments_are_listed():
    plain = build_report(outcomes())
    adjusted = build_report(
        outcomes(), adjustments=("o limiar foi de 0,6 para 0,7", "k_max de 10 para 8")
    )

    assert "Nenhum. Nenhum valor do protocolo foi alterado depois de ver o resultado." in plain
    assert "- o limiar foi de 0,6 para 0,7" in adjusted and "- k_max de 10 para 8" in adjusted
    assert "Nenhum. Nenhum valor" not in adjusted


def test_a_quick_run_is_labeled_so_nobody_mistakes_it_for_the_backtest():
    assert "VERIFICAÇÃO RÁPIDA" in build_report(outcomes(), quick=True)


def test_the_limitations_are_always_in_the_report_good_result_or_bad():
    for good in (outcomes(model=0.1, leader=0.9), outcomes(model=0.9, leader=0.1)):
        report = build_report(good)

        assert LIMITATIONS in report
        assert "3 eventos" in report and "VSC" in report and "heurísticas" in report


# ------------------------------------------------------------------ o resumo


def test_the_summary_says_beats_when_the_model_beats_both_comparators():
    text = section_summary(
        [o for o in outcomes(model=0.2, leader=0.6, mae=0.1, baseline=0.5)], FROZEN
    )

    assert "supera em 5 dos 5 horizontes, não se distingue em 0 e é pior em 0" in text
    assert "o Monte Carlo supera em 10, não se distingue em 0 e é pior em 0" in text
    assert "(diferença pareada" in text and ": supera)" in text
    assert 'o veredito é "supera"' in text


def test_the_summary_says_worse_with_all_the_letters_when_the_model_is_worse():
    text = section_summary(outcomes(model=0.6, leader=0.2, mae=0.5, baseline=0.1), FROZEN)

    assert "supera em 0 dos 5 horizontes, não se distingue em 0 e é pior em 5" in text
    assert "supera em 0, não se distingue em 0 e é pior em 10" in text
    assert ": é pior)" in text


def test_the_summary_names_the_variants_that_the_rule_distinguishes():
    none = section_summary(outcomes(), FROZEN)
    one = section_summary(outcomes(variant_shift={"fisico": (-0.1, -0.05)}), FROZEN)

    assert "Nenhuma variante se distingue do encolhimento" in none
    assert "Distinguíveis do principal pela regra: fisico." in one


def test_the_summary_reports_the_sc_rate_effect_in_plain_words():
    insensitive = section_summary(outcomes(), FROZEN)
    sensitive = section_summary(outcomes(sc_shift={2.0: 0.2}), FROZEN)

    assert "taxa 0.5x: sem diferença detectável; taxa 2x: sem diferença detectável" in insensitive
    assert "taxa 2x: Brier maior" in sensitive


def test_the_summary_reads_the_leakage_sign_in_plain_words():
    flattered = section_summary(outcomes(), FROZEN)  # o estado antigo foi fabricado com Brier menor

    assert "Vazamento do estado por contagem de voltas" in flattered
    assert "o estado antigo é mais otimista" in flattered


def test_the_summary_comes_first_and_the_notes_come_last_without_touching_the_protocol():
    report = build_report(
        outcomes(), notes=("docs/backtest-diagnostico.md: o favorito do Monte Carlo",)
    )

    assert report.index("## Resumo") < report.index("## Valores congelados")
    tail = report[report.index("## Ajustes pós-resultado") :]
    assert "Nenhum valor do protocolo foi alterado" in tail
    assert "NÃO alteram o protocolo" in tail and "o favorito do Monte Carlo" in tail
    assert "NÃO alteram o protocolo" not in build_report(outcomes())
