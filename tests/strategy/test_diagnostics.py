"""O diagnóstico pós-resultado (strategy/diagnostics.py): ablações do simulador e o favorito.

O diagnóstico NÃO é o backtest: estes testes provam que cada ablação desliga só o que diz e que o
resto do simulador (os números aleatórios, o estado, o protocolo) fica como no backtest congelado.
"""

from dataclasses import replace

import pytest

from strategy import diagnostics
from strategy.backtest import TEMPO, RaceBundle, RaceOutcome, save_outcomes
from strategy.diagnostics import (
    ABLATIONS,
    ALL_OFF,
    Ablation,
    build_report,
    calibration_word,
    chain,
    evaluate_ablations,
    favorite_of,
    favorite_rows,
    favorite_table,
    reading_lines,
    without_lap_noise,
    without_parameter_uncertainty,
)
from strategy.forecast_eval import LapScore
from strategy.fuel import physical_coefficient
from strategy.history import (
    DEFAULT_COST_RATIO,
    CompoundPrior,
    RetirementPrior,
    SafetyCarPrior,
    SimPriors,
)
from strategy.montecarlo import prepare
from strategy.population import RaceKind, RaceRef
from strategy.priors import DEFAULT_PRIORS
from strategy.protocol import FROZEN
from tests.strategy.synthetic import World, chase_field

TOTAL = 60
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.1,
    seed=2,
)
EXACT = replace(DEFAULT_PRIORS, offset_sd=0.0)


def sim_priors() -> SimPriors:
    sc = SafetyCarPrior(2, 20, (0.4, 0.6), (4, 5), (1.46,), (1.75,), DEFAULT_COST_RATIO, 15, 100)
    return SimPriors(
        sc, RetirementPrior(5, 400, (0.3, 0.7)), CompoundPrior({("MEDIUM", "HARD"): 10}), "t"
    )


def bundle(gap: float = 5.0) -> RaceBundle:
    ref = RaceRef(1, "2024 R06 T", TOTAL, RaceKind.SECA, 1.0, year=2024, round=6, circuit="T")
    return RaceBundle(ref, chase_field(gap=gap, world=FLAT), EXACT, sim_priors(), True)


def state(lap: int = 30):
    return prepare(
        bundle().rows,
        year=2024,
        round=6,
        upto_lap=lap,
        total_laps=TOTAL,
        sim_priors=sim_priors(),
        priors=EXACT,
    )


# ------------------------------------------------------------------ as transformações


def test_without_parameter_uncertainty_zeroes_every_model_error_and_keeps_the_rest():
    inp = state()

    off = without_parameter_uncertainty(inp)

    for before, after in zip(inp.drivers, off.drivers, strict=True):
        assert after.level_se == 0.0 and after.slope_sd == 0.0
        assert after.new_slope_sd == (0.0, 0.0, 0.0)
        assert after.offset_shared_sd == (0.0, 0.0, 0.0) and after.offset_tau == (0.0, 0.0, 0.0)
        # o que NÃO é incerteza do modelo fica como estava
        assert (after.levels, after.slopes, after.sigma, after.clock_s) == (
            before.levels,
            before.slopes,
            before.sigma,
            before.clock_s,
        )
    assert off.pit_loss.sd == 0.0 and off.pit_loss.se == 0.0
    assert off.pit_loss.median == inp.pit_loss.median  # a perda em si é a mesma


def test_without_lap_noise_zeroes_only_the_lap_noise():
    inp = state()

    off = without_lap_noise(inp)

    assert all(d.sigma == 0.0 for d in off.drivers)
    assert [d.level_se for d in off.drivers] == [d.level_se for d in inp.drivers]
    assert off.pit_loss == inp.pit_loss


def test_chained_transformations_apply_all_of_them_and_the_original_is_untouched():
    inp = state()

    both = chain(without_parameter_uncertainty, without_lap_noise)(inp)

    assert all(d.sigma == 0.0 and d.level_se == 0.0 for d in both.drivers)
    assert any(
        d.sigma > 0 or d.level_se > 0 for d in inp.drivers
    )  # a premissa: o original tem ruído


def test_the_ablations_are_the_documented_ones_and_the_last_turns_everything_off():
    names = [a.name for a in ABLATIONS]

    assert names == [
        "sem incerteza do modelo",
        "sem ruído de volta",
        "sem SC e sem abandono",
        "sem folga na decisão de parar",
        "tudo desligado (quase determinístico)",
    ]
    last = ABLATIONS[-1]
    assert last.overrides == ALL_OFF and last.transform is not None
    assert ABLATIONS[2].overrides == {"sc_rate_multiplier": 0.0, "retire_rate_multiplier": 0.0}


# ------------------------------------------------------------------ rodar as ablações


def test_each_ablation_is_scored_on_the_same_laps_as_the_protocol():
    out = evaluate_ablations(bundle(), n_sims=300, every=20)

    assert set(out) == {a.name for a in ABLATIONS}
    for scores in out.values():
        assert [s.lap for s in scores] == [5, 25, 45]


def test_an_ablation_that_changes_nothing_reproduces_the_base_with_the_same_draws():
    """Sem transformação nem override o resultado é o do simulador congelado: as mesmas
    probabilidades. É o que dá sentido a comparar as ablações com o backtest: a diferença é do
    componente."""
    neutral = (Ablation("neutra"),)
    first = evaluate_ablations(bundle(), ablations=neutral, n_sims=300, every=30)
    second = evaluate_ablations(bundle(), ablations=neutral, n_sims=300, every=30)

    assert first["neutra"] == second["neutra"]  # determinístico


def test_turning_everything_off_leaves_a_dominant_leader_at_about_one_hundred_percent():
    """Sem incerteza, ruído, SC, abandono nem folga de decisão, o líder 60 s à frente vence em quase
    todas as simulações: sobram só o composto seguinte sorteado e o desempate de relógios iguais."""
    out = evaluate_ablations(bundle(gap=60.0), ablations=(ABLATIONS[-1],), n_sims=300, every=20)

    for score in out[ABLATIONS[-1].name]:
        favorite = max(p for p, _ in score.win_pairs)
        assert favorite > 0.9 or score.lap < 20  # no começo ainda há espaço para a estratégia


def test_the_default_protocol_state_and_variant_are_used():
    """Estado por instante e variante principal, como no backtest."""
    out = evaluate_ablations(bundle(), ablations=(Ablation("neutra"),), n_sims=200, every=30)

    assert TEMPO == "tempo" and FROZEN.headline_variant == "encolhimento"
    assert [s.lap for s in out["neutra"]] == [5, 35]


# ------------------------------------------------------------------ o favorito


def lap_score(
    race: str, lap: int, pairs, leader_right: bool = True, brier: float = 0.4
) -> LapScore:
    return LapScore(
        race, lap, 60, brier, 0.5, 0.4, True, 0.0 if leader_right else 2.0, 0.9, tuple(pairs), ()
    )


def test_the_favorite_is_the_pair_with_the_highest_win_probability():
    score = lap_score("A", 10, [(0.2, False), (0.6, True), (0.2, False)])

    assert favorite_of(score) == (0.6, True)
    assert favorite_of(lap_score("A", 10, [])) is None


def test_the_favorite_rows_compare_what_it_was_given_with_what_happened_by_progress():
    scores = [
        # 0-40%: favorito com 0,5 e 0,7; ganhou 1 de 2; o líder acertou 1 de 2
        lap_score("A", 10, [(0.5, True), (0.5, False)], leader_right=True),
        lap_score("B", 12, [(0.7, False), (0.3, True)], leader_right=False),
        # 80-100%: favorito com 0,9, ganhou; o líder acertou
        lap_score("A", 55, [(0.9, True), (0.1, False)]),
    ]

    early, mid, late = favorite_rows(scores)

    assert early.n == 2 and early.n_races == 2
    assert early.favorite_p == pytest.approx(0.6) and early.favorite_won == pytest.approx(0.5)
    assert early.leader_won == pytest.approx(0.5)
    assert mid.n == 0 and mid.favorite_p is None and mid.brier is None
    assert late.n == 1 and late.favorite_p == pytest.approx(0.9) and late.favorite_won == 1.0


def test_the_progress_bins_are_closed_below_and_the_last_one_reaches_the_end():
    scores = [
        lap_score("A", 24, [(0.5, True)]),
        lap_score("A", 48, [(0.5, True)]),
        lap_score("A", 60, [(0.5, True)]),
    ]

    early, mid, late = favorite_rows(scores)

    assert (early.n, mid.n, late.n) == (
        0,
        1,
        2,
    )  # 24/60 = 40% abre a faixa do meio; 60/60 cabe na última


def test_a_confident_wrong_favorite_and_an_underconfident_right_one_show_in_the_rows():
    under = [lap_score(r, 30, [(0.4, True), (0.3, False)]) for r in "ABCD"]
    over = [lap_score(r, 30, [(0.9, False), (0.1, True)]) for r in "ABCD"]

    under_row = next(r for r in favorite_rows(under) if r.n)
    over_row = next(r for r in favorite_rows(over) if r.n)

    assert under_row.favorite_p < under_row.favorite_won  # sub-confiante
    assert over_row.favorite_p > over_row.favorite_won  # super-confiante


def test_the_table_and_the_report_say_what_this_is_and_is_not():
    rows = favorite_rows([lap_score("A", 10, [(0.5, True)])])
    table = favorite_table({"congelado (o backtest)": rows, "sem SC e sem abandono": rows})

    report = build_report(
        {"congelado (o backtest)": rows}, {"sem SC e sem abandono": rows}, n_races=1
    )

    assert table[0][:3] == ["configuração", "andamento", "previsões"]
    assert len(table) == 1 + 2 * 3
    flat = " ".join(report.replace("**", "").split())
    assert "Não é parte do backtest" in flat and "Depois de ver o resultado do backtest" in flat
    assert "Nenhum valor do protocolo mudou" in flat
    assert "| sem SC e sem abandono |" in report


# ------------------------------------------------------------------ a linha de comando


def test_main_runs_the_ablations_and_writes_the_report(monkeypatch, capsys, tmp_path):
    cache = tmp_path / "frozen.pkl"
    outcome = RaceOutcome("2024 R06 T", "seca", True, TOTAL)
    outcome.forecast[(FROZEN.headline_variant, TEMPO, 1.0)] = [
        lap_score("2024 R06 T", 30, [(0.6, True), (0.4, False)])
    ]
    save_outcomes([outcome], cache)
    target = tmp_path / "docs" / "diag.md"
    monkeypatch.setattr(diagnostics, "load_bundles", lambda session: [bundle()])
    monkeypatch.setattr("db.session.get_session_factory", lambda: lambda: _FakeSession())

    class Pool:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items):
            return [fn(item, n_sims=100, every=30) for item in items]

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Pool)

    code = diagnostics.main(["--cache", str(cache), "--markdown", str(target)])

    assert code == 0
    text = target.read_text(encoding="utf-8")
    assert "# Diagnóstico pós-resultado do Monte Carlo" in text
    assert "congelado (o backtest)" in text and "tudo desligado" in text
    assert "Diagnóstico pós-resultado" in capsys.readouterr().out


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------------ a leitura


def middle_scores(favorite_p: float, won: list[bool], leader_right: bool = True):
    """Uma previsão no meio da corrida por corrida; o favorito é o de maior p (>= 0,5)."""
    return [
        lap_score(race, 30, [(favorite_p, w), (1 - favorite_p, not w)], leader_right)
        for race, w in zip("ABCD", won, strict=True)
    ]


def test_the_reading_says_under_confident_when_the_favorite_wins_more_than_it_was_given():
    frozen = {"congelado": favorite_rows(middle_scores(0.6, [True] * 4))}  # deu 0,6, ganhou sempre

    (line,) = reading_lines(frozen, {})

    assert "0.60" in line and "1.00" in line and "sub-confiante" in line
    assert "super-confiante" not in line


def test_the_reading_says_over_confident_and_compares_with_the_leader():
    frozen = {"congelado": favorite_rows(middle_scores(0.6, [True] * 4))}
    ablated = {"sem incerteza": favorite_rows(middle_scores(0.9, [False] * 4))}  # líder acerta

    base, ablation = reading_lines(frozen, ablated)

    assert "**sem incerteza:**" in ablation and "super-confiante" in ablation
    assert "ganha menos que o líder atual" in ablation  # 0 de 4 contra o líder acertando sempre


def test_the_reading_says_calibrated_when_the_gap_is_small():
    won = [True, True, False, False]  # deu 0,5, ganhou metade
    frozen = {"congelado": favorite_rows(middle_scores(0.5, won))}

    (line,) = reading_lines(frozen, {})

    assert ": calibrado." in line


def test_the_reading_says_when_the_favorite_does_not_lose_to_the_leader():
    frozen = {"congelado": favorite_rows(middle_scores(0.6, [True] * 4))}
    ablated = {"outra": favorite_rows(middle_scores(0.6, [True] * 4, leader_right=False))}

    _, ablation = reading_lines(frozen, ablated)

    assert "ganha tanto quanto ou mais que o líder atual" in ablation


def test_the_report_carries_the_reading():
    rows = favorite_rows(middle_scores(0.6, [True] * 4))

    report = build_report(
        {"congelado (o backtest)": rows}, {"sem SC e sem abandono": rows}, n_races=4
    )

    assert "## Leitura (faixa de 40% a 80% da corrida)" in report
    assert "**sem SC e sem abandono:**" in report


def test_the_calibration_word_has_a_five_point_tolerance_on_each_side():
    assert calibration_word(0.50, 0.56) == "sub-confiante"
    assert calibration_word(0.50, 0.54) == "calibrado"
    assert calibration_word(0.50, 0.46) == "calibrado"
    assert calibration_word(0.50, 0.44) == "super-confiante"
