"""Testes da avaliação do Monte Carlo (strategy/forecast_eval.py).

As notas têm conta de papel; o bootstrap é testado nas propriedades que importam: a unidade é a
CORRIDA (mais voltas da mesma corrida não estreitam o intervalo), e o resultado é reproduzível.
"""

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from strategy import forecast_eval
from strategy.forecast_eval import (
    Interval,
    LapScore,
    brier_by_progress,
    build_report,
    cluster_bootstrap,
    current_leader,
    evaluate_population,
    evaluate_race,
    paired_bootstrap,
    per_race_mean,
    score_forecast,
)
from strategy.fuel import physical_coefficient
from strategy.history import (
    DEFAULT_COST_RATIO,
    CompoundPrior,
    RaceHistory,
    RetirementPrior,
    SafetyCarPrior,
    SimPriors,
)
from strategy.montecarlo import DriverForecast, RaceForecast, SimConfig
from strategy.population import RaceKind, RaceRef
from strategy.priors import RaceStats
from tests.strategy.synthetic import World, chase_field

TOTAL = 60
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)


def forecast_of(win: dict[str, float], podium: dict[str, float], lap: int = 30) -> RaceForecast:
    drivers = tuple(DriverForecast(d, win[d], podium[d], 0.0, 0.0, 0.0, 0.0, "modelo") for d in win)
    return RaceForecast(2024, 1, lap, TOTAL, 1000, drivers, 0.0, 1.0, 0.0, (), "encolhimento")


# ------------------------------------------------------------------ a nota de uma previsão


def test_the_scores_are_the_hand_computed_brier_of_win_and_podium():
    """A (0,6 / 0,9), B (0,3 / 0,7), C (0,1 / 0,4); venceu B, o pódio foi B, A e D, e D nem estava
    no simulador (previsto com probabilidade 0)."""
    forecast = forecast_of({"A": 0.6, "B": 0.3, "C": 0.1}, {"A": 0.9, "B": 0.7, "C": 0.4})

    score = score_forecast(forecast, ["B", "A", "D", "C"], race="X", current_leader="A")

    assert score.brier_win == pytest.approx(0.6**2 + 0.7**2 + 0.1**2)  # 0,36 + 0,49 + 0,01
    assert score.brier_podium == pytest.approx(0.1**2 + 0.3**2 + 0.4**2 + 1.0)  # + D ausente
    assert score.p_winner == pytest.approx(0.3) and not score.favorite_won
    assert (score.race, score.lap, score.total_laps) == ("X", 30, TOTAL)
    assert score.progress == pytest.approx(0.5)


def test_the_comparators_are_the_current_leader_and_the_uniform_forecast():
    forecast = forecast_of({"A": 0.6, "B": 0.3, "C": 0.1}, {"A": 0.9, "B": 0.7, "C": 0.4})

    score = score_forecast(forecast, ["B", "A", "C"], race="X", current_leader="A")

    # o líder atual (A) vence com p = 1: (1 - 0)^2 + (0 - 1)^2 + 0
    assert score.leader_brier == pytest.approx(2.0)
    # todos iguais (1/3): (1/3)^2 + (1/3 - 1)^2 + (1/3)^2 = 6/9
    assert score.uniform_brier == pytest.approx(6 / 9)


def test_a_perfect_forecast_scores_zero_and_a_confidently_wrong_one_scores_two():
    perfect = forecast_of({"A": 1.0, "B": 0.0}, {"A": 1.0, "B": 1.0})
    wrong = forecast_of({"A": 0.0, "B": 1.0}, {"A": 1.0, "B": 1.0})

    good = score_forecast(perfect, ["A", "B"], race="X", current_leader="A")
    bad = score_forecast(wrong, ["A", "B"], race="X", current_leader="B")

    assert good.brier_win == 0.0 and good.favorite_won and good.p_winner == 1.0
    assert bad.brier_win == pytest.approx(2.0) and not bad.favorite_won


def test_a_winner_the_simulator_did_not_have_counts_as_a_miss():
    forecast = forecast_of({"A": 0.7, "B": 0.3}, {"A": 0.9, "B": 0.9})

    score = score_forecast(forecast, ["Z", "A", "B"], race="X", current_leader="A")

    assert score.p_winner == 0.0
    assert score.brier_win == pytest.approx(0.7**2 + 0.3**2 + 1.0)
    assert score.uniform_brier == pytest.approx(2 * 0.5**2 + 1.0)


def test_the_current_leader_is_the_smallest_clock_on_that_lap():
    rows = chase_field(gap=5.0, world=FLAT)

    assert current_leader(rows, 30) == "R"
    assert current_leader(rows, 999) is None


# ------------------------------------------------------------------ bootstrap sobre corridas


def test_the_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    rng = np.random.default_rng(5)
    per_race = {f"R{i}": float(v) for i, v in enumerate(rng.normal(1.0, 0.3, 21))}

    first = cluster_bootstrap(per_race, seed=3)
    again = cluster_bootstrap(per_race, seed=3)
    other = cluster_bootstrap(per_race, seed=4)

    assert first == again  # mesma semente, mesmo intervalo
    assert (other.low, other.high) != (first.low, first.high)  # outra semente, outro intervalo
    assert first.mean == pytest.approx(np.mean(list(per_race.values())))
    assert first.low < first.mean < first.high and first.n_races == 21


def test_the_interval_width_matches_the_normal_approximation_for_the_mean():
    rng = np.random.default_rng(0)
    values = {f"R{i}": float(v) for i, v in enumerate(rng.normal(1.0, 0.3, 60))}

    interval = cluster_bootstrap(values, n_boot=4000)

    half = 1.96 * np.std(list(values.values()), ddof=1) / np.sqrt(60)
    assert (interval.high - interval.low) / 2 == pytest.approx(half, rel=0.15)


def test_identical_races_give_a_degenerate_interval_and_one_race_gives_a_point():
    assert cluster_bootstrap({"a": 0.5, "b": 0.5, "c": 0.5}) == Interval(0.5, 0.5, 0.5, 3)
    assert cluster_bootstrap({"a": 0.7}) == Interval(0.7, 0.7, 0.7, 1)
    assert cluster_bootstrap({}) is None


def lap_score(race: str, lap: int, value: float, total: int = 60) -> LapScore:
    return LapScore(race, lap, total, value, value, 0.5, True, value, value)


def test_the_unit_is_the_race_so_more_laps_of_the_same_races_do_not_narrow_the_interval():
    """O ponto do módulo: 5 corridas com 1 volta cada ou com 50 voltas cada (todas iguais dentro da
    corrida) têm o MESMO intervalo, porque a corrida é a unidade."""
    values = {"A": 0.4, "B": 0.9, "C": 0.6, "D": 1.2, "E": 0.8}
    few = [lap_score(race, 30, v) for race, v in values.items()]
    many = [lap_score(race, lap, v) for race, v in values.items() for lap in range(1, 51)]

    interval_few = cluster_bootstrap(per_race_mean(few, "brier_win"))
    interval_many = cluster_bootstrap(per_race_mean(many, "brier_win"))

    assert interval_few.n_races == interval_many.n_races == 5  # e não 250
    for field in ("mean", "low", "high"):
        assert getattr(interval_few, field) == pytest.approx(getattr(interval_many, field))


def test_the_laps_are_averaged_inside_each_race_before_the_races_are_compared():
    """A corrida X tem 3 voltas (1, 1, 4) e a Y uma (6). Por corrida: 2 e 6, média 4. Juntando as
    voltas (a média das 4) daria 3,0: a corrida com mais voltas pesaria mais."""
    scores = [
        lap_score("X", 10, 1.0),
        lap_score("X", 11, 1.0),
        lap_score("X", 12, 4.0),
        lap_score("Y", 10, 6.0),
    ]

    per_race = per_race_mean(scores, "brier_win")

    assert per_race == {"X": 2.0, "Y": 6.0}
    assert cluster_bootstrap(per_race).mean == 4.0


def test_the_progress_bins_split_the_laps_by_how_far_the_race_had_gone():
    scores = [lap_score("X", 3, 1.0), lap_score("X", 33, 2.0), lap_score("X", 57, 3.0)]

    assert per_race_mean(scores, "brier_win", low=0.0, high=0.1) == {"X": 1.0}  # 3/60 = 5%
    assert per_race_mean(scores, "brier_win", low=0.5, high=0.6) == {"X": 2.0}  # 33/60 = 55%
    assert per_race_mean(scores, "brier_win", low=0.9, high=1.0) == {"X": 3.0}  # 95%
    assert per_race_mean(scores, "brier_win", low=0.2, high=0.3) == {}


def test_the_last_progress_bin_includes_the_very_end():
    finished = lap_score("X", 60, 5.0)  # progresso 1,0

    assert per_race_mean([finished], "brier_win", low=0.9, high=1.0) == {"X": 5.0}


def test_the_brier_by_progress_has_one_interval_per_bin_and_none_where_there_is_no_data():
    scores = [lap_score(r, lap, 0.5) for r in "ABC" for lap in (5, 35)]

    table = brier_by_progress(scores, "brier_win", bins=10)

    assert len(table) == 10 and table[0][0] == 0.0 and table[-1][1] == 1.0
    filled = [interval for _, _, interval in table if interval is not None]
    assert len(filled) == 2 and all(i.n_races == 3 for i in filled)
    assert table[3][2] is None  # 30% a 40%: nenhuma volta


def test_the_paired_bootstrap_compares_races_present_in_both_and_measures_the_difference():
    a = {"X": 0.5, "Y": 0.7, "Z": 0.9}
    b = {"X": 0.4, "Y": 0.6, "W": 5.0}  # W só existe em b, Z só em a

    interval = paired_bootstrap(a, b)

    assert interval.n_races == 2 and interval.mean == pytest.approx(0.1)
    assert interval.low == pytest.approx(0.1) and interval.high == pytest.approx(0.1)
    assert paired_bootstrap(a, {}) is None


# ------------------------------------------------------------------ uma corrida


def ref(label="2024 R06 Teste", round_=6) -> RaceRef:
    return RaceRef(1, label, TOTAL, RaceKind.SECA, 1.0, year=2024, round=round_, circuit="T")


def quiet_priors() -> SimPriors:
    sc = SafetyCarPrior(0, 20, (), (4,), (1.46,), (1.75,), DEFAULT_COST_RATIO, 15, 100)
    return SimPriors(sc, RetirementPrior(0, 400, ()), CompoundPrior({("MEDIUM", "HARD"): 10}), "t")


def test_a_race_is_scored_lap_by_lap_against_the_order_on_the_track():
    rows = chase_field(gap=5.0, world=FLAT)

    scores = evaluate_race(
        rows,
        ref(),
        laps=[20, 40, 0, TOTAL],  # 0 e a última volta não têm o que prever
        sim_priors=quiet_priors(),
        config=SimConfig(n_sims=400),
    )

    assert [s.lap for s in scores] == [20, 40]
    for s in scores:
        assert s.race == "2024 R06 Teste" and s.total_laps == TOTAL
        assert 0.0 <= s.brier_win <= 2.0 and 0.0 <= s.p_winner <= 1.0
    assert scores[1].leader_brier in (0.0, 2.0)  # o líder atual vence ou não: 0 ou 2


def test_a_race_with_too_few_drivers_has_nothing_to_score():
    rows = [r for r in chase_field(gap=5.0, world=FLAT) if r.driver in ("R", "A")]

    assert evaluate_race(rows, ref(), laps=[20], sim_priors=quiet_priors()) == []


# ------------------------------------------------------------------ a população


def stat(label: str, round_: int) -> RaceStats:
    return RaceStats(label, label, 2024, round_, 0.02, 0.02, 22.0, 1.0, 8)


def history(label: str, round_: int) -> RaceHistory:
    return RaceHistory(
        label,
        2024,
        round_,
        TOTAL,
        ((10, 12),),
        (),
        (),
        20,
        (30,),
        (("MEDIUM", "HARD"),),
        None,
        None,
        (),
        (),
    )


def test_each_race_is_evaluated_with_both_sets_of_constants_computed_without_it(monkeypatch):
    refs = [ref(f"R{i}", i) for i in (1, 2, 3, 4)]
    seen = {}

    def spy(rows, race_ref, *, laps, priors, sim_priors, config, variant):
        seen[race_ref.label] = (priors, sim_priors, laps, config.n_sims)
        return []

    monkeypatch.setattr(forecast_eval, "split_population", lambda session, year=None: (refs, []))
    monkeypatch.setattr(
        forecast_eval,
        "load_race_stats",
        lambda session, year=None: [stat(r.label, r.round) for r in refs],
    )
    monkeypatch.setattr(
        forecast_eval,
        "load_race_histories",
        lambda session, year=None: [history(r.label, r.round) for r in refs],
    )
    monkeypatch.setattr(forecast_eval, "load_race_laps", lambda session, race_id: [])
    monkeypatch.setattr(forecast_eval, "evaluate_race", spy)

    by_race = evaluate_population(object(), every=10, n_sims=777)

    assert set(by_race) == {"R1", "R2", "R3", "R4"}
    for label, (priors, sim_priors, laps, sims) in seen.items():
        assert f"sem {label}" in priors.source and f"sem {label}" in sim_priors.source
        assert laps == list(range(5, TOTAL, 10)) and sims == 777
    assert seen["R4"][1].sc.races == 3  # as outras três: o histórico dela não entra


def test_rounds_limit_the_races_evaluated_but_not_the_pool_of_constants(monkeypatch):
    refs = [ref(f"R{i}", i) for i in (1, 2, 3, 4)]
    seen = {}

    def spy(rows, race_ref, *, laps, priors, sim_priors, config, variant):
        seen[race_ref.label] = sim_priors
        return []

    monkeypatch.setattr(forecast_eval, "split_population", lambda session, year=None: (refs, []))
    monkeypatch.setattr(
        forecast_eval,
        "load_race_stats",
        lambda session, year=None: [stat(r.label, r.round) for r in refs],
    )
    monkeypatch.setattr(
        forecast_eval,
        "load_race_histories",
        lambda session, year=None: [history(r.label, r.round) for r in refs],
    )
    monkeypatch.setattr(forecast_eval, "load_race_laps", lambda session, race_id: [])
    monkeypatch.setattr(forecast_eval, "evaluate_race", spy)

    evaluate_population(object(), rounds=[3])

    assert list(seen) == ["R3"] and seen["R3"].sc.races == 3


# ------------------------------------------------------------------ o relatório e a CLI


def population():
    return {
        "R1": [lap_score("R1", lap, 0.4) for lap in (10, 30, 50)],
        "R2": [lap_score("R2", lap, 0.9) for lap in (10, 30, 50)],
        "R3": [lap_score("R3", lap, 0.6) for lap in (10, 30, 50)],
    }


def test_the_text_report_states_the_unit_of_the_interval_and_the_comparators():
    text = build_report(population())

    assert "3 corridas, 9 previsões" in text and "não a volta" in text
    assert "o líder atual vence" in text.lower() or "o líder atual vence" in text
    assert "todos iguais" in text and "bootstrap" in text


def test_the_markdown_report_explains_why_the_race_and_not_the_lap_is_the_sample():
    md = build_report(population(), markdown=True)
    flat = " ".join(md.replace("**", "").split())  # o texto fixo quebra linhas e tem negrito

    assert "A unidade é a corrida, não a volta" in flat
    assert "tamanho efetivo da amostra é o número de corridas" in flat
    assert "bootstrap sobre corridas" in flat and "ordem na pista" in flat
    assert "| 0% a 10% |" in md or "andamento" in md


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(forecast_eval, "evaluate_population", lambda *a, **k: {})
    monkeypatch.setattr(forecast_eval, "get_session_factory", lambda: sessionmaker())

    assert forecast_eval.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_passes_its_options_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    calls = {}

    def fake(session, **kwargs):
        calls.update(kwargs)
        return population()

    monkeypatch.setattr(forecast_eval, "evaluate_population", fake)
    monkeypatch.setattr(forecast_eval, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "mc.md"

    code = forecast_eval.main(
        ["--round", "6", "--every", "10", "--sims", "500", "--markdown", str(target)]
    )

    assert code == 0
    assert (calls["rounds"], calls["every"], calls["n_sims"]) == ([6], 10, 500)
    assert "Brier" in capsys.readouterr().out
    assert "# Avaliação do Monte Carlo" in target.read_text(encoding="utf-8")
