"""O orquestrador do backtest (strategy/backtest.py): roda o protocolo congelado, uma corrida por
vez."""

from collections import Counter
from dataclasses import replace

import pytest
from sqlalchemy.orm import sessionmaker

from strategy import backtest
from strategy.backtest import (
    LIVE,
    ORACLE,
    TEMPO,
    VOLTA,
    RaceBundle,
    evaluate_bundle,
    load_bundles,
    load_outcomes,
    run_backtest,
    save_outcomes,
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
from strategy.population import RaceKind, RaceRef
from strategy.priors import DEFAULT_PRIORS, RaceStats
from strategy.protocol import FROZEN, Protocol, frozen_sim_config, frozen_undercut_config
from tests.strategy.synthetic import World, chase_field
from tests.strategy.test_backtest_report import outcomes as report_outcomes

TOTAL = 60
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)
EXACT = replace(DEFAULT_PRIORS, offset_sd=0.0)


def sim_priors() -> SimPriors:
    sc = SafetyCarPrior(2, 20, (0.4, 0.6), (4, 5), (1.46,), (1.75,), DEFAULT_COST_RATIO, 15, 100)
    return SimPriors(
        sc, RetirementPrior(5, 400, (0.3, 0.7)), CompoundPrior({("MEDIUM", "HARD"): 10}), "t"
    )


def ref(label="2024 R06 Teste", round_=6, kind=RaceKind.SECA) -> RaceRef:
    return RaceRef(1, label, TOTAL, kind, 1.0, year=2024, round=round_, circuit="T")


def bundle(*, main=True, gap=5.0, a_stop=45, r_stop=45, label="2024 R06 Teste", round_=6, **kw):
    rows = chase_field(a_stop=a_stop, r_stop=r_stop, gap=gap, world=FLAT)
    return RaceBundle(
        ref(label, round_, RaceKind.SECA if main else RaceKind.MOLHADA),
        rows,
        EXACT,
        sim_priors(),
        main,
    )


QUICK = {"n_sims": 200, "every": 20}  # laps 5, 25, 45


@pytest.fixture(scope="module")
def outcome():
    return evaluate_bundle(bundle(a_stop=31, r_stop=35), **QUICK)


# ------------------------------------------------------------------ uma corrida seca


def test_a_dry_race_runs_every_configuration_of_the_protocol(outcome):
    variants = set(FROZEN.variants)

    assert {v for v, snap in outcome.laptime if snap == TEMPO} == variants
    assert (FROZEN.headline_variant, VOLTA) in outcome.laptime  # o diagnóstico do vazamento
    forecast = {(v, snap, m) for v, snap, m in outcome.forecast}
    assert {(v, TEMPO, 1.0) for v in variants} <= forecast
    assert {(FROZEN.headline_variant, TEMPO, m) for m in FROZEN.sc_multipliers} <= forecast
    assert (FROZEN.headline_variant, VOLTA, 1.0) in forecast
    assert {(LIVE, TEMPO), (ORACLE, TEMPO), (LIVE, VOLTA)} == set(outcome.undercut)
    assert outcome.main and outcome.kind == "seca" and outcome.total_laps == TOTAL
    assert outcome.seconds > 0


def test_the_laps_evaluated_are_the_ones_of_the_protocol_thinned_by_every(outcome):
    scores = outcome.forecast[(FROZEN.headline_variant, TEMPO, 1.0)]

    assert [s.lap for s in scores] == [5, 25, 45]  # first_lap=5, a cada 20


def test_the_default_laps_are_every_fifth_from_lap_five():
    out = evaluate_bundle(bundle(), n_sims=100)

    scores = out.forecast[(FROZEN.headline_variant, TEMPO, 1.0)]
    assert [s.lap for s in scores] == list(range(5, TOTAL, 5))


def test_the_variants_use_the_same_races_laps_and_drivers(outcome):
    keys = [(v, TEMPO, 1.0) for v in FROZEN.variants]

    laps = {k: [s.lap for s in outcome.forecast[k]] for k in keys}

    assert len({tuple(v) for v in laps.values()}) == 1


def test_the_sc_multipliers_share_the_laps_and_move_only_the_probability_of_sc(outcome):
    base = outcome.forecast[(FROZEN.headline_variant, TEMPO, 1.0)]
    low = outcome.forecast[(FROZEN.headline_variant, TEMPO, 0.5)]
    high = outcome.forecast[(FROZEN.headline_variant, TEMPO, 2.0)]

    assert [s.lap for s in base] == [s.lap for s in low] == [s.lap for s in high]


def test_the_real_undercut_is_found_and_scored_in_both_modes():
    out = evaluate_bundle(bundle(a_stop=31, r_stop=35), **QUICK)

    live, oracle = out.undercut[(LIVE, TEMPO)], out.undercut[(ORACLE, TEMPO)]

    assert [(r.event.attacker, r.event.rival, r.event.lap) for r in live] == [("A", "R", 30)]
    assert [r.event.lap for r in oracle] == [30]
    assert live[0].realized == oracle[0].realized  # o mesmo resultado real nos dois modos


def test_the_skipped_events_are_reported_by_reason_counted_once(outcome):
    """Os motivos vêm só do estado por instante (o resultado): o diagnóstico do estado antigo vê os
    mesmos eventos e não pode duplicar a contagem."""
    from strategy.undercut_eval import evaluate_race_modes

    b = bundle(a_stop=31, r_stop=35)
    _, once = evaluate_race_modes(
        b.rows,
        TOTAL,
        race=b.ref.label,
        priors=b.priors,
        config=frozen_undercut_config(),
        snapshot=TEMPO,
        modes=(False, True),
    )

    assert outcome.skipped == dict(once) and sum(outcome.skipped.values()) > 0
    assert all(isinstance(k, str) for k in outcome.skipped)


# ------------------------------------------------------------------ uma corrida de fora


def test_a_race_outside_the_main_population_runs_only_the_headline_configuration():
    out = evaluate_bundle(bundle(main=False, a_stop=31, r_stop=35), **QUICK)

    assert not out.main and out.kind == "molhada"
    assert list(out.laptime) == [(FROZEN.headline_variant, TEMPO)]
    assert list(out.forecast) == [(FROZEN.headline_variant, TEMPO, 1.0)]
    assert list(out.undercut) == [(LIVE, TEMPO)]


# ------------------------------------------------------------------ o protocolo é o que manda


def test_the_backtest_runs_with_the_frozen_configs_not_the_module_defaults(monkeypatch):
    seen = {}

    def spy_configs(rows, race_ref, *, laps, priors, sim_priors, configs, config):
        seen["sim"] = config
        seen["configs"] = configs
        return {}

    def spy_modes(rows, total, **kwargs):
        seen["undercut"] = kwargs["config"]
        return {m: [] for m in kwargs["modes"]}, Counter()

    monkeypatch.setattr(backtest, "evaluate_race_configs", spy_configs)
    monkeypatch.setattr(backtest, "evaluate_race_modes", spy_modes)
    monkeypatch.setattr(backtest, "lap_time_records", lambda *a, **k: [])
    changed = Protocol(sc_wear=0.9, alert_threshold=0.7, undercut_sd_inflation=1.5, n_sims=123)

    evaluate_bundle(bundle(), changed)

    assert seen["sim"] == frozen_sim_config(changed) and seen["sim"].sc_wear == 0.9
    assert seen["sim"].n_sims == 123
    assert seen["undercut"] == frozen_undercut_config(changed)
    assert seen["undercut"].threshold == 0.7 and seen["undercut"].sd_inflation == 1.5


def test_the_n_sims_override_exists_only_for_the_verification_runs(monkeypatch):
    seen = {}
    monkeypatch.setattr(backtest, "evaluate_race_configs", lambda *a, **k: seen.update(k) or {})
    monkeypatch.setattr(
        backtest, "evaluate_race_modes", lambda *a, **k: ({m: [] for m in k["modes"]}, Counter())
    )
    monkeypatch.setattr(backtest, "lap_time_records", lambda *a, **k: [])

    evaluate_bundle(bundle(), n_sims=77)

    assert seen["config"].n_sims == 77 and FROZEN.n_sims == 5000


# ------------------------------------------------------------------ várias corridas


def test_the_races_run_one_per_process_and_come_back_in_the_given_order():
    bundles = [
        bundle(label="2024 R01 A", round_=1, a_stop=31, r_stop=35),
        bundle(label="2024 R02 B", round_=2, a_stop=33, r_stop=37),
    ]
    messages = []

    outcomes = run_backtest(bundles, workers=2, progress=messages.append, **QUICK)

    assert [o.label for o in outcomes] == ["2024 R01 A", "2024 R02 B"]
    assert len(messages) == 2 and all(" s" in m for m in messages)


def test_a_single_worker_runs_sequentially_with_the_same_result():
    one = [bundle(label="2024 R01 A", round_=1, a_stop=31, r_stop=35)]

    sequential = run_backtest(one, workers=1, **QUICK)
    parallel = run_backtest(one * 2, workers=2, **QUICK)

    assert sequential[0].forecast.keys() == parallel[0].forecast.keys()
    scores_a = [s.brier_win for s in sequential[0].forecast[(FROZEN.headline_variant, TEMPO, 1.0)]]
    scores_b = [s.brier_win for s in parallel[0].forecast[(FROZEN.headline_variant, TEMPO, 1.0)]]
    assert scores_a == scores_b  # a semente é da corrida e da volta, não do processo


def test_the_outcomes_survive_a_round_trip_through_the_cache(tmp_path, outcome):
    path = tmp_path / "cache" / "backtest.pkl"

    save_outcomes([outcome], path)
    (loaded,) = load_outcomes(path)

    assert loaded.label == outcome.label and loaded.forecast.keys() == outcome.forecast.keys()
    assert [s.brier_win for s in loaded.forecast[(FROZEN.headline_variant, TEMPO, 1.0)]] == [
        s.brier_win for s in outcome.forecast[(FROZEN.headline_variant, TEMPO, 1.0)]
    ]


# ------------------------------------------------------------------ as corridas e as constantes


def stat(label: str, round_: int) -> RaceStats:
    return RaceStats(label, label, 2024, round_, 0.02, 0.02, 22.0 + round_, 1.0, 8)


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


def test_each_dry_race_gets_constants_without_itself_and_the_others_get_all_of_them(monkeypatch):
    dry = [ref(f"R{i}", i) for i in (1, 2, 3, 4)]
    wet = [ref("W9", 9, RaceKind.MOLHADA)]
    monkeypatch.setattr(backtest, "split_population", lambda session, year=None: (dry, wet))
    monkeypatch.setattr(
        backtest,
        "load_race_stats",
        lambda session, year=None: [stat(r.label, r.round) for r in dry],
    )
    monkeypatch.setattr(
        backtest,
        "load_race_histories",
        lambda session, year=None: [history(r.label, r.round) for r in dry],
    )
    monkeypatch.setattr(backtest, "load_race_laps", lambda session, race_id: [])

    bundles = load_bundles(object())

    by = {b.ref.label: b for b in bundles}
    assert [b.main for b in bundles] == [True, True, True, True, False]
    for label in ("R1", "R2", "R3", "R4"):
        assert f"sem {label}" in by[label].priors.source
        assert f"sem {label}" in by[label].sim_priors.source
        assert by[label].sim_priors.sc.races == 3
    assert (
        by["W9"].sim_priors.sc.races == 4
    )  # a molhada vê as 4 secas: não tem corrida a deixar de fora
    assert "4 corridas" in by["W9"].priors.source


def test_rounds_limit_the_races_but_not_the_pool_of_constants(monkeypatch):
    dry = [ref(f"R{i}", i) for i in (1, 2, 3, 4)]
    monkeypatch.setattr(backtest, "split_population", lambda session, year=None: (dry, []))
    monkeypatch.setattr(
        backtest,
        "load_race_stats",
        lambda session, year=None: [stat(r.label, r.round) for r in dry],
    )
    monkeypatch.setattr(
        backtest,
        "load_race_histories",
        lambda session, year=None: [history(r.label, r.round) for r in dry],
    )
    monkeypatch.setattr(backtest, "load_race_laps", lambda session, race_id: [])

    (only,) = load_bundles(object(), rounds=[3])

    assert only.ref.label == "R3" and only.sim_priors.sc.races == 3


# ------------------------------------------------------------------ a linha de comando


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(backtest, "load_bundles", lambda session, rounds=None: [])
    monkeypatch.setattr(backtest, "get_session_factory", lambda: sessionmaker())

    assert backtest.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_reuses_the_cache_and_writes_the_report(monkeypatch, capsys, tmp_path, outcome):
    cache = tmp_path / "b.pkl"
    save_outcomes([outcome], cache)
    target = tmp_path / "docs" / "backtest.md"

    code = backtest.main(
        [
            "--reuse",
            "--cache",
            str(cache),
            "--markdown",
            str(target),
            "--adjustment",
            "o limiar foi para 0,7",
        ]
    )

    text = target.read_text(encoding="utf-8")
    assert code == 0 and "# Backtest da Fase 3" in text
    assert "o limiar foi para 0,7" in text and "Nenhum. Nenhum valor" not in text
    assert "Backtest da Fase 3" in capsys.readouterr().out


def test_reuse_needs_a_cache(capsys):
    with pytest.raises(SystemExit):
        backtest.main(["--reuse"])


def test_the_quick_flag_shrinks_the_run_and_is_labeled(monkeypatch, capsys):
    seen = {}

    def fake_run(bundles, **kwargs):
        seen.update(kwargs)
        return report_outcomes()  # o relatório tem um resumo: precisa de corridas para resumir

    monkeypatch.setattr(backtest, "load_bundles", lambda session, rounds=None: [bundle()])
    monkeypatch.setattr(backtest, "get_session_factory", lambda: sessionmaker())
    monkeypatch.setattr(backtest, "run_backtest", fake_run)

    backtest.main(["--quick"])

    assert seen["n_sims"] == 300 and seen["every"] == 20
    assert "VERIFICAÇÃO RÁPIDA" in capsys.readouterr().out


def test_the_cache_written_by_python_dash_m_can_be_read_by_another_process(tmp_path):
    """Com `python -m strategy.backtest` as classes do módulo viravam `__main__.RaceOutcome`, e o
    cache gravado não podia ser relido. O ponto de entrada importa o módulo pelo nome."""
    import subprocess
    import sys

    cache = tmp_path / "b.pkl"
    code = (
        "import sys; sys.argv = ['x']; import runpy; "
        "from strategy import backtest; "
        "from tests.strategy.test_backtest import bundle, QUICK; "
        "outs = backtest.run_backtest([bundle()], workers=1, **QUICK); "
        f"backtest.save_outcomes(outs, __import__('pathlib').Path({str(cache)!r}))"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

    (loaded,) = load_outcomes(cache)

    assert type(loaded).__module__ == "strategy.backtest"
