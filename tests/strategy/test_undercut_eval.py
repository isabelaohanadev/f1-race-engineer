"""Testes da avaliação do undercut nas paradas reais (strategy/undercut_eval.py).

Corridas sintéticas em que o undercut ACONTECE no gerador: a margem realizada é a diferença dos
relógios, conhecida, e a prevista tem a conta de papel de `test_undercut.py` (8,4 s no mundo plano).
As métricas (Brier, curva de calibração, precisão, fator de inflação) têm resposta calculada à mão.
"""

from collections import Counter
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from strategy import undercut_eval
from strategy.fuel import physical_coefficient
from strategy.population import RaceKind, RaceRef
from strategy.priors import DEFAULT_PRIORS, RaceStats
from strategy.snapshot import driver_crossing_ms
from strategy.undercut import (
    UndercutAssessment,
    UndercutConfig,
    assess_undercut,
    build_context,
    normal_cdf,
)
from strategy.undercut_eval import (
    EventResult,
    UndercutEvent,
    brier_score,
    build_report,
    calibrate_inflation,
    evaluate_population,
    evaluate_race,
    find_events,
    is_neutralized,
    loo_inflation,
    precision_at,
    realized_margin,
    reliability_curve,
    summarize,
)
from tests.strategy.synthetic import World, chase_field

TOTAL, N, K = 60, 30, 5
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)
EXACT = replace(DEFAULT_PRIORS, offset_sd=0.0)


def real_race(gap: float = 5.0, a_stop: int = N + 1, r_stop: int = N + K, world: World = FLAT):
    """A corrida em que o atacante para na N+1 e o rival na N+K: o undercut de verdade."""
    return chase_field(a_stop=a_stop, r_stop=r_stop, gap=gap, world=world)


def only_event(rows, **kwargs) -> UndercutEvent:
    events, _ = find_events(rows, TOTAL, race="X", **kwargs)
    assert len(events) == 1, events
    return events[0]


# ------------------------------------------------------------------ achar os eventos


def test_a_real_stop_with_a_rival_that_stops_later_is_one_event():
    event = only_event(real_race())

    assert (event.attacker, event.rival, event.lap) == ("A", "R", N)
    assert event.rival_stop == N + K and event.horizon == K and event.end_lap == N + K + 2
    assert event.gap_s == pytest.approx(5.0)
    assert (event.new_compound_a, event.new_compound_r) == (
        "HARD",
        "HARD",
    )  # os que de fato entraram


def test_the_stops_of_the_other_cars_are_left_out_with_their_reasons():
    """Os 12 carros lentos também param, mas atrás de quem já parou, ou sem o rival parar a seguir:
    ficam de fora e cada um é contado pelo motivo."""
    _, skipped = find_events(real_race(), TOTAL, race="X")

    assert (
        sum(skipped.values()) == 13
    )  # 12 carros lentos + o rival R (parou, sem rival que pare depois)
    assert set(skipped) <= {
        "rival em outro stint",
        "horizonte fora do intervalo",
        "sem rival à frente",
        "rival não parou",
        "volta inicial",
        "paradas na mesma volta",
    }


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"a_stop": 31, "r_stop": 999}, "rival não parou"),  # nunca entra nos boxes
        ({"a_stop": 31, "r_stop": 31}, "paradas na mesma volta"),
        ({"a_stop": 31, "r_stop": 45}, "horizonte fora do intervalo"),  # K = 15
        ({"a_stop": 31, "r_stop": 32}, "horizonte fora do intervalo"),  # K = 2
        ({"a_stop": 31, "r_stop": 20}, "rival em outro stint"),  # já parou: no stint 2 na volta N
        ({"a_stop": 55, "r_stop": 59}, "corrida acaba antes"),  # a volta final seria a 61
    ],
)
def test_each_reason_to_skip_a_stop_is_counted(kwargs, reason):
    rows = chase_field(world=FLAT, **kwargs)

    events, skipped = find_events(rows, TOTAL, race="X")

    assert not [e for e in events if (e.attacker, e.rival) == ("A", "R")], kwargs
    assert skipped[reason] >= 1, (kwargs, dict(skipped))


def test_the_last_lap_of_the_window_may_be_the_last_of_the_race():
    """A volta final é a s_r + 2: com s_r = 58 ela é a 60 (a última), e o evento vale."""
    events, _ = find_events(chase_field(a_stop=54, r_stop=58, world=FLAT), TOTAL, race="X")

    assert [(e.attacker, e.end_lap) for e in events if e.attacker == "A"] == [("A", 60)]


def test_an_early_stop_has_no_data_to_be_assessed():
    _, skipped = find_events(chase_field(a_stop=5, r_stop=9, world=FLAT), TOTAL, race="X")

    assert skipped["volta inicial"] >= 1


def test_a_second_stop_of_either_car_inside_the_window_discards_the_event():
    """A janela é N+1 até a volta final: uma 2ª parada do atacante ou do rival muda o resultado por
    algo que o undercut não explica."""
    rows = real_race()
    for driver, lap in (("A", N + 4), ("R", N + K + 1)):
        tampered = [
            replace(r, pit_in=True) if (r.driver, r.lap_number) == (driver, lap) else r
            for r in rows
        ]

        events, skipped = find_events(tampered, TOTAL, race="X")

        assert not [e for e in events if e.attacker == "A"], driver
        assert skipped["parada extra na janela"] >= 1, driver


def test_the_leader_has_no_rival_to_attack():
    rows = chase_field(a_stop=999, r_stop=31, world=FLAT)  # R, o líder, para; ninguém à frente dele

    _, skipped = find_events(rows, TOTAL, race="X")

    assert skipped["sem rival à frente"] >= 1


# ------------------------------------------------------------------ o resultado real


def test_the_realized_margin_is_the_difference_of_the_clocks_at_the_final_lap():
    rows = real_race()
    event = only_event(rows)
    clock = {(r.driver, r.lap_number): r.session_time_ms for r in rows}

    assert realized_margin(rows, event) == pytest.approx(
        (clock[("R", event.end_lap)] - clock[("A", event.end_lap)]) / 1000
    )
    assert realized_margin(rows, event) == pytest.approx(8.4, abs=1e-3)  # a conta de papel


def test_a_missing_clock_at_the_final_lap_has_no_realized_margin():
    rows = real_race()
    event = only_event(rows)
    without = [r for r in rows if not (r.driver == "R" and r.lap_number == event.end_lap)]

    assert realized_margin(without, event) is None


@pytest.mark.parametrize(
    ("status_laps", "neutralized"),
    [
        ({}, False),
        ({N - 1: "4"}, True),  # a janela começa na volta N-1
        ({N + 3: "6"}, True),
        ({N + K + 2: "2"}, True),  # inclusive a volta final
        ({N - 2: "4"}, False),  # antes da janela
        ({N + K + 3: "4"}, False),  # depois dela
    ],
)
def test_a_stop_is_neutralized_only_by_a_non_green_lap_inside_the_window(status_laps, neutralized):
    rows = real_race(world=replace(FLAT, status_laps=status_laps))

    assert is_neutralized(rows, only_event(rows)) is neutralized


# ------------------------------------------------------------------ avaliar um evento


def test_in_the_flat_world_the_prediction_is_the_realized_margin():
    """No mundo plano, sem ruído, e com o horizonte e os compostos reais: previsto = realizado =
    8,4."""
    rows = real_race()

    (result,), skipped = evaluate_race(rows, TOTAL, race="X", priors=EXACT)

    assert result.assessment.margin == pytest.approx(8.4, abs=1e-3)
    assert result.realized == pytest.approx(8.4, abs=1e-3)
    assert result.error == pytest.approx(0.0, abs=1e-3)
    assert result.success and result.primary and not result.neutralized
    assert result.probability() == pytest.approx(result.assessment.probability)
    assert sum(skipped.values()) == 13


def test_a_gap_the_new_tyres_cannot_close_is_a_failure_and_the_model_says_so():
    rows = real_race(gap=20.0)

    (result,), _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT)

    assert result.realized == pytest.approx(-6.6, abs=1e-3)
    assert not result.success
    assert result.assessment.margin == pytest.approx(-6.6, abs=1e-3)
    assert result.assessment.probability < 0.2


def test_the_live_mode_uses_the_expected_horizon_and_not_the_one_that_happened():
    """Ao vivo ninguém sabe que o rival para daqui a 5 voltas: o horizonte esperado (os stints
    concluídos de MEDIUM têm ~18 voltas de idade, e R já tem 30: o piso, 3). A margem de 3 voltas é
    1,5 s e a do horizonte real, 8,4 s: é o custo de não saber."""
    rows = real_race()

    (oracle,), _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT, oracle=True)
    (live,), _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT, oracle=False)

    assert oracle.assessment.horizon_k == K and live.assessment.horizon_k == 3
    assert oracle.assessment.margin == pytest.approx(8.4, abs=1e-3)
    assert live.assessment.margin == pytest.approx(1.5, abs=1e-3)
    assert live.realized == oracle.realized  # o resultado real é o mesmo nos dois modos


def test_the_assessment_of_an_event_only_uses_what_was_known_when_the_attacker_crossed_the_line():
    """Sem vazamento: a avaliação é igual à de uma corrida que PAROU no instante em que o atacante
    completou a volta N (o resultado real, do futuro, só serve de gabarito). No modo oráculo o
    horizonte e os compostos são informação futura por definição, e vêm do evento, não das
    voltas."""
    rows = real_race()
    (result,), _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT)
    t = driver_crossing_ms(rows, "A", N)
    stopped_at_t = [r for r in rows if r.session_time_ms <= t]

    ctx = build_context(stopped_at_t, upto_lap=N, total_laps=TOTAL, priors=EXACT, as_of_ms=t)
    direct = assess_undercut(
        ctx, "A", "R", new_compound="HARD", rival_new_compound="HARD", horizon_k=K
    )

    assert result.assessment == direct


def test_the_lap_count_snapshot_is_still_available_as_the_leakage_diagnostic():
    """O estado antigo (cada piloto na volta N dele) existe para medir o quanto ele flatearia o
    backtest; no mundo plano e sem ruído os dois estados dão a mesma margem do par."""
    rows = real_race()

    (instant,), _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT, snapshot="tempo")
    (by_laps,), _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT, snapshot="volta")

    assert instant.assessment.margin == pytest.approx(by_laps.assessment.margin, abs=0.05)
    assert instant.assessment.margin == pytest.approx(8.4, abs=0.05)


def test_a_neutralized_stop_is_evaluated_but_is_not_primary():
    rows = real_race(world=replace(FLAT, status_laps={N + 2: "4"}))

    (result,), _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT)

    assert result.neutralized and not result.primary
    assert summarize([result]).n_primary == 0
    assert summarize([result]).excluded == {"pista não verde na janela": 1}


def test_an_event_the_model_refuses_to_alert_is_not_primary_and_says_why():
    """Com a degradação suave (0,03 s/volta) o pneu novo não repõe a parada nas 14 voltas que sobram
    na volta 46: o modelo não alertaria, e o evento não conta para precisão nem calibração."""
    mild = replace(FLAT, slopes={"SOFT": 0.03, "MEDIUM": 0.03, "HARD": 0.0})
    rows = chase_field(a_stop=47, r_stop=51, gap=5.0, world=mild)

    results, _ = evaluate_race(rows, TOTAL, race="X", priors=EXACT)
    (late,) = [r for r in results if r.event.attacker == "A"]

    assert not late.primary
    assert "poucas voltas para recuperar a perda" in summarize([late]).excluded


# ------------------------------------------------------------------ as métricas, à mão


def test_the_brier_score_is_the_mean_squared_error_of_the_probability():
    probabilities = [0.9, 0.8, 0.7, 0.2, 0.1]
    outcomes = [True, True, False, False, False]

    assert brier_score(probabilities, outcomes) == pytest.approx(
        (0.1**2 + 0.2**2 + 0.7**2 + 0.2**2 + 0.1**2) / 5
    )
    assert brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)  # dizer sempre 50%
    assert brier_score([], []) is None


def test_the_reliability_curve_compares_the_predicted_to_the_observed_by_bin():
    probabilities = [0.1, 0.2, 0.7, 0.8, 0.9, 1.0]
    outcomes = [False, False, False, True, True, True]

    bins = reliability_curve(probabilities, outcomes)

    by_range = {(b.low, b.high): b for b in bins}
    assert by_range[(0.0, 0.2)].n == 1 and by_range[(0.0, 0.2)].observed_rate == 0.0
    assert by_range[(0.2, 0.4)].n == 1  # 0,2 cai na faixa de cima: o limite inferior é fechado
    assert by_range[(0.4, 0.5)].n == 0 and by_range[(0.4, 0.5)].mean_predicted is None
    assert (by_range[(0.6, 0.8)].n, by_range[(0.6, 0.8)].observed_rate) == (1, 0.0)
    top = by_range[(0.8, 1.0)]  # a última faixa inclui o 1,0
    assert top.n == 3 and top.mean_predicted == pytest.approx(0.9) and top.observed_rate == 1.0
    assert sum(b.n for b in bins) == len(probabilities)


def test_the_precision_at_a_threshold_is_the_share_of_alerts_that_worked():
    """O limiar é inclusivo: uma probabilidade de exatamente 0,6 alerta (o alerta liga em `>=`)."""
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.55, 0.2]
    outcomes = [True, True, False, True, True, False]

    assert precision_at(probabilities, outcomes, 0.6) == (4, pytest.approx(3 / 4))
    assert precision_at(probabilities, outcomes, 0.5) == (5, pytest.approx(4 / 5))
    assert precision_at(probabilities, outcomes, 0.95) == (0, None)


# ------------------------------------------------------------------ o fator de inflação


def fake_result(
    margin: float,
    error: float,
    *,
    race: str = "X",
    raw_sd: float = 1.0,
    used: float = 1.3,
    neutralized: bool = False,
    reasons: tuple[str, ...] = (),
    gap: float = 3.0,
    probability: float | None = None,
) -> EventResult:
    if probability is None:
        probability = normal_cdf(margin / (raw_sd * used))
    event = UndercutEvent(race, "A", "R", N, N + K, "HARD", "HARD", gap)
    assessment = UndercutAssessment(
        "A", "R", N, margin, margin, raw_sd * used, probability, K, 21.0, None, None, None, 6,
        reasons=reasons,
    )  # fmt: skip
    return EventResult(event, assessment, margin - error, neutralized, used)


def normal_errors(scale: float, n: int = 400) -> list[float]:
    """`n` erros com os quantis de uma normal de desvio `scale` (determinístico)."""
    return [NormalDist(0, scale).inv_cdf((i + 0.5) / n) for i in range(n)]


def test_the_inflation_factor_is_what_makes_the_ninety_percent_interval_cover_ninety_percent():
    """Erros de desvio 2 contra um desvio declarado de 1 (cru): o fator que cobre 90% é ~2."""
    results = [fake_result(5.0, e) for e in normal_errors(2.0)]

    fit = calibrate_inflation(results)

    assert fit.raw_factor == pytest.approx(2.0, rel=0.02)
    assert fit.factor == fit.raw_factor and fit.n == 400
    # com o fator em uso (1,3) a cobertura do intervalo de 90% era bem menor
    assert fit.coverage_before == pytest.approx(0.715, abs=0.03)  # P(|z| <= 1,645*1,3/2)


def test_the_inflation_factor_never_deflates_below_the_floor():
    """Se o desvio declarado já é grande demais (erros de 0,4 contra 1), os dados pedem 0,4, mas é
    um
    fator de INFLAÇÃO: o piso 1,0 fica, e o pedido real vai ao relatório."""
    results = [fake_result(5.0, e) for e in normal_errors(0.4)]

    fit = calibrate_inflation(results)

    assert fit.raw_factor == pytest.approx(0.4, rel=0.03)
    assert fit.factor == 1.0
    assert calibrate_inflation(results, floor=0.0).factor == pytest.approx(0.4, rel=0.03)


def test_only_primary_events_enter_the_inflation_calibration():
    good = [fake_result(5.0, e) for e in normal_errors(2.0, 100)]
    junk = [
        fake_result(5.0, 50.0, neutralized=True),
        fake_result(5.0, 50.0, reasons=("o rival já parou",)),
    ]

    assert calibrate_inflation(good + junk).raw_factor == calibrate_inflation(good).raw_factor
    assert calibrate_inflation([]).raw_factor is None and calibrate_inflation([]).factor == 1.0


def test_the_inflation_of_each_race_is_calibrated_only_on_the_others():
    """Calibrar no mesmo conjunto em que se mede deixaria o fator escolher a resposta."""
    wild = [fake_result(5.0, e, race="WILD") for e in normal_errors(3.0, 200)]
    calm = [fake_result(5.0, e, race="CALM") for e in normal_errors(0.5, 200)]

    fits = loo_inflation({"WILD": wild, "CALM": calm})

    assert fits["CALM"].raw_factor == pytest.approx(3.0, rel=0.03)  # só a WILD
    assert fits["WILD"].raw_factor == pytest.approx(0.5, rel=0.03)  # só a CALM
    assert fits["WILD"].factor == 1.0  # e o piso


def test_a_probability_can_be_redone_with_another_inflation_without_reevaluating():
    r = fake_result(3.0, 0.0, raw_sd=1.0, used=1.3, probability=normal_cdf(3.0 / 1.3))

    assert r.probability() == pytest.approx(normal_cdf(3.0 / 1.3))
    assert r.probability(2.0) == pytest.approx(normal_cdf(3.0 / 2.0))
    assert r.probability(2.0) < r.probability(1.0)


# ------------------------------------------------------------------ o resumo


def fake_set() -> list[EventResult]:
    """Seis eventos primários com probabilidade prevista e resultado conhecidos: o resultado real é
    +1 s (sucesso) ou -1 s (fracasso), e a margem é a que dá a probabilidade pedida."""
    cases = [(0.9, True), (0.8, True), (0.7, False), (0.55, True), (0.3, False), (0.1, False)]
    out = []
    for p, win in cases:
        margin = NormalDist().inv_cdf(p) * 1.3  # raw_sd 1 e fator 1,3: Phi(margem / 1,3) = p
        out.append(fake_result(margin, margin - (1.0 if win else -1.0), probability=p))
    return out


def test_the_summary_has_the_base_rate_the_brier_and_the_precision_of_the_alerts():
    probabilities = [0.9, 0.8, 0.7, 0.55, 0.3, 0.1]
    wins = [True, True, False, True, False, False]

    summary = summarize(fake_set(), threshold=0.6)

    assert summary.n_events == summary.n_primary == 6
    assert summary.base_rate == pytest.approx(0.5)
    assert summary.mean_predicted == pytest.approx(np.mean(probabilities))
    assert summary.brier == pytest.approx(brier_score(probabilities, wins))
    assert summary.brier < summary.brier_baseline == pytest.approx(0.25)  # bate o "taxa-base"
    assert summary.alerts == 3  # 0,9, 0,8 e 0,7; o 0,55 fica abaixo do limiar
    assert summary.precision == pytest.approx(2 / 3)  # só o de 0,7 falhou
    assert sum(b.n for b in summary.bins) == 6


def test_the_summary_can_look_only_at_the_attackers_in_contention():
    near = [fake_result(4.0, 0.0, gap=2.0, probability=normal_cdf(4.0 / 1.3))]
    far = [fake_result(4.0, 6.0, gap=30.0, probability=normal_cdf(4.0 / 1.3))]

    assert summarize(near + far).n_primary == 2
    close = summarize(near + far, max_gap_s=5.0)
    assert close.n_primary == 1 and close.base_rate == 1.0


def test_the_summary_with_per_race_factors_uses_them_in_the_probability_and_the_coverage():
    results = [fake_result(3.0, e, race="X") for e in normal_errors(1.0, 100)]

    in_use = summarize(results)
    wide = summarize(results, factors={"X": 3.0})

    assert wide.mean_predicted < in_use.mean_predicted  # mais desvio, mais perto de 0,5
    assert in_use.coverage_90 == pytest.approx(0.968, abs=0.02)  # |e| <= 1,645 * 1,3
    assert wide.coverage_90 == pytest.approx(1.0, abs=0.01)  # |e| <= 1,645 * 3


def test_the_summary_counts_what_it_leaves_out_by_reason():
    results = [
        fake_result(3.0, 0.0, probability=0.9),
        fake_result(3.0, 0.0, neutralized=True, probability=0.9),
        fake_result(3.0, 0.0, reasons=("o rival já parou",)),
        fake_result(3.0, 0.0, reasons=("poucas voltas para recuperar a perda", "pista não verde")),
    ]

    summary = summarize(results)

    assert summary.n_events == 4 and summary.n_primary == 1
    assert summary.excluded == {
        "pista não verde na janela": 1,
        "o rival já parou": 1,
        "poucas voltas para recuperar a perda": 1,
        "pista não verde": 1,
    }


# ------------------------------------------------------------------ a população


def stat(label: str, round_: int, pit_loss: float) -> RaceStats:
    return RaceStats(
        label=label,
        circuit=label,
        year=2024,
        round=round_,
        tau=0.02,
        between_compounds=0.02,
        pit_loss=pit_loss,
        pit_loss_sd=1.0,
        n_green_stops=8,
    )


def test_each_race_is_evaluated_with_the_constants_computed_without_it(monkeypatch):
    refs = [
        RaceRef(i, f"R{i}", TOTAL, RaceKind.SECA, 1.0, year=2024, round=i, circuit=f"R{i}")
        for i in (1, 2, 3, 4)
    ]
    stats = [stat(f"R{i}", i, loss) for i, loss in ((1, 20.0), (2, 22.0), (3, 24.0), (4, 40.0))]
    seen = {}
    rows = real_race()

    def spy(rows_, total, *, race, priors, **kwargs):
        seen[race] = priors
        return [], Counter()

    monkeypatch.setattr(undercut_eval, "split_population", lambda session, year=None: (refs, []))
    monkeypatch.setattr(undercut_eval, "load_race_stats", lambda session, year=None: stats)
    monkeypatch.setattr(undercut_eval, "load_race_laps", lambda session, race_id: rows)
    monkeypatch.setattr(undercut_eval, "evaluate_race", spy)

    by_race, _ = evaluate_population(object())

    assert set(by_race) == {"R1", "R2", "R3", "R4"}
    for label, priors in seen.items():
        assert f"sem {label}" in priors.source
    # a reserva de cada corrida é a mediana das OUTRAS três: a R4 (40 s) não entra na dela
    assert seen["R4"].pit_loss.mean == pytest.approx(22.0)  # mediana de (20, 22, 24)
    assert seen["R1"].pit_loss.mean == pytest.approx(24.0)  # mediana de (22, 24, 40)


def test_rounds_limit_the_races_evaluated_but_not_the_pool_of_constants(monkeypatch):
    refs = [
        RaceRef(i, f"R{i}", TOTAL, RaceKind.SECA, 1.0, year=2024, round=i, circuit=f"R{i}")
        for i in (1, 2, 3, 4)
    ]
    stats = [stat(f"R{i}", i, 20.0) for i in (1, 2, 3, 4)]
    seen = {}

    def spy(rows_, total, *, race, priors, **kwargs):
        seen[race] = priors
        return [], Counter()

    monkeypatch.setattr(undercut_eval, "split_population", lambda session, year=None: (refs, []))
    monkeypatch.setattr(undercut_eval, "load_race_stats", lambda session, year=None: stats)
    monkeypatch.setattr(undercut_eval, "load_race_laps", lambda session, race_id: [])
    monkeypatch.setattr(undercut_eval, "evaluate_race", spy)

    evaluate_population(object(), rounds=[3])

    assert list(seen) == ["R3"] and "3 corridas" in seen["R3"].source


# ------------------------------------------------------------------ o relatório e a CLI


def population_results():
    results = {}
    for label, gap in (("R1", 3.0), ("R2", 5.0), ("R3", 20.0)):
        res, _ = evaluate_race(real_race(gap=gap), TOTAL, race=label, priors=EXACT)
        results[label] = res
    return results, Counter({"rival não parou": 4})


def test_the_text_report_has_the_protocol_numbers_and_the_calibration():
    by_race, skipped = population_results()

    text = build_report(by_race, skipped, oracle=True)

    assert "oráculo" in text and "rival não parou: 4" in text
    assert "Brier" in text and "faixa de probabilidade" in text and "R3" in text


def test_the_markdown_report_states_the_selection_bias_and_the_live_mode_label():
    by_race, skipped = population_results()

    md = build_report(by_race, skipped, oracle=False, markdown=True)
    flat = " ".join(md.split())  # o texto fixo quebra linhas

    assert "ao vivo" in md
    assert "equipes tendem a parar quando funciona" in flat
    assert "revocação não é mensurável" in flat and "OUTRAS corridas" in flat
    assert "| R1 |" in md


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(undercut_eval, "evaluate_population", lambda *a, **k: ({}, Counter()))
    monkeypatch.setattr(undercut_eval, "get_session_factory", lambda: sessionmaker())

    assert undercut_eval.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path):
    by_race, skipped = population_results()
    calls = {}

    def fake(session, **kwargs):
        calls.update(kwargs)
        return by_race, skipped

    monkeypatch.setattr(undercut_eval, "evaluate_population", fake)
    monkeypatch.setattr(undercut_eval, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "avaliacao.md"

    assert undercut_eval.main(["--round", "6", "--live", "--markdown", str(target)]) == 0

    assert calls["rounds"] == [6] and calls["oracle"] is False
    assert "Brier" in capsys.readouterr().out
    assert "# Avaliação do alerta de undercut" in target.read_text(encoding="utf-8")


def test_the_threshold_of_the_cli_reaches_the_config(monkeypatch):
    seen = {}

    def fake(session, **kwargs):
        seen.update(kwargs)
        return {}, Counter()

    monkeypatch.setattr(undercut_eval, "evaluate_population", fake)
    monkeypatch.setattr(undercut_eval, "get_session_factory", lambda: sessionmaker())

    undercut_eval.main(["--threshold", "0.7"])

    assert isinstance(seen["config"], UndercutConfig) and seen["config"].threshold == 0.7
