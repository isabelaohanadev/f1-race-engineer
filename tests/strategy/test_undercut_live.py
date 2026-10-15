"""A avaliação da forma ao vivo do alerta (strategy/undercut_live.py).

Corridas sintéticas de resposta conhecida: com um carro `L` na frente, o rival `R` e o atacante `A`
andam a 88 s e `A` está 5 s atrás de `R`. O líder `L` completa as voltas antes dos dois, então no
instante dele os dois estão na volta anterior (o que a forma ao vivo tem de refletir). Sem `L`, o
rival é o próprio líder, que a forma ao vivo nunca avalia.
"""

from dataclasses import replace

import pytest

from strategy.fuel import physical_coefficient
from strategy.live import FIRST_LAP, leader_lap_assessments
from strategy.priors import DEFAULT_PRIORS
from strategy.protocol import FROZEN, frozen_undercut_config
from strategy.snapshot import crossing_ms, driver_crossing_ms
from strategy.undercut import AlertTracker, assess_undercut, build_context
from strategy.undercut_eval import find_events
from strategy.undercut_live import (
    DIFFERENT_PAIR,
    LATE,
    NO_ATTACKER,
    NO_RESULT,
    NO_RIVAL,
    PRIMARY,
    RIVAL_IS_LEADER,
    LeaderLap,
    RaceLive,
    build_report,
    evaluate_race_live,
    leader_laps,
    map_event,
    next_result,
    old_form,
    result_in_effect,
)
from tests.strategy.synthetic import Plan, World, chase_field

TOTAL, N, K = 45, 30, 5
FIRST = 25  # o fixture só calcula as voltas do líder a partir daqui (as de antes não importam)
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)
LEADER = Plan(
    "L", base=86.0
)  # 2 s por volta mais rápido que R e A, mas menos de uma volta à frente
CONFIG = frozen_undercut_config()


def race_with_leader():
    return chase_field(a_stop=N + 1, r_stop=N + K, gap=5.0, world=FLAT, extra=(LEADER,))


def race_without_leader():
    return chase_field(a_stop=N + 1, r_stop=N + K, gap=5.0, world=FLAT)


@pytest.fixture(scope="module")
def rows():
    return race_with_leader()


@pytest.fixture(scope="module")
def laps(rows):
    return leader_laps(rows, TOTAL, priors=DEFAULT_PRIORS, config=CONFIG, first_lap=FIRST)


@pytest.fixture(scope="module")
def crossings(laps):
    return {lap: live.t_ms for lap, live in laps.items()}


@pytest.fixture(scope="module")
def event(rows):
    events, _ = find_events(rows, TOTAL, race="X", config=CONFIG)
    (only,) = [e for e in events if (e.attacker, e.rival) == ("A", "R")]
    return only


def mapped(rows, laps, crossings, event, decision=PRIMARY):
    return map_event(event, rows, laps, crossings, decision=decision, inflation=1.3)


# ================================================================ qual resultado estava em vigor


def test_the_result_in_effect_is_the_latest_leader_lap_at_or_before_the_instant():
    crossings = {5: 450, 6: 540, 7: 630}

    assert result_in_effect(crossings, 449) is None  # ainda não havia resultado
    assert result_in_effect(crossings, 450) == 5  # no próprio instante já vale
    assert result_in_effect(crossings, 539) == 5
    assert result_in_effect(crossings, 5_000) == 7


def test_the_late_decision_takes_the_next_result_only_if_it_comes_before_the_pit_entry():
    crossings = {5: 450, 6: 540, 7: 630}

    assert next_result(crossings, 460, deadline_ms=600) == 6
    assert (
        next_result(crossings, 460, deadline_ms=539) is None
    )  # sairia depois da entrada nos boxes
    assert next_result(crossings, 540, deadline_ms=700) == 7  # o instante igual não é "depois"


# ================================================================ do evento ao resultado ao vivo


def test_the_event_is_decided_with_the_result_in_force_when_the_attacker_crosses_the_line(
    rows, laps, crossings, event
):
    live = mapped(rows, laps, crossings, event)

    t_dec = driver_crossing_ms(rows, "A", N)
    assert live.result is not None and live.reason is None
    assert live.leader_lap == N  # o líder completou a volta 30 antes de A (5 s atrás de R)
    assert crossings[N] <= t_dec < crossings[N + 1]
    assert live.staleness == 1  # A ainda estava na volta 29 no estado do líder
    assert (live.result.assessment.attacker, live.result.assessment.rival) == ("A", "R")


def test_the_live_assessment_is_the_one_the_consumer_function_builds_at_the_leaders_instant(
    rows, laps, crossings, event
):
    live = mapped(rows, laps, crossings, event)

    direct = assess_undercut(
        build_context(
            rows, upto_lap=N, total_laps=TOTAL, priors=DEFAULT_PRIORS, config=CONFIG,
            as_of_ms=crossing_ms(rows, N),
        ),
        "A",
    )  # fmt: skip
    _, from_consumer = leader_lap_assessments(
        rows, lap=N, total_laps=TOTAL, t_ms=crossing_ms(rows, N), priors=DEFAULT_PRIORS
    )

    assert live.result is not None
    assert live.result.assessment.probability == direct.probability
    assert live.result.assessment == next(a for a in from_consumer if a.attacker == "A")


def test_the_realized_margin_is_the_one_of_the_event_not_of_the_live_instant(
    rows, laps, crossings, event
):
    live = mapped(rows, laps, crossings, event)

    assert live.result is not None and live.result.event == event
    assert live.result.realized == pytest.approx(
        (
            driver_crossing_ms(rows, "R", event.end_lap)
            - driver_crossing_ms(rows, "A", event.end_lap)
        )
        / 1000
    )


def test_the_late_decision_uses_the_next_leader_lap_where_the_attacker_is_up_to_date(
    rows, laps, crossings, event
):
    live = mapped(rows, laps, crossings, event, decision=LATE)

    assert live.result is not None and live.decision == LATE
    assert live.leader_lap == N + 1  # o resultado da volta 31 sai antes de A entrar nos boxes
    assert live.staleness == 0  # e nele A já completou a 30


def test_the_leader_is_never_the_rival_in_the_live_form():
    rows = race_without_leader()  # aqui R lidera
    events, _ = find_events(rows, TOTAL, race="X", config=CONFIG)
    (event,) = [e for e in events if (e.attacker, e.rival) == ("A", "R")]
    laps = leader_laps(rows, TOTAL, priors=DEFAULT_PRIORS, config=CONFIG, first_lap=FIRST)
    crossings = {lap: live.t_ms for lap, live in laps.items()}

    for decision in (PRIMARY, LATE):
        live = mapped(rows, laps, crossings, event, decision)
        assert live.result is None and live.reason == RIVAL_IS_LEADER, decision


def test_an_event_before_any_live_result_is_left_out_with_its_reason(rows, laps, crossings, event):
    early = replace(event, lap=FIRST - 8)  # antes de o consumidor calcular qualquer volta

    assert mapped(rows, laps, crossings, early).reason == NO_RESULT


def stub(laps: dict[int, LeaderLap], lap: int, **changes) -> dict[int, LeaderLap]:
    live = laps[lap]
    assessment = live.assessments["A"]
    rivals = changes.pop("rival", assessment.rival)
    new = replace(live, assessments={**live.assessments, "A": replace(assessment, rival=rivals)})
    if "drop" in changes:
        new = replace(new, assessments={k: v for k, v in new.assessments.items() if k != "A"})
    return {**laps, lap: new}


def test_a_different_pair_no_rival_and_a_missing_attacker_each_have_their_own_reason(
    rows, laps, crossings, event
):
    assert mapped(rows, stub(laps, N, rival="D03"), crossings, event).reason == DIFFERENT_PAIR
    assert mapped(rows, stub(laps, N, rival=None), crossings, event).reason == NO_RIVAL
    assert mapped(rows, stub(laps, N, drop=True), crossings, event).reason == NO_ATTACKER


def test_the_tracker_alerts_come_from_the_hysteresis_over_the_whole_sequence_of_leader_laps(
    rows, laps
):
    tracker = AlertTracker(CONFIG)
    expected = {}
    for lap in sorted(laps):
        _, assessments = leader_lap_assessments(
            rows,
            lap=lap,
            total_laps=TOTAL,
            t_ms=laps[lap].t_ms,
            priors=DEFAULT_PRIORS,
            config=CONFIG,
        )
        expected[lap] = frozenset(tracker.update_all(assessments))

    assert {lap: live.alerts for lap, live in laps.items()} == expected


def test_a_future_that_never_happened_for_the_live_instant_does_not_change_the_live_result(
    rows, laps, crossings, event
):
    """Sem vazamento: troca tudo o que acontece depois do instante do líder da volta 30 por
    absurdos, e a probabilidade ao vivo do evento é a mesma (não depende do futuro)."""
    t = crossings[N]
    broken = [
        replace(r, lap_time_ms=500_000, compound="HARD", tyre_life=99, track_status="4")
        if r.session_time_ms > t and r.lap_number > N
        else r
        for r in rows
    ]
    laps_broken = leader_laps(broken, TOTAL, priors=DEFAULT_PRIORS, config=CONFIG, first_lap=N)
    crossings_broken = {lap: live.t_ms for lap, live in laps_broken.items()}

    original = mapped(rows, laps, crossings, event)
    corrupted = mapped(broken, laps_broken, crossings_broken, event)

    assert original.result is not None and corrupted.result is not None
    assert original.result.assessment.probability == corrupted.result.assessment.probability
    assert original.result.assessment.margin == corrupted.result.assessment.margin


def test_the_whole_race_evaluation_covers_the_same_events_as_the_backtest_search():
    rows = race_with_leader()
    live = evaluate_race_live(rows, TOTAL, race="X", priors=DEFAULT_PRIORS)
    events, skipped = find_events(rows, TOTAL, race="X", config=CONFIG)

    assert [e.event for e in live.events[PRIMARY]] == events
    assert [e.event for e in live.events[LATE]] == events
    assert live.skipped == dict(skipped)
    assert live.n_leader_laps == TOTAL - FIRST_LAP  # da volta 5 à penúltima, uma por volta


# ================================================================ o relatório


@pytest.fixture(scope="module")
def race_live(rows, laps, crossings, event):
    events_found, skipped = find_events(rows, TOTAL, race="X", config=CONFIG)
    return RaceLive(
        "X",
        {d: [mapped(rows, laps, crossings, e, d) for e in events_found] for d in (PRIMARY, LATE)},
        dict(skipped),
        len(laps),
    )


def flat(text: str) -> str:
    return " ".join(text.split())


def test_the_report_states_the_protocol_the_bias_the_n_per_bin_and_wilson(race_live):
    text = flat(build_report([race_live], {}))

    assert "**forma antiga**" in text and "**forma ao vivo**" in text
    assert "os números do backtest não valem para a ao vivo" in text
    assert "Wilson" in text and "eventos (n)" in text  # o n de cada faixa
    assert "**precisão é otimista**" in text and "revocação não é mensurável" in text
    assert "Uma seleção a mais" in text and "nenhum undercut contra o líder" in text
    assert "Diagnóstico, não aplicado" in text


def test_the_report_says_no_post_result_adjustment_unless_one_is_given(race_live):
    plain = build_report([race_live], {})
    adjusted = build_report([race_live], {}, adjustments=("o fator passou a 1,5",))

    assert "## Ajustes pós-resultado\n\nNenhum." in plain
    assert "- o fator passou a 1,5" in adjusted and "Nenhum." not in adjusted.split("## Ajustes")[1]


def test_the_report_keeps_the_frozen_threshold_and_inflation_and_never_quotes_the_old_precision(
    race_live,
):
    text = build_report([race_live], {})

    assert str(FROZEN.alert_threshold) in text and str(FROZEN.undercut_sd_inflation) in text
    assert "83%" not in text  # a precisão da forma antiga fica só no relatório do backtest


def test_old_form_indexes_the_backtest_results_by_race_attacker_and_lap(race_live):
    from strategy.backtest import LIVE, TEMPO, RaceOutcome

    result = next(e.result for e in race_live.events[PRIMARY] if e.result is not None)
    outcome = RaceOutcome("X", "seca", True, TOTAL, undercut={(LIVE, TEMPO): [result]})
    wet = RaceOutcome("Y", "molhada", False, TOTAL, undercut={(LIVE, TEMPO): [result]})

    index = old_form([outcome, wet])

    assert index == {("X", "A", N): result}  # só as secas do backtest principal


def test_the_event_says_whether_the_tracker_had_the_pair_switched_on_at_that_result(
    rows, laps, crossings, event
):
    on = {**laps, N: replace(laps[N], alerts=frozenset({("A", "R")}))}
    off = {**laps, N: replace(laps[N], alerts=frozenset({("A", "D00")}))}

    assert mapped(rows, on, crossings, event).tracker_alert is True
    assert mapped(rows, off, crossings, event).tracker_alert is False


# ================================================================ a comparação com a forma antiga


def pair_with(race: str, old_p: float, live_p: float, realized: float, base):
    """(antiga, ao vivo) de um evento com as probabilidades dadas e o resultado real dado."""
    from strategy.undercut_eval import EventResult

    event = replace(base.event, race=race)

    def with_probability(p: float):
        # desvio 0: `EventResult.probability()` devolve a do `assessment`, sem refazer a conta
        assessment = replace(base.assessment, probability=p, margin_sd=0.0)
        return EventResult(event, assessment, realized, False, 1.3)

    return with_probability(old_p), with_probability(live_p)


@pytest.fixture(scope="module")
def base_result(race_live):
    return next(e.result for e in race_live.events[PRIMARY] if e.result is not None)


def test_the_brier_difference_is_negative_when_the_live_form_errs_less(base_result):
    from strategy.undercut_live import brier_difference

    # deu certo (realized > 0): a ao vivo diz 0,9 e a antiga 0,6, em 5 corridas
    pairs = [pair_with(f"R{i}", 0.6, 0.9, 3.0, base_result) for i in range(5)]

    difference = brier_difference(pairs)

    assert difference is not None and difference.mean == pytest.approx(0.01 - 0.16)
    assert difference.high < 0 and difference.n_races == 5


def test_the_comparison_verdict_follows_the_confidence_interval_not_the_point(base_result):
    from strategy.undercut_live import _verdict, brier_difference

    better = brier_difference([pair_with(f"R{i}", 0.6, 0.9, 3.0, base_result) for i in range(5)])
    worse = brier_difference([pair_with(f"R{i}", 0.9, 0.6, 3.0, base_result) for i in range(5)])
    mixed = brier_difference(
        [pair_with(f"R{i}", 0.6, 0.9 if i % 2 else 0.3, 3.0, base_result) for i in range(6)]
    )

    assert _verdict(better) == "a forma ao vivo erra menos que a antiga"
    assert _verdict(worse) == "a forma ao vivo erra mais que a antiga"
    assert _verdict(mixed) == "não se distingue da antiga nesta amostra"
    assert _verdict(None) == "sem dados"


def test_common_pairs_keep_only_the_events_that_are_primary_in_both_forms(race_live, base_result):
    from strategy.undercut_live import common_pairs

    key = (base_result.event.race, base_result.event.attacker, base_result.event.lap)
    not_primary = replace(base_result, neutralized=True)

    assert common_pairs([race_live], {key: base_result}) == [
        (base_result, next(e.result for e in race_live.events[PRIMARY] if e.result is not None))
    ]
    assert common_pairs([race_live], {key: not_primary}) == []  # a antiga não é primária
    assert common_pairs([race_live], {}) == []  # o evento não está na forma antiga


def test_the_summary_is_generated_from_the_numbers_and_says_what_is_left_out(race_live):
    text = flat(build_report([race_live], {}))

    assert "## Resumo" in text
    assert "Nenhum undercut contra o líder é avaliado nem alertado ao vivo" in text
    assert "A precisão é otimista" in text and "não aplicado" in text


def test_post_result_notes_are_listed_apart_from_the_adjustments(race_live):
    text = flat(build_report([race_live], {}, notes=("acrescentei o resumo gerado",)))

    assert "NÃO alteram o protocolo" in text and "- acrescentei o resumo gerado" in text
    assert "Nenhum. O limiar e o fator de inflação são os congelados" in text
