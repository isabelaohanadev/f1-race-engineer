"""O estado ao vivo e o cálculo de cada volta (strategy/live.py e strategy/payload.py).

O que mais importa aqui, e que os testes prendem:
* a MESMA função do backtest: o resultado ao vivo é numericamente o do backtest;
* ordem e repetição: a mesma corrida com eventos embaralhados (dentro da janela) ou repetidos dá o
  mesmo resultado;
* nada do futuro: o que chega depois do instante do líder não entra no resultado dele.
"""

import json
import random
from dataclasses import replace

import pytest

from replay.events import LapCompletedEvent, RaceStatusEvent, StreamEntry
from strategy.live import (
    FIRST_LAP,
    Computation,
    Constants,
    Outcome,
    RaceTracker,
    Trigger,
    computation_job,
    compute,
)
from strategy.montecarlo import prepare, simulate
from strategy.payload import (
    BASELINE_NAME,
    EXPERIMENTAL,
    INDICATIVE,
    SCHEMA_VERSION,
    build_payload,
    dumps,
    reliability,
    stable,
    track_warnings,
)
from strategy.priors import DEFAULT_PRIORS
from strategy.protocol import frozen_sim_config
from strategy.snapshot import crossing_ms
from strategy.undercut import assess_all, build_context
from tests.strategy.synthetic import World, chase_field, stream_entries
from tests.strategy.test_montecarlo import sim_priors

TOTAL = 60
CONSTANTS = Constants(DEFAULT_PRIORS, sim_priors())
CONFIG = frozen_sim_config(n_sims=300)
WORLD = World(total_laps=TOTAL)
WAIT = 0.3


def entry(
    event, entry_id: int = 1, epoch: int = 1
) -> StreamEntry:  # o ID só precisa ser único e crescente
    return StreamEntry(f"{entry_id}-0", epoch, event)


def lap_entry(driver: str, lap: int, clock: int, *, n: int = 1, epoch: int = 1) -> StreamEntry:
    """Uma volta mínima: o relógio da sessão decide o que se sabe num instante."""
    return entry(
        LapCompletedEvent(driver, lap, clock, 90_000, "MEDIUM", None, session_time_ms=clock),
        n,
        epoch,
    )


def started(total: int = 20, epoch: int = 1) -> StreamEntry:
    return entry(RaceStatusEvent("started", 1, 0, total), 0, epoch)


def tracker(**kwargs) -> RaceTracker:
    t = RaceTracker(2024, 1, wait_s=WAIT, **kwargs)
    t.apply(started(), now=0.0)
    return t


# ================================================================ a janela e o disparo


def test_the_first_entry_of_a_lap_schedules_the_calculation_after_the_window():
    t = tracker()

    applied = t.apply(lap_entry("VER", 5, 450_000), now=10.0)

    assert applied.outcome is Outcome.APPLIED and applied.scheduled_lap == 5
    assert t.next_deadline == pytest.approx(10.0 + WAIT)
    assert t.due(10.0 + WAIT - 0.01) == []  # a janela ainda não fechou
    (trigger,) = t.due(10.0 + WAIT)
    assert (trigger.lap, trigger.leader_t_ms, trigger.wait_s) == (5, 450_000, WAIT)
    assert t.due(99.0) == []  # disparou uma vez


def test_the_window_counts_from_the_first_entry_not_from_the_last():
    t = tracker()
    t.apply(lap_entry("VER", 5, 450_000, n=1), now=10.0)
    t.apply(lap_entry("NOR", 5, 453_000, n=2), now=10.2)  # chegou dentro da janela

    assert t.apply(lap_entry("HAM", 5, 456_000, n=3), now=10.25).scheduled_lap is None
    (trigger,) = t.due(10.3)  # 10,0 + janela: a chegada dos outros não a empurra
    assert {r.driver for r in trigger.rows} == {"VER", "NOR", "HAM"}


def test_laps_before_the_first_lap_do_not_schedule_anything():
    t = tracker()

    applied = t.apply(lap_entry("VER", FIRST_LAP - 1, 360_000), now=1.0)

    assert applied.scheduled_lap is None and t.due(99.0) == []


def test_the_trigger_only_advances():
    """Uma volta velha de um retardatário (ou fora de ordem) nunca dispara um resultado velho."""
    t = tracker()
    t.apply(lap_entry("VER", 8, 720_000), now=1.0)

    assert t.apply(lap_entry("SLO", 6, 560_000, n=2), now=1.1).scheduled_lap is None
    assert [x.lap for x in t.due(9.0)] == [8]


def test_two_laps_waiting_together_come_out_in_lap_order():
    t = tracker(first_lap=5)
    t.apply(lap_entry("VER", 5, 450_000), now=1.0)
    t.apply(lap_entry("VER", 6, 540_000, n=2), now=1.1)  # a janela da 5 ainda não tinha fechado

    assert [x.lap for x in t.due(9.0)] == [5, 6]


def test_the_last_lap_is_not_calculated_and_without_the_total_the_lap_waits():
    t = RaceTracker(2024, 1, wait_s=WAIT)
    t.apply(started(total=10), now=0.0)
    t.apply(lap_entry("VER", 10, 900_000), now=1.0)  # a volta final é o fim da corrida

    assert t.due(9.0) == []

    late_start = RaceTracker(2024, 1, wait_s=WAIT)  # o `started` chega depois da primeira volta
    late_start.apply(entry(LapCompletedEvent("VER", 6, 540_000, session_time_ms=540_000), 1), 1.0)
    assert late_start.due(9.0) == []  # sem o total, espera (não descarta)
    late_start.apply(started(total=20), now=9.5)
    assert [x.lap for x in late_start.due(10.0)] == [6]


def test_the_rows_of_a_trigger_are_sorted_like_the_backtest_receives_them():
    t = tracker()
    for i, (driver, lap) in enumerate([("VER", 6), ("ALB", 5), ("VER", 5), ("ALB", 6)]):
        t.apply(
            lap_entry(driver, lap, 90_000 * lap + (100 if driver == "VER" else 200), n=i + 1), 1.0
        )

    (trigger,) = [x for x in t.due(9.0) if x.lap == 6]

    assert [(r.driver, r.lap_number) for r in trigger.rows] == [
        ("ALB", 5), ("ALB", 6), ("VER", 5), ("VER", 6)
    ]  # fmt: skip


# ================================================================ duplicatas, epoch e atraso


def test_a_duplicate_is_counted_and_changes_nothing():
    t = tracker()
    first = t.apply(lap_entry("VER", 5, 450_000), now=1.0)
    again = t.apply(lap_entry("VER", 5, 450_000, n=2), now=1.05)

    assert (first.outcome, again.outcome) == (Outcome.APPLIED, Outcome.DUPLICATE)
    assert again.scheduled_lap is None and t.duplicates == 1 and len(t.rows) == 1


def test_a_duplicate_with_different_content_keeps_the_first_entry():
    t = tracker()
    t.apply(lap_entry("VER", 5, 450_000), now=1.0)
    changed = entry(
        LapCompletedEvent("VER", 5, 450_000, 77_000, "SOFT", session_time_ms=450_000), 2
    )

    assert t.apply(changed, now=1.1).outcome is Outcome.DUPLICATE
    assert t.rows[("VER", 5)].lap_time_ms == 90_000 and t.rows[("VER", 5)].compound == "MEDIUM"


def test_an_entry_of_an_old_replay_is_ignored_and_a_new_replay_resets_everything():
    t = tracker()
    t.apply(entry(RaceStatusEvent("started", 1, 0, 20), 0, epoch=2), now=1.0)  # replay novo
    assert t.epoch == 2 and not t.rows

    old = t.apply(lap_entry("VER", 5, 450_000, epoch=1), now=1.1)
    assert old.outcome is Outcome.STALE_EPOCH and not t.rows and t.next_deadline is None

    t.apply(lap_entry("VER", 5, 450_000, epoch=2), now=1.2)
    t.apply(lap_entry("VER", 5, 450_000, n=2, epoch=2), now=1.25)  # uma repetição neste replay
    assert t.duplicates == 1
    reset = t.apply(entry(RaceStatusEvent("started", 1, 0, 30), 0, epoch=3), now=1.3)
    assert reset.new_epoch and (t.epoch, t.total_laps) == (3, 30)
    assert not t.rows and t.next_deadline is None and t.last_trigger_lap == 0
    assert (t.duplicates, t.late_laps) == (0, 0)  # os contadores são do replay


def test_a_lap_that_arrives_after_the_window_is_late_kept_and_does_not_reopen_the_calculation():
    t = tracker()
    t.apply(lap_entry("VER", 5, 450_000), now=1.0)
    (trigger,) = t.due(2.0)  # o resultado da volta 5 sai com o que havia
    assert {r.driver for r in trigger.rows} == {"VER"}

    # um carro que cruzou a linha ANTES do líder, e cuja entrada só agora chegou
    late = t.apply(lap_entry("NOR", 4, 440_000, n=2), now=2.5)
    after = t.apply(lap_entry("NOR", 5, 453_000, n=3), now=2.6)  # normal: depois do instante

    assert late.outcome is Outcome.LATE and late.scheduled_lap is None
    assert after.outcome is Outcome.APPLIED
    assert t.late_laps == 1 and t.due(99.0) == []  # nada foi reaberto
    assert ("NOR", 4) in t.rows  # fica no estado, para as voltas seguintes
    t.apply(lap_entry("VER", 6, 540_000, n=4), now=3.0)
    (next_trigger,) = t.due(4.0)
    assert ("NOR", 4) in {(r.driver, r.lap_number) for r in next_trigger.rows}
    assert next_trigger.late_laps == 1


def test_a_row_at_exactly_the_fired_instant_is_late_too():
    """Dois carros no mesmo milissegundo: o segundo a chegar, depois do resultado, é atrasado."""
    t = tracker()
    t.apply(lap_entry("VER", 5, 450_000), now=1.0)
    t.due(2.0)

    assert t.apply(lap_entry("NOR", 5, 450_000, n=2), now=2.1).outcome is Outcome.LATE


def test_lateness_is_measured_against_the_newest_result_not_the_oldest():
    t = tracker()
    t.apply(lap_entry("VER", 5, 450_000), now=1.0)
    t.due(2.0)
    t.apply(lap_entry("VER", 6, 540_000, n=2), now=3.0)
    t.due(4.0)  # dois resultados já saíram: o mais novo é o da volta 6

    between = t.apply(lap_entry("NOR", 5, 500_000, n=3), now=4.5)  # depois da 5, antes da 6

    assert between.outcome is Outcome.LATE  # o resultado da volta 6 já tinha saído sem ela


def test_resume_after_a_rebuild_calculates_the_latest_lap_without_waiting():
    t = RaceTracker(2024, 1, wait_s=WAIT)
    t.apply(started(total=20), 0.0, schedule=False)
    for n, (driver, lap, clock) in enumerate(
        [("VER", 5, 450_000), ("VER", 6, 540_000), ("NOR", 6, 543_000)]
    ):
        applied = t.apply(lap_entry(driver, lap, clock, n=n + 1), 0.0, schedule=False)
        assert applied.scheduled_lap is None  # a reconstrução não agenda nada

    trigger = t.resume()

    assert trigger is not None and trigger.lap == 6 and trigger.leader_t_ms == 540_000
    assert t.last_trigger_lap == 6 and t.next_deadline is None
    assert t.apply(lap_entry("VER", 7, 630_000, n=9), now=1.0).scheduled_lap == 7  # segue dali


def test_resume_with_nothing_calculable_returns_nothing():
    t = tracker()
    t.apply(lap_entry("VER", 2, 180_000), 0.0, schedule=False)  # antes da primeira volta avaliada

    assert t.resume() is None


# ================================================================ a mesma função do backtest


@pytest.fixture(scope="module")
def race():
    return chase_field(world=WORLD)


@pytest.fixture(scope="module")
def entries(race):
    return stream_entries(race, total_laps=TOTAL)


def feed(
    entries_, lap_limit: int | None = None, *, wait_s: float = WAIT
) -> tuple[RaceTracker, list[Trigger]]:
    """Alimenta um rastreador com as entradas em ordem, 1 ms entre elas, e dispara quando a janela
    fecha. `lap_limit`: para depois da primeira entrada de uma volta acima do limite."""
    t = RaceTracker(2024, 1, wait_s=wait_s)
    triggers: list[Trigger] = []
    now = 0.0
    for e in entries_:
        if (
            lap_limit is not None
            and isinstance(e.event, LapCompletedEvent)
            and e.event.lap > lap_limit
        ):
            break
        now += 0.001
        t.apply(e, now)
        triggers += t.due(now)
    triggers += t.due(now + wait_s + 1.0)
    return t, triggers


def result_for(trigger: Trigger) -> Computation:
    computation = compute(trigger, CONSTANTS, sim_config=CONFIG)
    assert computation is not None
    return computation


def payload_of(c: Computation, **kwargs) -> dict:
    t = c.trigger
    defaults = dict(
        year=t.year, round_number=t.round, epoch=t.epoch, lap=t.lap, total_laps=t.total_laps,
        forecast=c.forecast, sim_input=c.sim_input, leader=c.leader, assessments=c.assessments,
        alerts=set(), warnings=c.warnings, cars_known=t.cars_known, late_laps=t.late_laps,
        duplicates=t.duplicates, wait_s=t.wait_s, computed_at_ms=0,
    )  # fmt: skip
    return build_payload(**(defaults | kwargs))


def test_the_live_result_is_numerically_the_backtest_result(race, entries):
    """O mesmo rows, a mesma `prepare` e `simulate`, os mesmos números: o consumidor ao vivo não
    tem um segundo caminho de cálculo."""
    _, triggers = feed(entries, lap_limit=30)
    live = result_for(next(x for x in triggers if x.lap == 30))

    # o backtest recebe as voltas como `load_race_laps` as devolve: por (piloto, volta)
    ordered = sorted(race, key=lambda r: (r.driver, r.lap_number))
    backtest_input = prepare(
        ordered, year=2024, round=1, upto_lap=30, total_laps=TOTAL,
        sim_priors=CONSTANTS.sim_priors, priors=CONSTANTS.priors, snapshot="tempo",
    )  # fmt: skip
    backtest = simulate(backtest_input, CONFIG)
    context = build_context(
        ordered, upto_lap=30, total_laps=TOTAL, priors=CONSTANTS.priors,
        as_of_ms=backtest_input.t_ms, config=None,
    )  # fmt: skip

    assert live.forecast.win_probabilities() == backtest.win_probabilities()
    assert [d.podium for d in live.forecast.drivers] == [d.podium for d in backtest.drivers]
    assert live.sim_input.t_ms == crossing_ms(race, 30)
    assert [a.probability for a in live.assessments] == [a.probability for a in assess_all(context)]


def test_events_after_the_leaders_instant_do_not_change_the_result(entries):
    """Sem vazamento do futuro: as voltas de depois do instante do líder estão no stream e no
    rastreador quando a janela fecha, e o resultado é o mesmo."""
    _, lean = feed(entries, lap_limit=30)
    t = RaceTracker(2024, 1, wait_s=WAIT)
    now = 0.0
    for e in entries:  # até a volta 33 inteira chegar ANTES de a janela da volta 30 fechar
        if isinstance(e.event, LapCompletedEvent) and e.event.lap > 33:
            break
        t.apply(e, now := now + 0.0001)
    (rich,) = [x for x in t.due(now + 1.0) if x.lap == 30]

    assert len(rich.rows) > len(next(x for x in lean if x.lap == 30).rows)
    plain, with_future = result_for(next(x for x in lean if x.lap == 30)), result_for(rich)
    assert plain.forecast.win_probabilities() == with_future.forecast.win_probabilities()


def window_group(entries, lap: int):
    """O que veio antes (em ordem) e as entradas depois do instante do líder na volta anterior, até
    um pouco depois do dele na volta `lap`: tudo o que um produtor fora de ordem poderia entregar
    dentro de uma janela."""
    laps = [e for e in entries if isinstance(e.event, LapCompletedEvent)]

    def leader_t(n: int) -> int:
        return min(e.event.session_time_ms for e in laps if e.event.lap == n)

    before, at = leader_t(lap - 1), leader_t(lap)
    head = [
        e for e in entries if isinstance(e.event, RaceStatusEvent) and e.event.state == "started"
    ]
    head += [e for e in laps if e.event.session_time_ms <= before]
    group = [e for e in laps if before < e.event.session_time_ms <= at + 20_000]
    return head, group


@pytest.mark.parametrize("seed", range(6))
def test_shuffled_events_inside_the_window_give_the_same_result(entries, seed):
    """A mesma corrida com a chegada embaralhada, desde que tudo caiba na janela, dá o mesmo
    resultado: o estado é função do conjunto de voltas, não da ordem em que elas chegaram."""
    head, group = window_group(entries, 30)

    def run(order) -> dict:
        t = RaceTracker(2024, 1, wait_s=WAIT)
        now = 0.0
        for e in head:  # o que veio antes, em ordem, e as voltas anteriores já disparadas
            t.apply(e, now := now + 0.001)
            t.due(now)
        t.due(now + 10.0)
        base = now + 10.0
        for k, e in enumerate(order):
            t.apply(e, base + 0.001 * k)  # 30 entradas em ~0,03 s: bem dentro de 0,3 s
        (trigger,) = [x for x in t.due(base + 1.0) if x.lap == 30]
        return stable(payload_of(result_for(trigger)))

    shuffled = list(group)
    random.Random(seed).shuffle(shuffled)

    assert run(shuffled) == run(group)


def test_the_same_entries_delivered_twice_give_the_same_result_and_the_repeats_are_counted(entries):
    clean_tracker, clean = feed(entries, lap_limit=30)
    doubled = [e for entry_ in entries for e in (entry_, entry_)]  # cada entrada duas vezes
    twice_tracker, twice = feed(doubled, lap_limit=30)

    a = payload_of(result_for(next(x for x in clean if x.lap == 30)))
    b = payload_of(result_for(next(x for x in twice if x.lap == 30)))

    assert stable(a)["forecast"] == stable(b)["forecast"] and a["alerts"] == b["alerts"]
    assert clean_tracker.duplicates == 0 and twice_tracker.duplicates > 0
    assert b["inputs"]["duplicates"] == next(x for x in twice if x.lap == 30).duplicates


def test_a_calculation_is_reproducible_run_after_run(entries):
    _, triggers = feed(entries, lap_limit=30)
    trigger = next(x for x in triggers if x.lap == 30)

    assert stable(payload_of(result_for(trigger))) == stable(payload_of(result_for(trigger)))


def test_too_few_cars_means_no_forecast(race):
    two = tuple(r for r in race if r.driver in ("R", "A"))
    entries_ = stream_entries(list(two), total_laps=TOTAL)

    _, triggers = feed(entries_, lap_limit=30)

    assert compute(next(x for x in triggers if x.lap == 30), CONSTANTS, sim_config=CONFIG) is None


def test_the_job_carries_epoch_and_lap_in_its_tag_and_honours_the_stop_signal(entries):
    from strategy.montecarlo import SimulationCancelled

    _, triggers = feed(entries, lap_limit=30)
    trigger = next(x for x in triggers if x.lap == 30)
    job = computation_job(replace(trigger, epoch=7), CONSTANTS, sim_config=CONFIG)

    assert job.tag == (2024, 1, 7, 30)
    with pytest.raises(SimulationCancelled):
        job.run(lambda: True)


# ================================================================ o payload


def test_reliability_flips_to_experimental_at_eighty_percent():
    assert reliability(47, 60) == INDICATIVE  # 78%
    assert reliability(48, 60) == EXPERIMENTAL  # exatamente 80%: a faixa em que ele foi pior
    assert reliability(59, 60) == EXPERIMENTAL


def test_the_payload_puts_the_baseline_beside_the_monte_carlo_and_says_how_many_laps_are_left(
    entries,
):
    _, triggers = feed(entries, lap_limit=30)
    c = result_for(next(x for x in triggers if x.lap == 30))

    p = payload_of(c)

    assert (p["v"], p["type"], p["lap"], p["total_laps"], p["laps_left"]) == (
        SCHEMA_VERSION,
        "strategy",
        30,
        TOTAL,
        30,
    )
    assert p["baseline"] == {"name": BASELINE_NAME, "leader": c.leader}
    drivers = p["forecast"]["drivers"]
    assert [d["baseline_win"] for d in drivers].count(1.0) == 1  # só o líder atual
    assert next(d for d in drivers if d["driver"] == c.leader)["baseline_win"] == 1.0
    assert sum(d["win"] for d in drivers) == pytest.approx(1.0, abs=1e-9)
    assert p["reliability"] == INDICATIVE
    assert drivers == sorted(drivers, key=lambda d: (-d["win"], d["driver"]))


def test_the_payload_reports_how_many_cars_had_reported_and_the_window(entries):
    _, triggers = feed(entries, lap_limit=30)
    trigger = next(x for x in triggers if x.lap == 30)
    p = payload_of(result_for(trigger))

    assert p["inputs"]["cars_reported"] == 14  # os 14 carros do cenário, no instante do líder
    assert p["inputs"]["cars_known"] == trigger.cars_known == 14
    assert p["inputs"]["wait_s"] == WAIT and p["inputs"]["late_laps"] == 0
    assert p["inputs"]["constants"] == "teste"


def test_the_schema_version_is_pinned_to_the_contract_documented():
    """Mudar o significado ou remover um campo é uma decisão (e sobe a versão), não um acidente."""
    assert SCHEMA_VERSION == 1


def test_cars_reported_counts_the_cars_without_a_model_too(entries):
    _, triggers = feed(entries, lap_limit=30)
    c = result_for(next(x for x in triggers if x.lap == 30))
    fewer = replace(c.sim_input, drivers=c.sim_input.drivers[:-1], without_model=("XXX",))

    assert payload_of(c, sim_input=fewer)["inputs"]["cars_reported"] == 14  # 13 com modelo + 1 sem


def test_a_nan_never_leaves_in_the_json():
    with pytest.raises(ValueError):
        dumps({"x": float("nan")})


def test_the_payload_is_json_without_nan_and_round_trips(entries):
    _, triggers = feed(entries, lap_limit=30)
    p = payload_of(result_for(next(x for x in triggers if x.lap == 30)), computed_at_ms=123)

    assert json.loads(dumps(p)) == p


def test_alerts_listed_are_only_the_ones_the_tracker_turned_on(entries):
    _, triggers = feed(entries, lap_limit=30)
    c = result_for(next(x for x in triggers if x.lap == 30))
    pair = next(
        (a.attacker, a.rival)
        for a in c.assessments
        if a.rival is not None and a.probability is not None
    )

    quiet, loud = payload_of(c), payload_of(c, alerts={pair})

    assert quiet["alerts"] == []
    assert [(a["attacker"], a["rival"]) for a in loud["alerts"]] == [pair]
    assert set(loud["alerts"][0]) >= {"probability", "margin_s", "horizon_k", "pit_loss_s"}


@pytest.mark.parametrize(
    ("status", "codes"),
    [
        ("1", []),
        ("4", ["safety_car"]),
        ("6", ["virtual_safety_car"]),
        ("5", ["red_flag"]),
        ("26", ["virtual_safety_car"]),
    ],
)
def test_the_warnings_name_what_the_simulator_does_and_does_not_model(status, codes):
    world = replace(WORLD, status_laps={29: status, 30: status})
    rows = chase_field(world=world)
    entries_ = stream_entries(rows, total_laps=TOTAL)
    _, triggers = feed(entries_, lap_limit=30)
    c = result_for(next(x for x in triggers if x.lap == 30))

    p = payload_of(c)

    assert [w["code"] for w in p["warnings"]] == codes
    for w in p["warnings"]:
        assert w["text"]  # com o texto que o cliente mostra


def test_track_warnings_use_half_the_field_for_the_previous_lap_and_the_leader_for_the_current():
    from strategy.laps import LapRow

    def row(driver, lap, status):
        return LapRow(driver, lap, 90_000, "MEDIUM", 1, 1, None, lap * 1000, False, False, status)

    leader_only = [row("VER", 30, "6")] + [row(d, 29, "1") for d in "ABCD"]
    half_field = (
        [row("VER", 30, "1")]
        + [row(d, 29, "6") for d in "ABCD"[:2]]
        + [row(d, 29, "1") for d in "CD"]
    )
    few = [row("VER", 30, "1")] + [row("A", 29, "6")] + [row(d, 29, "1") for d in "BCD"]

    assert [w["code"] for w in track_warnings(leader_only, 30, 0)] == ["virtual_safety_car"]
    assert [w["code"] for w in track_warnings(half_field, 30, 0)] == ["virtual_safety_car"]
    assert track_warnings(few, 30, 0) == []  # um carro em quatro não é a pista


# ================================================================ o contrato com a API


def test_the_payload_has_exactly_the_fields_the_api_documents_and_validates(race, entries):
    """O que o consumidor grava e o que o `GET` devolve são o mesmo documento. Um campo novo no
    payload que a API não conhece seria descartado EM SILÊNCIO na resposta: aqui falha."""
    from app.schemas import (
        StrategyBaselineOut,
        StrategyDriverOut,
        StrategyForecastOut,
        StrategyInputsOut,
        StrategyOut,
        StrategyWarningOut,
        UndercutAlertOut,
    )

    world = replace(WORLD, status_laps={29: "4", 30: "4"})  # com um aviso de pista para validar
    rows = chase_field(world=world)
    _, triggers = feed(stream_entries(rows, total_laps=TOTAL), lap_limit=30)
    c = result_for(next(x for x in triggers if x.lap == 30))
    pair = next(
        (a.attacker, a.rival) for a in c.assessments if a.rival and a.probability is not None
    )
    p = payload_of(c, alerts={pair})

    StrategyOut.model_validate(p)
    assert set(p) == set(StrategyOut.model_fields)
    assert set(p["baseline"]) == set(StrategyBaselineOut.model_fields)
    assert set(p["forecast"]) == set(StrategyForecastOut.model_fields)
    assert set(p["forecast"]["drivers"][0]) == set(StrategyDriverOut.model_fields)
    assert set(p["alerts"][0]) == set(UndercutAlertOut.model_fields)
    assert set(p["warnings"][0]) == set(StrategyWarningOut.model_fields)
    assert set(p["inputs"]) == set(StrategyInputsOut.model_fields)


def test_the_consumer_assessments_are_the_ones_of_the_shared_leader_lap_function(entries):
    """`strategy.undercut_live` avalia a forma ao vivo com `leader_lap_assessments`; o consumidor
    entrega exatamente o que ela devolve (se um dia divergirem, a avaliação deixa de valer)."""
    from strategy.live import leader_lap_assessments

    _, triggers = feed(entries, lap_limit=30)
    c = result_for(next(x for x in triggers if x.lap == 30))

    _, shared = leader_lap_assessments(
        c.trigger.rows,
        lap=30,
        total_laps=TOTAL,
        t_ms=c.sim_input.t_ms,
        priors=CONSTANTS.priors,
    )

    assert c.assessments == shared
