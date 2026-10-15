"""Testes do histórico de SC, abandonos e trocas de composto (strategy/history.py), com corridas
sintéticas de SC, ritmo e custo de parada plantados."""

from dataclasses import replace

import pytest
from sqlalchemy.orm import sessionmaker

from strategy import history
from strategy.history import (
    DEFAULT_COST_RATIO,
    DEFAULT_GAP_S,
    DEFAULT_LAP_RATIO,
    MIN_FIELD,
    RaceHistory,
    all_races_history,
    build_report,
    compound_transitions,
    compute_race_history,
    lap_flags,
    leave_one_out_history,
    measure_field_stop_losses,
    measure_gap,
    measure_lap_ratio,
    retirements,
    runs,
)
from strategy.laps import EntryRow
from strategy.population import RaceKind, RaceRef
from tests.strategy.synthetic import Plan, World, build_race

TOTAL = 60
# Um mundo plano: ritmo constante de 90 s, sem diferença de composto, sem combustível. Tudo o que
# muda um tempo de volta é plantado (SC, parada), então a medição tem resposta exata.
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=0.0,
    noise=0.0,
)


def field(world: World, *, cars: int = 12, spacing: float = 0.0, stops=None, **plan):
    """`cars` carros idênticos (90 s), `spacing` s de distância entre eles na largada."""
    stops = stops or {}
    plans = [
        Plan(f"D{i:02d}", base=90.0, grid_gap=spacing * i, stops=stops.get(i, ()), **plan)
        for i in range(cars)
    ]
    return build_race(plans, world)


def sc_world(laps: range, **kwargs) -> World:
    return replace(FLAT, status_laps={n: "4" for n in laps}, **kwargs)


# ------------------------------------------------------------------ períodos


def test_a_lap_is_under_sc_when_half_the_field_records_the_code():
    rows = field(sc_world(range(10, 13)), cars=10)
    flipped = [
        replace(r, track_status="1") if r.lap_number == 11 and r.driver >= "D06" else r
        for r in rows
    ]  # 4 dos 10 carros ainda mostram verde na volta 11: 60% sob SC
    barely = [
        replace(r, track_status="1") if r.lap_number == 11 and r.driver >= "D05" else r
        for r in rows
    ]  # 5 dos 10: exatamente a metade, que ainda vale
    minority = [
        replace(r, track_status="1") if r.lap_number == 11 and r.driver >= "D04" else r
        for r in rows
    ]  # 6 dos 10 verdes: só 40% sob SC

    assert lap_flags(flipped, "4")[11] and lap_flags(barely, "4")[11]
    assert not lap_flags(minority, "4")[11]


def test_runs_are_the_contiguous_stretches_with_the_flag_on():
    flags = {n: n in {3, 4, 5, 9, 12, 13} for n in range(1, 14)}

    assert runs(flags) == ((3, 5), (9, 9), (12, 13))
    assert runs({n: False for n in range(1, 5)}) == ()
    assert runs({n: True for n in range(1, 5)}) == ((1, 4),)  # termina na última volta


def test_a_composite_status_code_counts_as_the_code_it_contains():
    """ "124" é amarela, SC e fim do VSC ao mesmo tempo na volta: tem o "4"."""
    rows = field(replace(FLAT, status_laps={20: "124", 21: "124"}), cars=6)

    assert runs(lap_flags(rows, "4")) == ((20, 21),)
    assert runs(lap_flags(rows, "6")) == ()


# ------------------------------------------------------------------ o que o SC faz


def test_the_lap_time_under_sc_is_measured_against_the_drivers_green_pace():
    rows = field(sc_world(range(20, 33), sc_factor=1.5))

    ratio = measure_lap_ratio(rows, lap_flags(rows, "4"))

    assert ratio == pytest.approx(1.5, abs=1e-6)


def test_a_race_with_too_few_sc_laps_has_no_ratio_and_no_gap():
    rows = field(sc_world(range(20, 23), sc_factor=1.5))  # 1 volta de miolo x 12 carros: 12 medidas

    assert measure_lap_ratio(rows, lap_flags(rows, "4")) == pytest.approx(1.5, abs=1e-6)
    short = field(sc_world(range(20, 22), sc_factor=1.5), cars=4)

    assert measure_lap_ratio(short, lap_flags(short, "4")) is None
    assert measure_gap(short, lap_flags(short, "4")) is None


def test_the_ratio_ignores_the_first_and_last_lap_of_the_sc_and_the_pit_laps():
    """A volta de entrada e a de saída do SC são mistas (parte verde, parte SC); só o miolo conta.
    Uma
    parada no miolo também sai: a volta de box tem o tempo de box."""
    world = sc_world(range(20, 33), sc_factor=1.5)
    rows = field(world, stops={0: ((25, "HARD"),)})

    assert measure_lap_ratio(rows, lap_flags(rows, "4")) == pytest.approx(1.5, abs=1e-6)


def test_the_spacing_between_cars_under_sc_is_the_distance_of_the_pack():
    rows = field(sc_world(range(20, 33), sc_factor=1.5), spacing=1.75)

    assert measure_gap(rows, lap_flags(rows, "4")) == pytest.approx(1.75, abs=1e-6)


def test_lapped_cars_do_not_count_as_part_of_the_pack_spacing():
    """Um carro com uma volta de atraso (80 s atrás) não é "colado": distâncias maiores que meia
    volta
    ficam de fora."""
    pack = field(sc_world(range(20, 33), sc_factor=1.5), cars=10, spacing=1.75)
    lapped = build_race(
        [Plan(f"D{i:02d}", base=90.0, grid_gap=1.75 * i) for i in range(10)]
        + [Plan("LAP", base=90.0, grid_gap=1.75 * 9 + 100.0)],
        sc_world(range(20, 33), sc_factor=1.5),
    )

    assert measure_gap(lapped, lap_flags(lapped, "4")) == measure_gap(pack, lap_flags(pack, "4"))


def test_the_first_and_last_lap_of_the_sc_are_not_the_middle_of_it():
    """A volta em que o SC entra e a em que sai são mistas (parte verde): só o miolo mede o ritmo
    sob SC. Aqui o SC dura 3 voltas e as duas das pontas andam a 100 s (1,11x), o miolo a 1,5x."""
    rows = field(sc_world(range(20, 23), sc_factor=1.5))
    mixed = [replace(r, lap_time_ms=100_000) if r.lap_number in (20, 22) else r for r in rows]

    assert measure_lap_ratio(mixed, lap_flags(mixed, "4")) == pytest.approx(1.5, abs=1e-6)


def test_cars_a_lap_apart_have_no_pack_spacing_at_all():
    """Carros 100 s um do outro (mais que meia volta sob SC) são retardatários, não pelotão: nenhuma
    distância vale, e a medida some em vez de dar 100 s."""
    rows = field(sc_world(range(20, 33), sc_factor=1.5), spacing=100.0)

    assert measure_gap(rows, lap_flags(rows, "4")) is None


# ------------------------------------------------------------------ o custo da parada, no campo


def test_a_green_stop_costs_the_planted_loss_against_the_field():
    """12 + 9 s de sobretaxa contra 11 carros que não pararam, no mesmo ritmo: 21 s."""
    rows = field(FLAT, stops={0: ((25, "HARD"),)})

    green, under_sc = measure_field_stop_losses(rows, lap_flags(rows, "4"))

    assert green == (pytest.approx(21.0),) and under_sc == ()


def test_the_loss_does_not_depend_on_the_pace_of_the_pitting_driver_but_the_old_measure_does():
    """O motivo de medir contra o campo: sob SC todos andam 1,5x mais devagar, e a perda contra o
    próprio ritmo normal seria um número sem sentido; contra o campo é a sobretaxa de box (10,5)."""
    rows = field(
        sc_world(range(20, 33), sc_factor=1.5, sc_pit_factor=0.5), stops={0: ((25, "HARD"),)}
    )

    green, under_sc = measure_field_stop_losses(rows, lap_flags(rows, "4"))

    assert green == () and under_sc == (pytest.approx(10.5),)


def test_a_stop_where_the_track_changes_state_in_the_middle_is_neither_green_nor_sc():
    """O SC começa na volta da saída: a parada não é de bandeira verde nem sob SC, e fica de
    fora."""
    world = replace(FLAT, status_laps={26: "4", 27: "4", 28: "4"})  # a entrada é a 25
    rows = field(world, stops={0: ((25, "HARD"),)})

    green, under_sc = measure_field_stop_losses(rows, lap_flags(rows, "4"))

    assert green == () and under_sc == ()


def test_a_stop_needs_enough_cars_that_did_not_stop_to_be_measured():
    few = field(FLAT, cars=MIN_FIELD, stops={0: ((25, "HARD"),)})  # só 7 carros de referência
    enough = field(FLAT, cars=MIN_FIELD + 1, stops={0: ((25, "HARD"),)})

    assert measure_field_stop_losses(few, {})[0] == ()
    assert measure_field_stop_losses(enough, {})[0] == (pytest.approx(21.0),)


def test_cars_that_stopped_nearby_are_not_the_reference():
    """Se os outros também pararam na volta 25, não há campo parado de referência e a conta some em
    vez de dar zero."""
    stops = {i: ((25, "HARD"),) for i in range(12)}
    rows = field(FLAT, stops=stops)

    assert measure_field_stop_losses(rows, {}) == ((), ())


def test_early_stops_are_left_out():
    rows = field(FLAT, stops={0: ((2, "HARD"),)})  # a volta 1 não tem referência

    assert measure_field_stop_losses(rows, {}) == ((), ())


# ------------------------------------------------------------------ abandonos e trocas


def entries(*pairs: tuple[str, str | None]) -> list[EntryRow]:
    return [EntryRow(driver, None, status) for driver, status in pairs]


def test_only_the_retired_status_is_a_retirement_and_the_last_lap_is_recorded():
    rows = [r for r in field(FLAT, cars=6) if not (r.driver == "D01" and r.lap_number > 20)]
    ent = entries(
        ("D00", "Finished"),
        ("D01", "Retired"),
        ("D02", "Lapped"),
        ("D03", "Disqualified"),  # correu: não é abandono
        ("D04", "Did not start"),  # não largou: sai da conta
        ("D05", None),  # status desconhecido: não é "terminou"
    )

    starters, laps = retirements(ent, rows)

    assert starters == 4  # D00, D01, D02, D03
    assert laps == (20,)


def test_a_driver_who_retired_without_a_lap_has_lap_zero():
    rows = [r for r in field(FLAT, cars=3) if r.driver != "D02"]

    assert retirements(entries(("D02", "Retired")), rows) == (1, (0,))


def test_the_compound_transitions_come_from_consecutive_dry_stints():
    rows = field(
        FLAT,
        cars=4,
        stops={0: ((20, "HARD"),), 1: ((15, "SOFT"), (40, "HARD"))},
        compound="MEDIUM",
    )

    assert compound_transitions(rows) == (("MEDIUM", "HARD"), ("MEDIUM", "SOFT"), ("SOFT", "HARD"))


def test_a_wet_compound_is_not_a_dry_transition():
    wet = replace(
        FLAT,
        slopes={**FLAT.slopes, "INTERMEDIATE": 0.0},
        offsets={**FLAT.offsets, "INTERMEDIATE": 0.0},
    )
    rows = field(wet, cars=2, stops={0: ((20, "INTERMEDIATE"),)}, compound="MEDIUM")

    assert compound_transitions(rows) == ()


# ------------------------------------------------------------------ uma corrida inteira


def ref(label="2024 R09 Teste", round_=9) -> RaceRef:
    return RaceRef(1, label, TOTAL, RaceKind.SECA, 1.0, year=2024, round=round_, circuit="Teste")


def test_a_race_history_collects_everything_the_race_says():
    world = replace(
        FLAT, status_laps={**{n: "4" for n in range(20, 33)}, 45: "6", 46: "6"}, sc_factor=1.5
    )
    rows = field(world, spacing=1.75, stops={0: ((10, "HARD"),), 1: ((25, "HARD"),)})
    rows = [r for r in rows if not (r.driver == "D05" and r.lap_number > 30)]
    ent = entries(*[(f"D{i:02d}", "Retired" if i == 5 else "Finished") for i in range(12)])

    h = compute_race_history(ref(), rows, ent)

    assert h.sc_periods == ((20, 32),) and h.vsc_periods == ((45, 46),) and h.red_periods == ()
    assert h.starters == 12 and h.retirement_laps == (30,)
    assert h.transitions == (("MEDIUM", "HARD"), ("MEDIUM", "HARD"))
    assert h.sc_lap_ratio == pytest.approx(1.5, abs=1e-6)
    assert h.sc_gap == pytest.approx(1.75, abs=1e-6)
    assert len(h.green_stop_losses) == 1 and len(h.sc_stop_losses) == 1
    assert (h.label, h.year, h.round, h.total_laps) == ("2024 R09 Teste", 2024, 9, TOTAL)


# ------------------------------------------------------------------ sem a corrida avaliada


def make(
    label: str,
    round_: int,
    *,
    sc=(),
    retired=(),
    starters=20,
    green=(),
    under_sc=(),
    ratio=None,
    gap=None,
    transitions=(),
    year=2024,
) -> RaceHistory:
    return RaceHistory(
        label=label, year=year, round=round_, total_laps=60, sc_periods=tuple(sc),
        vsc_periods=(), red_periods=(), starters=starters, retirement_laps=tuple(retired),
        transitions=tuple(transitions), sc_lap_ratio=ratio, sc_gap=gap,
        green_stop_losses=tuple(green), sc_stop_losses=tuple(under_sc),
    )  # fmt: skip


@pytest.fixture
def races() -> list[RaceHistory]:
    return [
        make("A", 1, sc=[(6, 8)], retired=[12, 30], ratio=1.4, gap=1.5, green=[22.0] * 10),
        make(
            "B", 2, sc=[(30, 35), (50, 52)], retired=[40], ratio=1.6, gap=2.0, under_sc=[11.0] * 10
        ),
        make("C", 3, retired=[], transitions=[("MEDIUM", "HARD")], green=[22.0] * 10),
        make(
            "D",
            4,
            sc=[(12, 17)],
            retired=[58, 1],
            transitions=[("MEDIUM", "HARD"), ("SOFT", "MEDIUM")],
        ),
    ]


def test_the_sc_constants_of_a_race_never_include_the_race(races):
    """Trocar os dados da corrida avaliada por absurdos NÃO muda as constantes dela."""
    absurd = [make("B", 2, sc=[(1, 60)] * 9, retired=[1] * 20, ratio=9.0, gap=99.0)]
    others = [r for r in races if r.label != "B"]

    honest = leave_one_out_history(races, "B")
    tampered = leave_one_out_history([*others, *absurd], "B")

    assert honest == tampered
    assert honest.sc.events == 2 and honest.sc.races == 3  # A (1), D (1), C (0)


def test_the_constants_count_events_fractions_and_durations_from_the_other_races(races):
    priors = leave_one_out_history(races, "C")

    assert (priors.sc.events, priors.sc.races) == (4, 3)
    assert priors.sc.rate == pytest.approx(4 / 3)
    assert sorted(priors.sc.start_fractions) == pytest.approx(
        sorted([6 / 60, 30 / 60, 50 / 60, 12 / 60])
    )
    assert sorted(priors.sc.durations) == [3, 3, 6, 6]
    assert sorted(priors.sc.lap_ratios) == [1.4, 1.6] and sorted(priors.sc.gaps) == [1.5, 2.0]


def test_the_retirement_prior_counts_retirements_and_starters(races):
    priors = leave_one_out_history(races, "C")

    assert (priors.retirement.retirements, priors.retirement.starters) == (5, 60)
    assert priors.retirement.rate == pytest.approx(5 / 60)
    assert sorted(priors.retirement.fractions) == pytest.approx(
        sorted([12 / 60, 30 / 60, 40 / 60, 58 / 60, 1 / 60])
    )


def test_the_compound_counts_come_from_the_other_races(races):
    priors = leave_one_out_history(races, "A")

    assert priors.compounds.counts == {("MEDIUM", "HARD"): 2, ("SOFT", "MEDIUM"): 1}
    assert leave_one_out_history(races, "C").compounds.counts == {
        ("MEDIUM", "HARD"): 1,
        ("SOFT", "MEDIUM"): 1,
    }


def test_the_sc_stop_cost_is_a_ratio_of_the_pooled_losses_or_the_default(races):
    """Com paradas verdes e sob SC bastantes nas outras corridas: mediana sob SC / mediana verde (11
    / 22 = 0,5). Sem elas, o padrão."""
    enough = leave_one_out_history(races, "D")  # A tem 10 verdes, B tem 10 sob SC
    too_few = leave_one_out_history(races, "A")  # sem as verdes de A: só as de C

    assert enough.sc.cost_ratio[0] == pytest.approx(0.5) and enough.sc.n_sc_stops == 10
    assert too_few.sc.cost_ratio[0] == pytest.approx(0.5)  # C tem 10 verdes: ainda basta
    assert leave_one_out_history(races[:1], "A").sc.cost_ratio == DEFAULT_COST_RATIO
    assert leave_one_out_history(races, "D").sc.n_green_stops == 20


def test_the_defaults_fill_in_what_the_other_races_could_not_measure():
    lone = [make("A", 1), make("B", 2)]

    priors = leave_one_out_history(lone, "A")

    assert priors.sc.lap_ratios == (DEFAULT_LAP_RATIO,) and priors.sc.gaps == (DEFAULT_GAP_S,)
    assert priors.sc.events == 0 and priors.sc.rate == 0.0
    assert priors.retirement.rate == 0.0


def test_earlier_only_uses_just_the_races_before_it_in_the_calendar(races):
    priors = leave_one_out_history(races, "C", earlier_only=True)

    assert priors.sc.races == 2 and priors.sc.events == 3  # A e B
    assert "só anteriores" in priors.source and "sem C" in priors.source
    with pytest.raises(ValueError, match="não está no histórico"):
        leave_one_out_history(races, "Z", earlier_only=True)


def test_the_ordering_is_by_calendar_across_seasons(races):
    mixed = [*races, make("OLD", 24, year=2023, sc=[(10, 12)])]

    assert leave_one_out_history(mixed, "A", earlier_only=True).sc.events == 1  # só a de 2023


def test_the_live_constants_use_every_race_and_say_so(races):
    priors = all_races_history(races)

    assert priors.sc.races == 4 and "NÃO use em backtest" in priors.source.replace(
        "não use", "NÃO use"
    )


# ------------------------------------------------------------------ relatório


def test_the_report_counts_the_events_and_warns_about_the_late_ones(races):
    text = build_report(races, markdown=False)

    assert "4 períodos em 3 de 4" in text.replace("**", "") or "períodos em 3 de 4" in text
    assert "Só 3 dos 4 começam depois da volta 10" in text.replace("**", "")
    assert "A" in text and "B" in text


def test_the_markdown_report_explains_the_stop_cost_method_and_what_is_missing(races):
    md = build_report(races, markdown=True)
    flat = " ".join(md.split())

    assert "contra o CAMPO" in flat and "desgaste do pneu sob SC" in flat
    assert "Poucos dados" in flat and "| A |" in md


def test_main_without_races_explains_how_to_ingest(monkeypatch, capsys):
    monkeypatch.setattr(history, "load_race_histories", lambda session, year=None: [])
    monkeypatch.setattr(history, "get_session_factory", lambda: sessionmaker())

    assert history.main([]) == 1
    assert "python -m ingestion" in capsys.readouterr().out


def test_main_prints_and_writes_the_markdown_file(monkeypatch, capsys, tmp_path, races):
    monkeypatch.setattr(history, "load_race_histories", lambda session, year=None: races)
    monkeypatch.setattr(history, "get_session_factory", lambda: sessionmaker())
    target = tmp_path / "docs" / "sc.md"

    assert history.main(["--markdown", str(target)]) == 0

    assert "Safety Car" in capsys.readouterr().out
    assert "| A |" in target.read_text(encoding="utf-8")
