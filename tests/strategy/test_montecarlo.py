"""Testes do Monte Carlo da corrida (strategy/montecarlo.py).

Corridas sintéticas de resposta conhecida e, onde há conta de papel, a conta (a probabilidade de SC
pela fórmula fechada da posteriori Gamma-Poisson, a de abandono pela fórmula condicional). Nenhum
teste depende do valor de uma semente: as tolerâncias são múltiplos do erro de Monte Carlo.
"""

import math
from dataclasses import fields, replace

import numpy as np
import pytest

from strategy.fuel import FuelVariant, physical_coefficient
from strategy.history import (
    DEFAULT_COST_RATIO,
    CompoundPrior,
    RetirementPrior,
    SafetyCarPrior,
    SimPriors,
)
from strategy.laps import LapRow
from strategy.montecarlo import (
    COMPOUNDS,
    SimConfig,
    SimulationCancelled,
    _next_compound,
    _sc_elapsed,
    beyond_mass,
    compress_under_sc,
    draw_all,
    hold_in_traffic,
    make_streams,
    mc_error,
    mixture_weight,
    poisson_counts,
    prepare,
    remaining_sc_laps,
    retirement_laps,
    sample_starts,
    sc_schedule,
    sensitivity,
    simulate,
)
from strategy.undercut import UndercutConfig
from tests.strategy.synthetic import Plan, World, build_race, chase_field

TOTAL = 60
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)
NO_WEAR = replace(
    FLAT, slopes={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
)  # com o k físico, como FLAT
COUNTS = {("MEDIUM", "HARD"): 10, ("SOFT", "MEDIUM"): 4, ("SOFT", "HARD"): 6}
FREE = UndercutConfig(pass_advantage_s=-1e9, follow_gap_s=0.0)  # ultrapassagem livre


def sim_priors(
    events: int = 0,
    races: int = 20,
    starts: tuple[float, ...] = (),
    durations: tuple[int, ...] = (4,),
    retire: tuple[int, int, tuple[float, ...]] = (0, 400, ()),
) -> SimPriors:
    sc = SafetyCarPrior(
        events, races, starts, durations, (1.46,), (1.75,), DEFAULT_COST_RATIO, 15, 100
    )
    return SimPriors(sc, RetirementPrior(*retire), CompoundPrior(dict(COUNTS)), "teste")


def prep(rows, lap, priors=None, *, total=TOTAL, **kwargs):
    return prepare(
        rows,
        year=2024,
        round=1,
        upto_lap=lap,
        total_laps=total,
        sim_priors=priors or sim_priors(),
        **kwargs,
    )


def duel(speed: float, gap: float, *, n: int = 30, total: int = TOTAL, world=NO_WEAR, **world_kw):
    """O líder R e o perseguidor A, `speed` s/volta mais rápido e a `gap` s atrás NA VOLTA n (o
    gerador não tem tráfego: o atraso na largada compensa o ritmo de n voltas)."""
    w = replace(world, total_laps=total, **world_kw)
    plans = [Plan("R", base=90.0), Plan("A", base=90.0 - speed, grid_gap=gap + speed * n)]
    plans += [Plan(f"D{i}", base=100 + i, grid_gap=60 + 5 * i) for i in range(5)]
    return build_race(plans, w)


def deterministic(inp):
    """O mesmo estado sem a incerteza do modelo (ruído de volta, erros de nível, inclinação e
    diferença de composto, dispersão da perda): só ficam os sorteios que o teste controla."""
    return replace(
        inp,
        drivers=tuple(
            replace(
                d,
                sigma=0.0,
                level_se=0.0,
                slope_sd=0.0,
                new_slope_sd=(0.0, 0.0, 0.0),
                offset_shared_sd=(0.0, 0.0, 0.0),
                offset_tau=(0.0, 0.0, 0.0),
            )
            for d in inp.drivers
        ),
        pit_loss=replace(inp.pit_loss, sd=0.0, se=0.0),
    )


def forced_retirements(inp, config, who, last_lap_plus_one):
    """Sorteios em que SÓ `who` abandona, e na volta `last_lap_plus_one` exatamente (o piloto
    completa
    a anterior). Precisa de um prior com UMA fração histórica nessa volta, peso ~1 e sem
    espalhar."""
    draws = draw_all(inp, config)
    names = [d.driver for d in inp.drivers]
    ret_u = np.ones_like(draws.ret_u)
    for driver in who:
        ret_u[:, names.index(driver)] = 0.0
    return replace(
        draws,
        ret_u=ret_u,
        ret_u_comp=np.zeros_like(draws.ret_u_comp),
        ret_u_pos=np.zeros_like(draws.ret_u_pos),
        ret_u_jit=np.full_like(draws.ret_u_jit, 0.5),
    )


def mc_tolerance(p: float, n: int, sigmas: float = 4.0) -> float:
    return sigmas * math.sqrt(max(p * (1 - p), 1e-4) / n)


def win(forecast, driver: str) -> float:
    return forecast.win_probabilities()[driver]


def corrupt_after(rows: list[LapRow], lap: int) -> list[LapRow]:
    return [
        replace(
            r,
            lap_time_ms=1_000,
            compound="SOFT",
            tyre_life=99,
            stint=9,
            track_status="4",
            session_time_ms=1,
            pit_in=True,
            pit_out=True,
        )
        if r.lap_number > lap
        else r
        for r in rows
    ]


# ------------------------------------------------------------------ as peças do sorteio


def test_the_poisson_count_by_inversion_has_the_right_distribution():
    """A contagem sai da inversão da CDF a partir de UMA uniforme por simulação: P(0) = e^-lambda e
    a
    média é lambda (com o teto alto o bastante para não cortar)."""
    u = (np.arange(100_000) + 0.5) / 100_000
    rate = np.full(u.shape, 0.8)

    counts = poisson_counts(u, rate, cap=12)

    assert (counts == 0).mean() == pytest.approx(math.exp(-0.8), abs=1e-3)
    assert counts.mean() == pytest.approx(0.8, abs=2e-3)
    assert (counts == 1).mean() == pytest.approx(0.8 * math.exp(-0.8), abs=1e-3)


def test_a_higher_rate_never_gives_fewer_events_from_the_same_uniform():
    """É o que alinha a sensibilidade (0,5x, 1x, 2x) nos mesmos números aleatórios."""
    u = np.random.default_rng(3).random(5_000)
    low = poisson_counts(u, np.full(u.shape, 0.3), cap=3)
    high = poisson_counts(u, np.full(u.shape, 0.9), cap=3)

    assert (high >= low).all() and (high > low).any()


def test_the_event_count_is_capped():
    counts = poisson_counts(np.array([0.999999]), np.array([5.0]), cap=3)

    assert counts.tolist() == [3]


def test_the_mass_beyond_lap_n_mixes_the_history_and_the_uniform():
    """Frações 0,1 e 0,5 numa corrida de 60 voltas, na volta 30, com peso 0,5 no histórico: nenhuma
    está DEPOIS da volta 30 (0,5 x 60 = 30, não é maior), então só o uniforme conta: 0,5 x 0,5."""
    assert beyond_mass((0.1, 0.5), 60, 30, 0.5) == pytest.approx(0.25)
    assert beyond_mass((0.1, 0.9), 60, 30, 0.5) == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)
    assert beyond_mass((), 60, 30, 0.5) == pytest.approx(0.5)  # sem histórico: o uniforme
    assert beyond_mass((0.9,), 60, 60, 1.0) == 0.0  # acabou a corrida


def test_the_mixture_weight_grows_with_the_events_seen():
    assert mixture_weight(0, 4.0) == 0.0
    assert mixture_weight(4, 4.0) == pytest.approx(0.5)
    assert mixture_weight(36, 4.0) == pytest.approx(0.9)


def test_the_start_laps_are_always_after_lap_n_and_inside_the_race():
    rng = np.random.default_rng(1)
    u = [rng.random((5_000, 3)) for _ in range(3)]

    starts = sample_starts(
        *u, (0.1, 0.5, 0.7, 0.9), total_laps=60, upto_lap=30, weight=0.8, jitter=3
    )

    assert starts.min() >= 31 and starts.max() <= 60


def test_the_historical_start_dominates_when_the_weight_is_one_and_the_uniform_when_zero():
    rng = np.random.default_rng(2)
    u = [rng.random((20_000, 1)) for _ in range(3)]
    kwargs = {"total_laps": 60, "upto_lap": 30, "jitter": 3}

    hist = sample_starts(*u, (0.7,), weight=1.0, **kwargs)  # volta 42
    uniform = sample_starts(*u, (0.7,), weight=0.0, **kwargs)

    assert hist.min() >= 39 and hist.max() <= 45  # 42 +- 3
    assert uniform.min() == 31 and uniform.max() == 60
    assert uniform.mean() == pytest.approx(45.5, abs=0.5)


def test_without_a_historical_event_after_lap_n_the_start_is_uniform():
    rng = np.random.default_rng(4)
    u = [rng.random((5_000, 1)) for _ in range(3)]

    starts = sample_starts(*u, (0.05, 0.1), total_laps=60, upto_lap=30, weight=0.9, jitter=3)

    assert starts.min() >= 31 and starts.max() <= 60 and starts.std() > 5


def test_the_same_race_and_lap_give_the_same_numbers_and_each_component_its_own():
    first = {name: g.random(4) for name, g in make_streams(2024, 6, 30).items()}
    again = {name: g.random(4) for name, g in make_streams(2024, 6, 30).items()}
    other_lap = make_streams(2024, 6, 31)["sc"].random(4)
    other_race = make_streams(2024, 7, 30)["sc"].random(4)

    assert all(np.array_equal(first[name], again[name]) for name in first)
    assert not np.array_equal(first["sc"], other_lap)
    assert not np.array_equal(first["sc"], other_race)
    assert len({tuple(v) for v in first.values()}) == len(first)  # nenhum componente repete outro


def test_the_next_compound_follows_the_history_and_never_the_unavailable_one():
    counts = np.zeros((3, 3)) + 0.5
    counts[1, 2] += 100  # MEDIUM -> HARD
    counts[1, 0] += 1  # MEDIUM -> SOFT
    current = np.ones((4000, 1), dtype=int)  # MEDIUM
    used = np.zeros((4000, 1, 3), dtype=bool)
    used[..., 1] = True
    available = np.array([[True, True, True]])
    u = (np.arange(4000)[:, None] + 0.5) / 4000

    choice = _next_compound(counts, current, used, available, u)[:, 0]

    share_hard = (choice == 2).mean()
    assert share_hard == pytest.approx(100.5 / (100.5 + 1.5), abs=0.01)
    assert not (choice == 1).any()  # antes de dois compostos, nunca repete o único usado

    only_soft = np.array([[True, True, False]])
    assert (_next_compound(counts, current, used, only_soft, u)[:, 0] == 0).all()


def test_with_two_compounds_used_the_same_compound_is_allowed_again():
    counts = np.zeros((3, 3)) + 0.5
    counts[1, 1] += 100
    current = np.ones((2000, 1), dtype=int)
    used = np.zeros((2000, 1, 3), dtype=bool)
    used[..., 1] = used[..., 2] = True
    available = np.array([[True, True, True]])
    u = (np.arange(2000)[:, None] + 0.5) / 2000

    choice = _next_compound(counts, current, used, available, u)[:, 0]

    assert (choice == 1).mean() > 0.9


def test_a_safety_car_in_progress_is_counted_back_from_lap_n():
    world = replace(FLAT, status_laps={28: "4", 29: "4", 30: "4"})
    rows = chase_field(world=world)

    assert _sc_elapsed(rows, 30) == 3 and _sc_elapsed(rows, 29) == 2
    assert _sc_elapsed(rows, 31) == 0 and _sc_elapsed(rows, 27) == 0


# ------------------------------------------------------------------ preparar o estado na volta N


@pytest.fixture(scope="module")
def field_rows():
    return chase_field(gap=5.0, world=FLAT)


@pytest.fixture(scope="module")
def inp(field_rows):
    return prep(field_rows, 30)


def test_each_driver_is_read_from_the_lap_n_row(inp):
    by = {d.driver: d for d in inp.drivers}

    assert COMPOUNDS[by["R"].compound] == "MEDIUM" and by["R"].age == 30
    assert by["R"].used == (False, True, False) and not by["R"].pit_now
    assert by["R"].source == "modelo"
    assert by["R"].levels[1] == pytest.approx(88.0, abs=0.05)  # o nível do próprio composto
    assert by["R"].levels[2] == pytest.approx(88.0, abs=0.05)  # HARD: sem diferença no mundo plano
    assert by["R"].clock_s < by["A"].clock_s  # R está à frente
    assert inp.without_model == () and inp.laps_left == 30 and inp.sc_elapsed == 0


def test_a_driver_who_entered_the_pits_on_lap_n_changes_tyre_on_the_next_one():
    rows = chase_field(r_stop=30, world=FLAT)

    by = {d.driver: d for d in prep(rows, 30).drivers}

    assert by["R"].pit_now and not by["A"].pit_now


def test_a_driver_who_just_stopped_has_no_level_yet_and_keeps_the_previous_stint():
    rows = chase_field(a_stop=27, world=FLAT)  # saída na 28: no 29, 1 volta limpa de 3

    by = {d.driver: d for d in prep(rows, 29).drivers}

    assert by["A"].source == "troca recente" and COMPOUNDS[by["A"].compound] == "HARD"
    assert by["A"].used == (False, True, True)
    assert by["A"].levels[2] == pytest.approx(88.0, abs=0.1)  # a diferença MEDIUM -> HARD vale 0


def test_a_driver_without_a_stint_level_is_projected_from_his_recent_times():
    rows = [
        replace(r, track_status="2") if r.driver == "D03" and r.lap_number >= 3 else r
        for r in chase_field(world=FLAT)
    ]

    by = {d.driver: d for d in prep(rows, 30).drivers}

    assert by["D03"].source == "plano"


def test_a_driver_with_no_clean_lap_at_all_is_left_out_and_reported():
    rows = [
        replace(r, track_status="2") if r.driver == "D03" else r for r in chase_field(world=FLAT)
    ]

    result = prep(rows, 30)

    assert result.without_model == ("D03",)
    assert "D03" not in {d.driver for d in result.drivers}


def test_the_state_at_lap_n_ignores_everything_after_n(field_rows, inp):
    assert prep(corrupt_after(field_rows, 30), 30) == inp


# ------------------------------------------------------------------ propriedades da simulação


def test_the_win_probabilities_sum_to_one_and_the_podium_to_three(inp):
    forecast = simulate(inp, SimConfig(n_sims=3000))

    assert sum(d.win for d in forecast.drivers) == pytest.approx(1.0)
    assert sum(d.podium for d in forecast.drivers) == pytest.approx(3.0)


def test_the_sums_hold_with_retirements_even_when_nearly_everybody_retires(field_rows):
    """Com abandonos a classificação é por voltas completadas e depois o relógio: sempre há um
    vencedor e três no pódio, mesmo que todos abandonem."""
    heavy = sim_priors(retire=(390, 400, tuple(i / TOTAL for i in range(1, TOTAL))))
    forecast = simulate(prep(field_rows, 30, heavy), SimConfig(n_sims=3000))

    assert sum(d.win for d in forecast.drivers) == pytest.approx(1.0)
    assert sum(d.podium for d in forecast.drivers) == pytest.approx(3.0)
    assert min(d.retire for d in forecast.drivers) > 0.4


def test_a_leader_sixty_seconds_ahead_with_few_laps_to_go_wins(field_rows):
    rows = chase_field(gap=60.0, world=FLAT)  # A, 60 s atrás de R
    forecast = simulate(prep(rows, 55), SimConfig(n_sims=3000))

    assert win(forecast, "R") > 0.99
    assert win(forecast, "A") < 0.01


def test_two_identical_cars_split_the_win_fifty_fifty_within_the_monte_carlo_error():
    """Sem tráfego, dois carros iguais e juntos: 50/50. Com tráfego e relógios iguais, quem fica na
    frente é sorteado (desempate), então continua 50/50."""
    rows = chase_field(gap=0.0, world=FLAT)
    n = 5000
    for config in (SimConfig(n_sims=n, traffic=FREE), SimConfig(n_sims=n)):
        forecast = simulate(prep(rows, 30), config)

        assert win(forecast, "R") == pytest.approx(0.5, abs=mc_tolerance(0.5, n))
        assert win(forecast, "A") == pytest.approx(0.5, abs=mc_tolerance(0.5, n))


def test_the_monte_carlo_error_is_the_binomial_one(inp):
    forecast = simulate(inp, SimConfig(n_sims=2000))

    for d in forecast.drivers:
        assert d.win_error == pytest.approx(math.sqrt(d.win * (1 - d.win) / 2000))
        assert d.podium_error == pytest.approx(math.sqrt(d.podium * (1 - d.podium) / 2000))
    assert mc_error(0.5, 100) == pytest.approx(0.05) and mc_error(0.0, 100) == 0.0


def test_a_finished_race_is_decided_by_the_clocks(field_rows):
    forecast = simulate(prep(field_rows, TOTAL), SimConfig(n_sims=500))

    assert forecast.drivers and win(forecast, "R") == 1.0
    assert forecast.sc_probability == 0.0


# ------------------------------------------------------------------ reprodutibilidade


def test_the_same_race_and_lap_give_the_same_forecast(inp):
    config = SimConfig(n_sims=1500)

    first = simulate(inp, config)
    second = simulate(inp, config)

    assert replace(first, duration_s=0.0) == replace(second, duration_s=0.0)


def test_another_lap_or_race_gives_other_numbers():
    """Os sorteios mudam com a corrida e com a volta. Num cenário simétrico (50/50) a vitória muda
    junto; num quase determinístico a mesma conta daria o mesmo número por acaso."""
    rows = chase_field(gap=0.0, world=FLAT)
    base = prep(rows, 30)
    config = SimConfig(n_sims=3000, traffic=FREE)
    reference = draw_all(base, config)
    forecast = simulate(base, config)

    for other in (replace(base, upto_lap=31), replace(base, round=2)):
        other_draws = draw_all(other, config)
        assert not np.array_equal(other_draws.level_err, reference.level_err)
        assert not np.array_equal(other_draws.sc_u_count, reference.sc_u_count)
        assert simulate(other, config).win_probabilities() != forecast.win_probabilities()


def test_the_forecast_at_lap_n_ignores_everything_after_n(field_rows):
    config = SimConfig(n_sims=1500)
    clean = simulate(prep(field_rows, 30), config)
    dirty = simulate(prep(corrupt_after(field_rows, 30), 30), config)

    assert replace(clean, duration_s=0.0) == replace(dirty, duration_s=0.0)


def test_the_variants_share_exactly_the_same_random_numbers(field_rows):
    """Os três `k` mudam VALORES do modelo e nunca a quantidade sorteada: os sorteios são idênticos
    e a diferença entre variantes é sinal."""
    config = SimConfig(n_sims=800)
    draws = [draw_all(prep(field_rows, 30, variant=v), config) for v in FuelVariant]

    for other in draws[1:]:
        for f in fields(draws[0]):
            if f.name != "streams":
                assert np.array_equal(getattr(draws[0], f.name), getattr(other, f.name)), f.name


def test_the_sc_rate_multiplier_uses_the_same_draws_so_the_difference_is_signal(field_rows):
    inp = prep(field_rows, 30, sim_priors(8, 8, (0.7,) * 4, (4, 5)))
    out = sensitivity(inp, SimConfig(n_sims=2000))

    # os mesmos números aleatórios: a probabilidade de SC só pode crescer com a taxa
    assert out[0.5].sc_probability < out[1.0].sc_probability < out[2.0].sc_probability


def test_common_random_numbers_make_a_small_change_visible_where_independent_seeds_hide_it():
    """Perturbar o ritmo de A em 5 ms muda a vitória dele por bem menos que o ruído de uma
    simulação.
    Com os mesmos números a diferença sai limpa; com sementes diferentes, é só ruído."""
    rows = chase_field(gap=2.0, world=FLAT)
    base = prep(rows, 45)
    nudged = replace(
        base,
        drivers=tuple(
            replace(d, levels=tuple(x + 0.005 for x in d.levels)) if d.driver == "A" else d
            for d in base.drivers
        ),
    )
    config = SimConfig(n_sims=2000)
    crn, independent = [], []
    for round_ in range(1, 9):
        a = replace(base, round=round_)
        b = replace(nudged, round=round_)
        draws = draw_all(a, config)
        crn.append(
            win(simulate(b, config, draws=draws), "A") - win(simulate(a, config, draws=draws), "A")
        )
        independent.append(
            win(simulate(replace(nudged, round=round_ + 100), config), "A")
            - win(simulate(a, config), "A")
        )

    assert np.std(crn) < 0.5 * np.std(independent)


# ------------------------------------------------------------------ Safety Car


def test_the_chance_of_a_safety_car_matches_the_closed_form_of_the_gamma_poisson_model():
    """lambda ~ Gamma(eventos + 0,5, taxa = corridas) e o número de SCs que faltam é Poisson(lambda
    *
    massa): P(algum) = 1 - E[exp(-lambda * m)] = 1 - (1 + m/corridas)^-(eventos + 0,5)."""
    events, races, starts = 8, 10, tuple(x / TOTAL for x in (1, 1, 1, 3, 7, 23, 28, 35))
    rows = chase_field(gap=5.0, world=FLAT)
    priors = sim_priors(events, races, starts, (2, 3, 4, 5))
    n = 5000

    forecast = simulate(prep(rows, 20, priors), SimConfig(n_sims=n))

    mass = beyond_mass(starts, TOTAL, 20, mixture_weight(events, 4.0))
    expected = 1 - (1 + mass / races) ** (-(events + 0.5))
    assert forecast.sc_probability == pytest.approx(expected, abs=mc_tolerance(expected, n))


def test_the_rate_multiplier_scales_the_chance_as_the_closed_form_says():
    events, races, starts = 8, 10, (0.7,) * 4
    rows = chase_field(gap=5.0, world=FLAT)
    inp = prep(rows, 20, sim_priors(events, races, starts, (4,)))
    n = 5000
    mass = beyond_mass(starts, TOTAL, 20, mixture_weight(events, 4.0))

    for multiplier, forecast in sensitivity(inp, SimConfig(n_sims=n)).items():
        expected = 1 - (1 + multiplier * mass / races) ** (-(events + 0.5))
        assert forecast.sc_probability == pytest.approx(expected, abs=mc_tolerance(expected, n))


def test_no_safety_car_when_the_rate_is_switched_off(field_rows):
    inp = prep(field_rows, 30, sim_priors(8, 8, (0.7,) * 4, (4,)))

    assert simulate(inp, SimConfig(n_sims=1000, sc_rate_multiplier=0.0)).sc_probability == 0.0


def test_a_safety_car_helps_the_faster_chaser_and_the_more_likely_the_more_it_helps():
    """O perseguidor 0,8 s/volta mais rápido, a 25 s do líder e sem paradas: em 30 voltas ele ganha
    24
    s, não alcança. O SC fecha o intervalo, e a vitória dele sobe junto com a taxa de SC."""
    rows = duel(0.75, 28.0)
    inp = prep(rows, 30, sim_priors(8, 8, tuple(x / TOTAL for x in (23, 28, 35)), (3, 4, 5)))
    out = sensitivity(inp, SimConfig(n_sims=4000, max_stops=0))

    a = [win(out[m], "A") for m in (0.5, 1.0, 2.0)]
    assert a[0] < a[1] < a[2]
    assert (
        win(simulate(inp, SimConfig(n_sims=1000, max_stops=0, sc_rate_multiplier=0.0)), "A") < 0.05
    )


def test_a_safety_car_already_in_progress_closes_the_gap_from_the_first_lap():
    world = dict(status_laps={28: "4", 29: "4", 30: "4"}, sc_factor=1.46)
    quiet = prep(duel(0.75, 28.0), 30, sim_priors(0, 20, (), (5,)))
    under = prep(duel(0.75, 28.0, **world), 30, sim_priors(0, 20, (), (5,)))
    config = SimConfig(n_sims=2000, max_stops=0, sc_rate_multiplier=0.0)

    assert under.sc_elapsed == 3
    assert win(simulate(quiet, config), "A") < 0.05
    assert win(simulate(under, config), "A") > 0.95


def test_a_stop_under_safety_car_is_cheap_so_more_cars_stop():
    """Todos já usaram dois compostos e têm 15 voltas de pneu, 20 voltas pela frente, e a degradação
    suave (0,03 s/volta) faz o ganho de parar (15 s) ficar abaixo da perda (21 s) em bandeira verde.
    Com o SC quase certo a perda cai para ~12 s e os carros param."""
    mild = replace(FLAT, slopes={"SOFT": 0.03, "MEDIUM": 0.03, "HARD": 0.0})
    plans = [
        Plan(f"C{i}", base=90.0, compound="SOFT", stops=((25, "MEDIUM"),), grid_gap=2.0 * i)
        for i in range(10)
    ]
    rows = build_race(plans, mild)
    green = prep(rows, 40, sim_priors(0, 20, (), (4,)))
    sure = prep(rows, 40, sim_priors(8, 8, (0.7,) * 4, (6, 6, 6)))
    config = SimConfig(n_sims=3000)

    def stops(forecast):
        return sum(d.expected_stops for d in forecast.drivers) / len(forecast.drivers)

    calm = simulate(green, replace(config, sc_rate_multiplier=0.0))
    with_sc = simulate(sure, replace(config, sc_rate_multiplier=3.0))

    assert with_sc.sc_probability > 0.8 and calm.sc_probability == 0.0
    assert stops(with_sc) > 3 * stops(calm)


# ------------------------------------------------------------------ as peças puras do tempo


def test_the_sc_schedule_paints_exactly_the_laps_each_event_covers():
    """upto_lap 30, 10 voltas que faltam (31..40 = índices 0..9). Simulação 0: 1 SC na 33 por 3
    voltas (índices 2..4). Simulação 1: 2 SCs, na 31 por 2 e na 38 por 4 (o segundo passa do fim).
    Simulação 2: nenhum novo, mas um em curso que ainda dura 2 voltas."""
    count = np.array([1, 2, 0])
    starts = np.array([[33, 50], [31, 38], [35, 35]])
    durations = np.array([[3, 3], [2, 4], [5, 5]])

    mask = sc_schedule(
        count, starts, durations, upto_lap=30, laps=10, remaining=np.array([0, 0, 2])
    )

    expected = np.zeros((3, 10), dtype=bool)
    expected[0, 2:5] = True
    expected[1, 0:2] = True
    expected[1, 7:10] = True
    expected[2, 0:2] = True
    assert (mask == expected).all()


def test_an_event_that_is_not_counted_paints_nothing():
    mask = sc_schedule(
        np.array([0]),
        np.array([[31]]),
        np.array([[5]]),
        upto_lap=30,
        laps=10,
        remaining=np.array([0]),
    )

    assert not mask.any()


def test_a_safety_car_in_progress_lasts_the_historical_duration_minus_what_it_already_did():
    durations = np.array([3, 5, 8])
    u = np.array([0.0, 0.99])

    assert remaining_sc_laps(u, durations, 4).tolist() == [1, 4]  # 5 - 4 e 8 - 4
    assert remaining_sc_laps(u, durations, 9).tolist() == [1, 1]  # nenhum tão longo: mais uma volta
    assert remaining_sc_laps(u, durations, 0).tolist() == [0, 0]  # sem SC em curso


def test_a_retiring_driver_completes_the_lap_before_the_one_he_retires_on():
    u = np.array([0.1, 0.9])

    laps = retirement_laps(u, np.array([0.5, 0.5]), np.array([42, 42]), 60)

    assert laps.tolist() == [41, 61]  # abandona na 42: completou a 41; a outra termina (L + 1)


def test_traffic_holds_a_car_that_cannot_pass_and_lets_through_one_that_can():
    inf = np.inf
    tentative = np.array([[100.0, 99.5, 99.0], [100.0, 99.5, inf]])
    equal = np.full_like(tentative, 90.0)
    rule = UndercutConfig(pass_advantage_s=0.7, follow_gap_s=0.7)

    held = hold_in_traffic(tentative, equal, rule)

    # sem vantagem de ritmo, cada um fica a 0,7 s do da frente, e a retenção encadeia
    assert held[0].tolist() == pytest.approx([100.0, 100.7, 101.4])
    assert held[1, :2].tolist() == pytest.approx([100.0, 100.7]) and held[1, 2] == inf

    faster = np.array([[90.0, 89.0, 88.0]])  # 1 s/volta de vantagem sobre o da frente: passa
    free = hold_in_traffic(tentative[:1], faster, rule)
    assert free[0].tolist() == pytest.approx([100.0, 99.5, 99.0])


def test_a_car_behind_one_that_would_finish_after_it_is_not_held():
    """O da frente termina DEPOIS do de trás (parou nos boxes): não há de quem ficar atrás."""
    tentative = np.array([[110.0, 99.0]])

    held = hold_in_traffic(tentative, np.array([[100.0, 90.0]]), UndercutConfig())

    assert held[0].tolist() == pytest.approx([110.0, 99.0])


def test_under_safety_car_the_pack_closes_up_and_the_lapped_cars_keep_their_distance():
    """Líder de 90 s a 1,5x = 135 s. Atrás: a 4 s, a 9 s e (retardatário) a 100 s do primeiro. O
    pelotão fecha a 1,75 s, o retardatário mantém os ~91 s que o separavam do carro da frente."""
    prev = np.array([[1000.0, 1004.0, 1009.0, 1100.0]])

    out = compress_under_sc(
        prev, np.array([90.0]), np.array([1.5]), np.array([1.75]), np.zeros((1, 4))
    )

    assert out[0].tolist() == pytest.approx([1135.0, 1136.75, 1138.5, 1138.5 + 91.0])


def test_the_pit_penalty_under_safety_car_comes_after_the_compression_and_only_on_who_pits():
    prev = np.array([[1000.0, 1004.0, 1009.0]])

    out = compress_under_sc(
        prev, np.array([90.0]), np.array([1.5]), np.array([1.75]), np.array([[0.0, 12.0, 0.0]])
    )

    assert out[0].tolist() == pytest.approx([1135.0, 1136.75 + 12.0, 1138.5])


def test_a_car_that_retired_has_no_clock_under_safety_car():
    prev = np.array([[1000.0, 1004.0, np.inf]])

    out = compress_under_sc(
        prev, np.array([90.0]), np.array([1.5]), np.array([1.75]), np.zeros((1, 3))
    )

    assert out[0, :2].tolist() == pytest.approx([1135.0, 1136.75]) and out[0, 2] == np.inf


# ------------------------------------------------------------------ cenários que isolam uma regra


def test_two_cars_that_retire_on_the_same_lap_are_ordered_by_the_clock_of_the_lap_they_completed():
    """P e Q abandonam na volta 40 (completaram a 39). Q é 3 s/volta mais rápido e começa a 25,5 s
    de P na volta 30: na 39 Q já está 1,5 s à frente. Dois carros lentos terminam a corrida e levam
    1º e 2º; o 3º é o melhor dos que abandonaram, e tem de ser Q."""
    plans = [
        Plan("P", base=90.0),
        Plan("Q", base=87.0, grid_gap=25.5 + 3.0 * 30),
        Plan("D1", base=100.0, grid_gap=200.0),
        Plan("D2", base=101.0, grid_gap=260.0),
    ]
    rows = build_race(plans, replace(NO_WEAR, total_laps=TOTAL))
    inp = deterministic(prep(rows, 30, sim_priors(retire=(1000, 2000, (40 / TOTAL,)))))
    config = SimConfig(
        n_sims=500, traffic=FREE, max_stops=0, start_jitter_laps=0, sc_rate_multiplier=0.0
    )

    forecast = simulate(inp, config, draws=forced_retirements(inp, config, ["P", "Q"], 40))
    podium = {d.driver: d.podium for d in forecast.drivers}

    assert {d.driver: d.retire for d in forecast.drivers}["Q"] == 1.0
    assert podium["Q"] == 1.0 and podium["P"] == 0.0


def test_a_retired_car_does_not_hold_back_the_one_behind_it():
    """O líder R abandona na volta 31 (completou a 30) e A vinha a 5 s dele: A tem de ser livre para
    vencer. Um carro abandonado com relógio finito seguraria A para sempre."""
    plans = [
        Plan("R", base=90.0),
        Plan("A", base=90.0, grid_gap=5.0),
        Plan("D1", base=100.0, grid_gap=200.0),
        Plan("D2", base=101.0, grid_gap=260.0),
    ]
    rows = build_race(plans, replace(NO_WEAR, total_laps=TOTAL))
    inp = deterministic(prep(rows, 30, sim_priors(retire=(1000, 2000, (31 / TOTAL,)))))
    config = SimConfig(n_sims=500, max_stops=0, start_jitter_laps=0, sc_rate_multiplier=0.0)

    forecast = simulate(inp, config, draws=forced_retirements(inp, config, ["R"], 31))

    assert win(forecast, "A") == 1.0 and win(forecast, "R") == 0.0


def test_the_minimum_tyre_age_blocks_a_voluntary_stop():
    """Pneu MEDIUM que degrada 1 s/volta, 10 voltas de idade, 5 pela frente: parar compensa, e para
    com `min_stint_laps` = 5. Exigindo 30 voltas de idade, nenhum para (já usaram dois
    compostos)."""
    steep = replace(NO_WEAR, slopes={"SOFT": 0.0, "MEDIUM": 1.0, "HARD": 0.0})
    plans = [
        Plan(f"C{i}", base=90.0, compound="SOFT", stops=((45, "MEDIUM"),), grid_gap=2.0 * i)
        for i in range(6)
    ]
    inp = prep(build_race(plans, replace(steep, total_laps=TOTAL)), 55)

    free = simulate(inp, SimConfig(n_sims=1000, sc_rate_multiplier=0.0, min_stint_laps=5))
    blocked = simulate(inp, SimConfig(n_sims=1000, sc_rate_multiplier=0.0, min_stint_laps=30))

    assert sum(d.expected_stops for d in free.drivers) / 6 > 0.9
    assert sum(d.expected_stops for d in blocked.drivers) == 0.0


def pit_now_field():
    """R entra nos boxes na volta 30 (a volta de entrada já está no relógio: 12 s de sobretaxa) e
    sai 5 s à frente de C e E, que vinham atrás. A, na frente, e dois lentos completam o grid."""
    plans = [
        Plan("A", base=90.0, grid_gap=10.0),
        Plan("R", base=90.0, grid_gap=5.0, stops=((30, "HARD"),)),
        Plan("C", base=90.0, grid_gap=22.0),
        Plan("E", base=90.0, grid_gap=23.0),
        Plan("D1", base=100.0, grid_gap=200.0),
        Plan("D2", base=101.0, grid_gap=260.0),
    ]
    return build_race(plans, replace(NO_WEAR, total_laps=TOTAL))


def test_the_lap_of_entry_is_already_in_the_clock_so_only_the_rest_of_the_loss_is_added():
    """O excesso da volta de entrada (os 12 s plantados) é medido da volta N, e a volta de saída
    soma
    só `perda - excesso`: senão a perda conta duas vezes."""
    inp = prep(pit_now_field(), 30)
    by = {d.driver: d for d in inp.drivers}

    assert by["R"].pit_now and by["R"].inlap_excess == pytest.approx(12.0, abs=0.01)
    assert by["A"].inlap_excess == 0.0 and not by["A"].pit_now


def test_a_stop_that_started_on_lap_n_still_costs_the_out_lap_and_drops_the_car_behind_the_others():
    """R sai 5 s à frente de C e E e perde ainda ~10,5 s na volta de saída: termina atrás dos dois,
    e
    fica fora do pódio (A, C, E). Sem a perda restante R subiria ao pódio."""
    inp = deterministic(prep(pit_now_field(), 30))

    config = SimConfig(n_sims=500, max_stops=0, sc_rate_multiplier=0.0, retire_rate_multiplier=0.0)

    forecast = simulate(inp, config)
    podium = {d.driver: d.podium for d in forecast.drivers}

    assert podium["A"] == podium["C"] == podium["E"] == 1.0
    assert podium["R"] == 0.0


def test_the_out_lap_costs_only_the_rest_of_the_loss_so_the_car_stays_ahead_of_far_rivals():
    """R sai 7 s atrás de A e perde ~10,5 s na volta de saída: termina 17,5 s atrás de A, ainda à
    frente de E (20 s) e de F (22 s), e fica em 2º. Se a perda inteira (22,5 s) fosse somada de
    novo,
    R cairia para 29,5 s, atrás dos dois, e sairia do pódio."""
    plans = [
        Plan("A", base=90.0, grid_gap=10.0),
        Plan("R", base=90.0, grid_gap=5.0, stops=((30, "HARD"),)),
        Plan("E", base=90.0, grid_gap=30.0),
        Plan("F", base=90.0, grid_gap=32.0),
        Plan("D1", base=100.0, grid_gap=200.0),
        Plan("D2", base=101.0, grid_gap=260.0),
    ]
    inp = deterministic(prep(build_race(plans, replace(NO_WEAR, total_laps=TOTAL)), 30))
    config = SimConfig(n_sims=500, max_stops=0, sc_rate_multiplier=0.0, retire_rate_multiplier=0.0)

    forecast = simulate(inp, config)
    podium = {d.driver: d.podium for d in forecast.drivers}

    assert podium["R"] == 1.0 and podium["E"] == 1.0 and podium["F"] == 0.0


# ------------------------------------------------------------------ paradas dos outros carros


def stop_world():
    mild = replace(FLAT, slopes={"SOFT": 0.03, "MEDIUM": 0.03, "HARD": 0.0})
    plans = [
        Plan("R", base=88.0),  # um composto só: MEDIUM, 45 voltas de idade na volta 45
        Plan("A", base=88.0, stops=((20, "HARD"),), grid_gap=25.0),  # já usou dois
    ]
    plans += [
        Plan(f"D{i}", base=94 + i, stops=((18 + i, "HARD"),), grid_gap=40 + 3 * i) for i in range(6)
    ]
    return build_race(plans, mild)


def test_a_car_that_used_only_one_compound_stops_before_the_end_in_every_simulation():
    """A regra de dois compostos: mesmo quando parar não compensa (15 voltas e degradação suave),
    quem ainda não usou dois compostos de seco para, e quem já usou não para sem motivo."""
    forecast = simulate(prep(stop_world(), 45), SimConfig(n_sims=2000))
    by = {d.driver: d for d in forecast.drivers}

    assert by["R"].expected_stops == pytest.approx(1.0, abs=0.01)
    assert by["A"].expected_stops < 0.1


def test_a_car_on_old_tyres_with_many_laps_left_stops_and_one_on_fresh_tyres_does_not():
    rows = chase_field(gap=5.0, world=FLAT)  # R e A: 30 voltas de pneu, 30 pela frente
    forecast = simulate(prep(rows, 30), SimConfig(n_sims=2000))
    by = {d.driver: d for d in forecast.drivers}

    assert by["R"].expected_stops > 0.95 and by["A"].expected_stops > 0.95
    assert by["D11"].expected_stops < 0.1  # já parou na 23, pneu de HARD que não degrada


def test_no_car_stops_more_than_the_cap_and_a_zero_cap_means_no_stops():
    """Volta 40: ninguém está entrando nos boxes agora (a troca que começou na volta N conta como
    uma
    parada, o que o teto de zero não desfaz)."""
    rows = chase_field(gap=5.0, world=FLAT)
    inp = prep(rows, 40)

    capped = simulate(inp, SimConfig(n_sims=1500, max_stops=1))
    none = simulate(inp, SimConfig(n_sims=1500, max_stops=0))

    assert max(d.expected_stops for d in capped.drivers) <= 1.0
    assert (
        max(d.expected_stops for d in capped.drivers) > 0.9
    )  # R e A, com 40 voltas de pneu, param
    assert max(d.expected_stops for d in none.drivers) == 0.0


def test_a_stop_already_under_way_on_lap_n_counts_and_costs_the_loss_on_the_next_lap():
    """D00 entra nos boxes na volta 12: troca de pneu no começo da 13 e a perda cai nessa volta,
    então
    ele sai atrás de quem estava perto e a parada conta nas paradas esperadas."""
    rows = chase_field(gap=5.0, world=FLAT)

    forecast = simulate(prep(rows, 12), SimConfig(n_sims=800, max_stops=0))
    by = {d.driver: d for d in forecast.drivers}

    assert by["D00"].expected_stops == 1.0
    assert all(d.expected_stops == 0.0 for d in forecast.drivers if d.driver != "D00")


# ------------------------------------------------------------------ abandonos


def test_the_chance_of_retiring_matches_the_conditional_formula():
    """r ~ Beta; dado que o piloto corre até a volta N, retira-se depois com
    r*m / (1 - r*(1-m)), com `m` a massa do histórico depois da volta N."""
    retirements, starters = 200, 400
    fractions = tuple(i / TOTAL for i in range(1, TOTAL))
    rows = chase_field(gap=5.0, world=FLAT)
    n = 4000

    forecast = simulate(
        prep(rows, 35, sim_priors(retire=(retirements, starters, fractions))), SimConfig(n_sims=n)
    )

    mass = beyond_mass(fractions, TOTAL, 35, mixture_weight(retirements, 4.0))
    r = (retirements + 0.5) / (starters + 1)
    expected = r * mass / (1 - r * (1 - mass))
    for d in forecast.drivers:
        assert d.retire == pytest.approx(expected, abs=mc_tolerance(expected, n) + 0.01)


def test_the_leader_wins_less_when_retirements_are_likely_by_about_the_chance_of_retiring():
    rows = chase_field(gap=60.0, world=FLAT)
    safe = simulate(prep(rows, 35), SimConfig(n_sims=3000))
    risky = simulate(
        prep(rows, 35, sim_priors(retire=(200, 400, tuple(i / TOTAL for i in range(1, TOTAL))))),
        SimConfig(n_sims=3000),
    )

    leader = next(d for d in risky.drivers if d.driver == "R")
    assert win(safe, "R") > 0.99
    assert win(risky, "R") == pytest.approx(1 - leader.retire, abs=0.03)


def test_the_retirement_rate_multiplier_switches_it_off(field_rows):
    inp = prep(field_rows, 30, sim_priors(retire=(200, 400, (0.7,) * 50)))

    off = simulate(inp, SimConfig(n_sims=1000, retire_rate_multiplier=0.0))

    assert max(d.retire for d in off.drivers) == 0.0


# ------------------------------------------------------------------ tráfego


@pytest.mark.parametrize(
    ("speed", "limit", "chaser_wins"),
    [(0.3, 0.7, False), (1.5, 0.7, True), (0.8, 0.7, True), (0.8, 1.0, False)],
)
def test_a_faster_car_only_passes_with_the_advantage_the_undercut_rule_asks_for(
    speed, limit, chaser_wins
):
    """A regra é a do undercut: sem vantagem de ritmo de `pass_advantage_s` por volta, o perseguidor
    fica a `follow_gap_s` atrás e não passa, mesmo mais rápido. Sem paradas, 20 voltas."""
    rows = duel(speed, 0.5, n=20, total=40)
    inp = prep(rows, 20, total=40)
    config = SimConfig(n_sims=2000, max_stops=0, traffic=UndercutConfig(pass_advantage_s=limit))

    forecast = simulate(inp, config)

    leader, chaser = win(forecast, "R"), win(forecast, "A")
    assert (chaser > 0.99) if chaser_wins else (leader > 0.99)


def test_free_overtaking_lets_the_faster_car_through_whatever_the_advantage():
    rows = duel(0.3, 0.5, n=20, total=40)

    forecast = simulate(prep(rows, 20, total=40), SimConfig(n_sims=2000, max_stops=0, traffic=FREE))

    assert win(forecast, "A") > 0.99


# ------------------------------------------------------------------ execução


def test_cancelling_stops_the_simulation_with_an_exception(inp):
    calls = []

    def stop_after_three():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(SimulationCancelled):
        simulate(inp, SimConfig(n_sims=500), should_stop=stop_after_three)

    assert len(calls) == 4  # checou uma vez por volta, e parou na quarta


def test_a_simulation_that_is_not_cancelled_checks_once_per_lap(inp):
    calls = []

    simulate(inp, SimConfig(n_sims=200), should_stop=lambda: calls.append(1) or False)

    assert len(calls) == inp.laps_left


def test_there_must_be_a_driver_to_simulate():
    rows = [replace(r, track_status="2") for r in chase_field(world=FLAT)]

    with pytest.raises(ValueError, match="nenhum piloto"):
        simulate(prep(rows, 30), SimConfig(n_sims=100))


def test_the_duration_is_measured_and_a_whole_race_fits_the_budget():
    """Orçamento: 5.000 simulações, 20 carros, ~48 voltas por frente em ~1 s no desenvolvimento; a
    volta dura ~90 s. O teto aqui é folgado para máquinas lentas de CI."""
    rows = chase_field(gap=5.0, world=FLAT, slow_cars=18)
    inp = prep(rows, 12, sim_priors(8, 21, (0.4, 0.5), (4, 5)))

    forecast = simulate(inp, SimConfig(n_sims=5000))

    assert len(inp.drivers) == 20 and forecast.n_sims == 5000
    assert 0 < forecast.duration_s < 15.0


def test_the_forecast_reports_what_it_used(inp):
    forecast = simulate(inp, SimConfig(n_sims=500, sc_rate_multiplier=2.0))

    assert (forecast.year, forecast.round, forecast.lap, forecast.total_laps) == (
        2024,
        1,
        30,
        TOTAL,
    )
    assert forecast.sc_multiplier == 2.0 and forecast.n_sims == 500
    assert forecast.variant == "encolhimento" and forecast.without_model == ()
    assert {d.source for d in forecast.drivers} <= {"modelo", "troca recente", "plano"}
