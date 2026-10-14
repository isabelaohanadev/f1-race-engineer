"""Testes do alerta de undercut (strategy/undercut.py) com corridas sintéticas de GANHO E PERDA
CONHECIDOS.

A estratégia dos testes: um mundo sem ruído em que a margem sai de uma conta de papel
(`hand_margin`),
um mundo contrafactual em que o undercut ACONTECE no gerador (a margem realizada é a diferença dos
relógios), e os casos de tráfego, regras duras, histerese e vazamento do futuro.
"""

from dataclasses import replace

import numpy as np
import pytest

from strategy.degradation import SlopeEstimate
from strategy.fuel import FuelVariant, coefficient_sd, physical_coefficient
from strategy.laps import LapRow
from strategy.pace import Offset, StintLevel
from strategy.pitloss import SD_FLOOR, PitLoss
from strategy.priors import DEFAULT_PRIORS
from strategy.undercut import (
    DEFAULT_SD_INFLATION,
    AlertTracker,
    CarState,
    UndercutAssessment,
    UndercutConfig,
    _hard_rules,
    _margin_sd,
    _Side,
    assess_all,
    assess_undercut,
    build_context,
    car_ahead,
    car_states,
    expected_stint_length,
    flat_pace,
    likely_next_compound,
    normal_cdf,
)
from tests.strategy.synthetic import Plan, World, chase_field, clock_at

TOTAL, N, K = 60, 30, 5

# Um mundo em que a conta cabe no papel: o MEDIUM degrada 0,10 s/volta, o HARD não degrada, os
# compostos não têm diferença de ritmo, e o `k` plantado é exatamente o físico.
FLAT = World(
    total_laps=TOTAL,
    slopes={"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.0},
    offsets={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    kappa=physical_coefficient(TOTAL),
    noise=0.0,
)
EXACT = replace(DEFAULT_PRIORS, offset_sd=0.0)  # sem o espalhamento entre corridas dos priors
SLOPE = 0.10


def hand_margin(n: int, k: int, gap: float) -> float:
    """A margem do undercut no mundo plano, à mão.

    O atacante A (a `gap` s atrás) entra na volta N+1 e o rival R na N+K; os dois andam K+2 voltas
    até o fim da N+K+2 e pagam a mesma parada, que cancela. Combustível também cancela (as mesmas
    voltas). O que sobra é o pneu: A anda 1 volta de MEDIUM (idade N+1) e K+1 de HARD (que não
    degrada); R anda K de MEDIUM (idades N+1..N+K) e 2 de HARD. A diferença de MEDIUM entre os dois
    é
    SLOPE * (K*N + K*(K+1)/2 - (N+1)); a de HARD é zero. A vantagem de A é essa menos o `gap`."""
    return SLOPE * (k * n + k * (k + 1) / 2 - (n + 1)) - gap


def undercut_context(gap: float = 5.0, world: World = FLAT, *, lap: int = N, **kwargs):
    rows = chase_field(gap=gap, world=world)
    return build_context(rows, upto_lap=lap, total_laps=TOTAL, priors=EXACT, **kwargs)


@pytest.fixture(scope="module")
def ctx():
    return undercut_context()


@pytest.fixture(scope="module")
def result(ctx):
    return assess_undercut(ctx, "A", "R", horizon_k=K)


# ------------------------------------------------------------------ a margem, à mão


def test_the_margin_is_the_hand_computed_gain_of_the_undercut(result):
    assert hand_margin(N, K, 5.0) == pytest.approx(8.4)  # a conta de papel, conferida

    assert result.margin == pytest.approx(8.4, abs=1e-3)
    assert result.margin_free == pytest.approx(8.4, abs=1e-3)
    assert result.blocked_by is None
    assert result.reasons == ()
    assert result.horizon_k == K
    assert result.compounds == ("MEDIUM", "HARD", "MEDIUM", "HARD")


@pytest.mark.parametrize("gap", [0.0, 5.0, 10.0, 20.0])
def test_each_second_of_gap_costs_the_attacker_one_second_of_margin(gap):
    result = assess_undercut(undercut_context(gap), "A", "R", horizon_k=K)

    assert result.margin == pytest.approx(hand_margin(N, K, gap), abs=1e-3)
    assert (result.probability > 0.5) == (result.margin > 0)


def test_a_deficit_smaller_than_the_following_distance_is_held_at_that_distance():
    """No fim, A vem 0,6 s atrás de R e sem vantagem de ritmo para passar (os dois no HARD, que não
    degrada): o modelo o segura a 0,7 s (`follow_gap_s`), então o déficit de 0,6 vira 0,7. É a regra
    de ultrapassagem, e a descontinuidade (0,1 s) é o preço de não deixar o carro "passar" um outro
    de mesmo ritmo."""
    gap = 13.4 + 0.6  # a margem livre seria -0,6
    result = assess_undercut(undercut_context(gap), "A", "R", horizon_k=K)

    assert result.margin_free == pytest.approx(-0.6, abs=1e-3)
    assert result.margin == pytest.approx(-0.7, abs=1e-3)
    assert result.blocked_by == "R"


@pytest.mark.parametrize("k", [3, 5, 8])
def test_a_longer_horizon_gives_the_new_tyres_more_laps_to_pay_back(ctx, k):
    result = assess_undercut(ctx, "A", "R", horizon_k=k)

    assert result.margin == pytest.approx(hand_margin(N, k, 5.0), abs=1e-3)


def test_the_pit_loss_itself_cancels_in_the_margin_but_not_in_where_the_car_rejoins(result):
    """Os dois carros pagam a mesma parada, então a média da perda não muda a margem. Muda o ponto
    de
    retorno: uma perda 8 s maior devolve o atacante 8 s mais atrás."""
    slow_pit = undercut_context(world=replace(FLAT, in_extra=20.0))
    slower = assess_undercut(slow_pit, "A", "R", horizon_k=K)

    assert slower.margin == pytest.approx(result.margin, abs=1e-3)
    assert slower.pit_loss - result.pit_loss == pytest.approx(8.0, abs=0.3)
    assert slower.rejoin_gap_s - result.rejoin_gap_s == pytest.approx(8.0, abs=0.3)


def test_the_attacker_rejoins_behind_the_rival_by_the_gap_plus_the_loss_minus_the_new_tyre_gain(
    result,
):
    """O retorno à pista, pelos relógios: atrás de R, a `gap` + `perda` menos o que a volta de saída
    ganha sobre a volta de R no pneu gasto (SLOPE * (N+2))."""
    expected = 5.0 + result.pit_loss - SLOPE * (N + 2)

    assert result.rejoin_behind == "R"
    assert result.rejoin_gap_s == pytest.approx(expected, abs=1e-3)


# ------------------------------------------------------------------ previsto contra realizado


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("k", [4, 6])
def test_the_predicted_margin_matches_the_one_realized_in_the_counterfactual_race(seed, k):
    """Com compostos de verdade (ritmo e degradação diferentes) e ruído, o modelo ajustado SÓ até a
    volta N prevê a margem que o gerador realiza quando o undercut acontece de fato (a parada de A
    na N+1 e a de R na N+K). O erro fica dentro do desvio declarado."""
    world = World(total_laps=TOTAL, noise=0.12, seed=seed)
    observed = chase_field(gap=3.0, world=world)
    result = assess_undercut(
        build_context(observed, upto_lap=N, total_laps=TOTAL), "A", "R", horizon_k=k
    )
    counterfactual = chase_field(a_stop=N + 1, r_stop=N + k, gap=3.0, world=world)
    realized = clock_at(counterfactual, "R", N + k + 2) - clock_at(counterfactual, "A", N + k + 2)

    assert result.blocked_by is None
    assert abs(result.margin - realized) < 1.6
    assert abs(result.margin - realized) < result.margin_sd


def test_the_counterfactual_race_shares_the_whole_history_up_to_lap_n():
    """A premissa do teste acima: o mundo em que o undercut acontece tem a MESMA história até N."""
    world = World(total_laps=TOTAL, noise=0.12, seed=1)
    observed = chase_field(gap=3.0, world=world)
    counterfactual = chase_field(a_stop=N + 1, r_stop=N + 5, gap=3.0, world=world)

    def upto(rows):
        return [r for r in rows if r.lap_number <= N]

    assert upto(observed) == upto(counterfactual)


# ------------------------------------------------------------------ tráfego


def blocker_field(gap: float, a_stop: int = 45, r_stop: int = 45) -> list[LapRow]:
    """O cenário com um carro `C` ligeiramente mais lento que o pneu novo de A, posicionado 0,3 s à
    frente de onde A volta à pista. Duas passadas: a primeira mede onde A voltaria; a segunda recua
    `C` até ele ficar a 0,3 s (o relógio dele muda 1:1 com o `grid_gap`)."""

    def build(x: float, a: int, r: int) -> list[LapRow]:
        c = Plan("C", base=84.95, grid_gap=x)  # ritmo = A no pneu novo + 0,25..0,65 s
        return chase_field(a_stop=a, r_stop=r, gap=gap, world=FLAT, extra=(c,))

    x = 100.0
    for _ in range(2):
        probe = build_context(build(x, 45, 45), upto_lap=N, total_laps=TOTAL, priors=EXACT)
        found = assess_undercut(probe, "A", "R", horizon_k=K)
        assert found.rejoin_behind == "C", "a premissa do cenário: A volta logo atrás de C"
        x += found.rejoin_gap_s - 0.3
    return build(x, a_stop, r_stop)


def test_rejoining_behind_a_slower_car_cancels_part_of_the_gain():
    """C anda um pouco mais devagar que A no pneu novo e está 0,3 s à frente do ponto de retorno: A
    não passa (a vantagem sobre ele é menor que o limite), fica 0,7 s atrás dele até o fim, e a
    margem cai da livre (8,4 s) para o que R deixa de folga sobre C."""
    observed = blocker_field(5.0)
    result = assess_undercut(
        build_context(observed, upto_lap=N, total_laps=TOTAL, priors=EXACT), "A", "R", horizon_k=K
    )
    real = blocker_field(5.0, a_stop=N + 1, r_stop=N + K)
    behind_c = clock_at(real, "R", N + K + 2) - clock_at(real, "C", N + K + 2) - 0.7

    assert result.blocked_by == "C"
    assert result.margin_free == pytest.approx(8.4, abs=1e-3)
    assert behind_c > 0, "a premissa: R termina atrás de C, então A ainda fica à frente dele"
    assert result.margin == pytest.approx(behind_c, abs=0.3)  # 0,21 s de viés da perda medida
    assert result.margin < result.margin_free - 2


def test_a_gain_smaller_than_the_traffic_is_not_an_alert():
    """O ganho que some: livre de tráfego a margem é +1,4 s (e haveria alerta); atrás de C, que A
    não consegue passar e que termina na frente de R, é negativa, e não há alerta."""
    free = assess_undercut(undercut_context(12.0), "A", "R", horizon_k=K)
    observed = blocker_field(12.0)
    jammed = assess_undercut(
        build_context(observed, upto_lap=N, total_laps=TOTAL, priors=EXACT), "A", "R", horizon_k=K
    )

    real = blocker_field(12.0, a_stop=N + 1, r_stop=N + K)
    behind_c = clock_at(real, "R", N + K + 2) - clock_at(real, "C", N + K + 2) - 0.7

    assert free.margin == pytest.approx(1.4, abs=1e-3)
    assert AlertTracker().update(free)
    assert jammed.blocked_by == "C"
    assert jammed.margin_free == pytest.approx(1.4, abs=1e-3)
    assert jammed.margin < 0 < jammed.margin_free
    # R, ao parar, cai para trás de C e passa a ser o carro mais próximo de A, mas com a volta de
    # box ele é passável e C continua segurando: o piso é o do carro que A NÃO consegue passar
    assert jammed.margin == pytest.approx(behind_c, abs=0.3)
    assert not AlertTracker().update(jammed)
    assert normal_cdf(jammed.margin_free / jammed.margin_sd) > 0.6 > jammed.probability


def test_a_blocker_that_can_be_passed_does_not_hold_the_attacker_back():
    """O mesmo C, mas com o ritmo de pneu gasto tão ruim que A o passa: nenhum bloqueio, e a margem
    é a livre."""
    c = Plan("C", base=88.0, grid_gap=100.0)  # na idade de hoje, 3 s/volta mais lento que A novo
    rows = chase_field(gap=5.0, world=FLAT, extra=(c,))
    result = assess_undercut(
        build_context(rows, upto_lap=N, total_laps=TOTAL, priors=EXACT), "A", "R", horizon_k=K
    )

    assert result.blocked_by is None
    assert result.margin == pytest.approx(result.margin_free)


# ------------------------------------------------------------------ as regras duras


@pytest.mark.parametrize("status", ["2", "4", "5", "6", "7", "124"])
def test_no_alert_when_the_track_is_not_green(status):
    """Amarela, SC, vermelha, VSC, fim do VSC e o código composto: não se alerta, mas a margem ainda
    é devolvida (para exibir)."""
    world = replace(FLAT, status_laps={N - 1: status, N: status})
    result = assess_undercut(undercut_context(world=world), "A", "R", horizon_k=K)

    assert "pista não verde nas voltas recentes" in result.reasons
    assert not result.eligible
    assert result.margin is not None


def test_a_safety_car_a_few_laps_ago_does_not_block_the_alert():
    world = replace(FLAT, status_laps={N - 6: "4"})

    result = assess_undercut(undercut_context(world=world), "A", "R", horizon_k=K)

    assert "pista não verde nas voltas recentes" not in result.reasons


def test_no_alert_when_most_of_the_field_is_not_on_green_even_if_the_pair_is():
    """A bandeira vermelha ou o SC pode estar no registro de um carro e não do outro: se metade do
    grid está fora da verde, não é hora de undercut."""
    rows = chase_field(gap=5.0, world=FLAT)
    others = {f"D{i:02d}" for i in range(12)}
    rows = [
        replace(r, track_status="4") if r.lap_number == N and r.driver in others else r
        for r in rows
    ]
    ctx = build_context(rows, upto_lap=N, total_laps=TOTAL, priors=EXACT)

    result = assess_undercut(ctx, "A", "R", horizon_k=K)

    assert "pista não verde nas voltas recentes" in result.reasons


def test_no_alert_when_the_rival_has_already_stopped():
    rows = chase_field(r_stop=N - 8, world=FLAT)
    ctx = build_context(rows, upto_lap=N, total_laps=TOTAL, priors=EXACT)

    result = assess_undercut(ctx, "A", "R", horizon_k=K)

    assert "o rival já parou" in result.reasons
    assert not result.eligible


def test_no_alert_when_the_attacker_has_already_stopped():
    rows = chase_field(a_stop=N - 8, world=FLAT)
    ctx = build_context(rows, upto_lap=N, total_laps=TOTAL, priors=EXACT)

    result = assess_undercut(ctx, "A", "R", horizon_k=K)

    assert "o atacante já parou" in result.reasons


def test_no_alert_while_one_of_the_cars_is_in_the_pits():
    rows = chase_field(r_stop=N, world=FLAT)  # a volta N é a de entrada de R
    ctx = build_context(rows, upto_lap=N, total_laps=TOTAL, priors=EXACT)

    result = assess_undercut(ctx, "A", "R", horizon_k=K)

    assert "um dos carros está nos boxes" in result.reasons


def test_no_alert_when_too_few_laps_remain_to_recover_the_loss():
    """Faltam 3 voltas: a perda de ~21 s não volta com 3 voltas de pneu novo, e o horizonte nem
    cabe."""
    rows = chase_field(world=FLAT)
    ctx = build_context(rows, upto_lap=TOTAL - 3, total_laps=TOTAL, priors=EXACT)

    result = assess_undercut(ctx, "A", "R")

    assert "poucas voltas para recuperar a perda" in result.reasons
    assert not result.eligible


def test_the_alert_is_withdrawn_when_the_laps_left_are_fewer_than_the_new_tyres_need():
    """Com a degradação suave (MEDIUM 0,03 s/volta) o pneu novo leva ~15 voltas para repor a
    parada. Na volta 45 (15 voltas sobrando) ainda compensa; na 46 (14), não repõe até o fim: a
    regra
    vem das voltas para recuperar, e não do piso de 5 voltas nem do horizonte."""
    mild = replace(FLAT, slopes={"SOFT": 0.03, "MEDIUM": 0.03, "HARD": 0.0})
    rows = chase_field(a_stop=59, r_stop=59, world=mild)

    def at(lap):
        ctx = build_context(rows, upto_lap=lap, total_laps=TOTAL, priors=EXACT)
        return assess_undercut(ctx, "A", "R")

    enough, short = at(45), at(46)

    assert enough.laps_to_recover == 15  # a de entrada, a de saída e 13 de ganho
    assert short.laps_to_recover is None  # não repõe até o fim da corrida
    assert "poucas voltas para recuperar a perda" not in enough.reasons
    assert "poucas voltas para recuperar a perda" in short.reasons
    assert short.horizon_k + 2 <= TOTAL - 46  # não é o horizonte que barra
    assert UndercutConfig().min_laps_left <= TOTAL - 46  # nem o piso


def test_a_slope_taken_from_all_compounds_never_supports_an_alert(ctx):
    """A previsão com origem "todos os compostos" cobre 68% (e não 95%): é um motivo duro. Testada
    na regra, porque um mundo sintético que a produza com um nível de ritmo disponível é frágil."""
    a, r = ctx.states["A"], ctx.states["R"]

    assert _hard_rules(ctx, a, r, ("piloto", "composto")) == []
    assert _hard_rules(ctx, a, r, ("piloto", "todos os compostos")) == [
        'previsão com origem "todos os compostos"'
    ]


def test_the_leader_has_no_rival_and_no_alert(ctx):
    leader = assess_undercut(ctx, "R")
    nobody = assess_undercut(ctx, "NINGUEM")

    assert leader.reasons == ("sem rival à frente",) and leader.probability is None
    assert nobody.reasons == ("sem rival à frente",)


def test_without_a_pit_loss_estimate_there_is_no_assessment(ctx):
    no_loss = replace(ctx, pit_loss=None)

    result = assess_undercut(no_loss, "A", "R")

    assert result.probability is None and "perda no pit" in result.reasons[0]


def test_a_compound_nobody_has_run_cannot_be_assessed(ctx):
    result = assess_undercut(ctx, "A", "R", new_compound="HYPERSOFT", rival_new_compound="HARD")

    assert result.probability is None and "sem diferença de ritmo" in result.reasons[0]


# ------------------------------------------------------------------ a confiança


def test_the_probability_is_the_normal_cdf_of_the_margin_over_the_inflated_deviation(result):
    assert result.probability == pytest.approx(normal_cdf(result.margin / result.margin_sd))
    parts = np.sqrt(sum(v**2 for v in result.components.values()))
    assert result.margin_sd == pytest.approx(parts * DEFAULT_SD_INFLATION)


def test_inflating_the_deviation_pulls_a_confident_probability_toward_one_half(ctx):
    """O fator de inflação (calibrado no backtest) alarga o desvio: a probabilidade de uma margem
    positiva cai, a de uma negativa sobe, e o sinal da margem não muda."""
    positive = assess_undercut(ctx, "A", "R", horizon_k=K)
    wide = assess_undercut(
        replace(ctx, config=UndercutConfig(sd_inflation=3 * DEFAULT_SD_INFLATION)),
        "A",
        "R",
        horizon_k=K,
    )
    losing_ctx = undercut_context(gap=15.0)
    losing = assess_undercut(losing_ctx, "A", "R", horizon_k=K)
    losing_wide = assess_undercut(
        replace(losing_ctx, config=UndercutConfig(sd_inflation=3 * DEFAULT_SD_INFLATION)),
        "A",
        "R",
        horizon_k=K,
    )

    assert wide.margin == pytest.approx(positive.margin)
    assert wide.margin_sd == pytest.approx(3 * positive.margin_sd)
    assert 0.5 < wide.probability < positive.probability
    assert losing.margin < 0 and losing.probability < losing_wide.probability < 0.5


def test_the_pit_stop_dispersion_enters_twice_and_the_prior_race_spread_is_visible(result):
    """Duas paradas independentes: sqrt(2) vezes a dispersão (no piso, num mundo sem variação)."""
    assert result.components["dispersão entre paradas"] == pytest.approx(np.sqrt(2) * SD_FLOOR)

    spread = build_context(chase_field(world=FLAT), upto_lap=N, total_laps=TOTAL)
    with_spread = assess_undercut(spread, "A", "R", horizon_k=K)

    # a diferença de composto passa de ~0 para o espalhamento entre corridas vezes as voltas
    # líquidas
    assert result.components["diferença de composto"] < 0.01
    assert with_spread.components["diferença de composto"] == pytest.approx(
        DEFAULT_PRIORS.offset_sd * (K - 1), rel=0.05
    )


def test_the_k_error_is_measured_by_refitting_the_whole_model(ctx, result):
    """O componente do `k` é o quanto a margem muda quando o modelo inteiro é refeito com `k +- sd`;
    com o `k` bem identificado é pequeno, e o desvio dele vem de `coefficient_sd`."""
    assert result.kappa == pytest.approx(ctx.pace.kappa)
    assert result.kappa_sd == pytest.approx(coefficient_sd(ctx.pace.degradation.fuel))
    assert 0 < result.components["erro do k"] < 1.0


def test_the_physical_k_variant_carries_the_prior_uncertainty_of_the_constant():
    """Com o `k` físico (peso zero no estimado), o erro do `k` é o do próprio 0,03 s/kg: 50% dele,
    e a margem depende dele bem mais que no encolhimento, que o mede."""
    physical = undercut_context(variant=FuelVariant.FISICO)
    shrunk = undercut_context(variant=FuelVariant.ENCOLHIMENTO)

    a = assess_undercut(physical, "A", "R", horizon_k=K)
    b = assess_undercut(shrunk, "A", "R", horizon_k=K)

    assert a.kappa_sd == pytest.approx(0.5 * physical_coefficient(TOTAL))
    assert a.kappa_sd > 4 * b.kappa_sd
    assert a.components["erro do k"] > 4 * b.components["erro do k"]
    assert a.margin == pytest.approx(b.margin, abs=1e-3)  # o mundo foi plantado com o k físico


# ------------------------------------------------------------------ a álgebra da incerteza


def make_side(
    *,
    age: float,
    mean_age: float,
    se_ref: float,
    slope_sd: float,
    new_slope_sd: float,
    offset: Offset,
    new_compound: str,
    sigma: float = 0.3,
) -> _Side:
    slope = SlopeEstimate(
        "X", "MEDIUM", 0.07, slope_sd * 0.5, slope_sd, slope_sd, sigma, 20, 1.0, "piloto"
    )
    cur = StintLevel("X", "MEDIUM", 1, 90.0, 90.0, se_ref, mean_age, 20, 1, 20, slope, 1.0)
    state = CarState("X", 1000.0, "MEDIUM", int(age), 1, "1", False, False)
    return _Side(
        state, cur, new_compound, offset, 0.04, new_slope_sd, sigma, ("piloto", "composto")
    )


def simulate_margin_sd(a: _Side, r: _Side, k: int, rng, draws: int = 200_000) -> float:
    """O desvio da margem por simulação: sorteia os parâmetros incertos e soma os tempos de volta
    EXPLICITAMENTE (o nível na idade zero é `L - idade_média * inclinação`), sem nenhuma fórmula
    de coeficientes. Os dois lados têm a mesma diferença de composto sorteada UMA vez, na parte
    compartilhada."""

    def total(side: _Side, old_ages, new_count, shared):
        level = rng.normal(0, side.cur.se_ref, draws)
        slope_err = rng.normal(0, side.cur.slope.predictive_sd, draws)
        new_slope_err = rng.normal(0, side.slope_new_sd, draws)
        own_offset = rng.normal(0, side.offset.tau, draws)
        mean_age = side.cur.mean_age
        old = sum(level + slope_err * (age - mean_age) for age in old_ages)
        level0_new_err = level - mean_age * slope_err + shared + own_offset
        new = sum(level0_new_err + new_slope_err * j for j in range(1, new_count + 1))
        return old + new

    shared = rng.normal(0, np.hypot(a.offset.se, a.offset.race_sd), draws)
    same = (a.cur.compound, a.new_compound) == (r.cur.compound, r.new_compound)
    shared_r = shared if same else rng.normal(0, np.hypot(r.offset.se, r.offset.race_sd), draws)
    a_total = total(a, [a.age + 1], k + 1, shared)
    r_total = total(r, [r.age + i for i in range(1, k + 1)], 2, shared_r)
    return float(np.std(r_total - a_total))


@pytest.mark.parametrize("new_r", ["HARD", "SOFT"])
def test_the_linear_propagation_matches_a_simulation_of_the_same_model(new_r):
    """A fórmula dos coeficientes líquidos (`_margin_sd`) contra o desvio sorteado, com e sem o par
    de compostos compartilhado. Sem ruído de volta nem parada, só a parte dos parâmetros."""
    rng = np.random.default_rng(7)
    offset = Offset("MEDIUM", "HARD", 0.3, 0.08, 0.05, 6, "pares", 0.2)
    other = Offset("MEDIUM", "SOFT", -0.4, 0.1, 0.06, 6, "pares", 0.25)
    a = make_side(
        age=30, mean_age=22, se_ref=0.03, slope_sd=0.004, new_slope_sd=0.006, offset=offset,
        new_compound="HARD",
    )  # fmt: skip
    r = make_side(
        age=32, mean_age=20, se_ref=0.05, slope_sd=0.006, new_slope_sd=0.008,
        offset=offset if new_r == "HARD" else other, new_compound=new_r,
    )  # fmt: skip
    loss = PitLoss(21.0, 0.0, 0.0, 21.0, 4, 0, None, 1.0, "corrida")
    k = 6

    parts = _margin_sd(a, r, k, loss)
    parameters = np.sqrt(
        parts["ritmo do atacante"] ** 2
        + parts["ritmo do rival"] ** 2
        + parts["diferença de composto"] ** 2
    )

    assert parameters == pytest.approx(simulate_margin_sd(a, r, k, rng), rel=0.02)


def test_sharing_the_compound_pair_shrinks_the_compound_term_by_the_net_laps():
    """Os dois trocam MEDIUM para HARD: o erro da diferença é o mesmo e só a diferença de voltas no
    pneu novo (K+1 contra 2) conta; com pares diferentes os termos somam em quadratura."""
    shared_offset = Offset("MEDIUM", "HARD", 0.3, 1.0, 0.0, 6, "pares", 0.0)
    other_offset = Offset("MEDIUM", "SOFT", 0.3, 1.0, 0.0, 6, "pares", 0.0)
    kwargs = {"age": 30, "mean_age": 20, "se_ref": 0.0, "slope_sd": 0.0, "new_slope_sd": 0.0}
    a = make_side(**kwargs, offset=shared_offset, new_compound="HARD", sigma=0.0)
    same = make_side(**kwargs, offset=shared_offset, new_compound="HARD", sigma=0.0)
    different = make_side(**kwargs, offset=other_offset, new_compound="SOFT", sigma=0.0)
    loss = PitLoss(21.0, 0.0, 0.0, 21.0, 4, 0, None, 1.0, "corrida")
    k = 6

    assert _margin_sd(a, same, k, loss)["diferença de composto"] == pytest.approx(k - 1)
    assert _margin_sd(a, different, k, loss)["diferença de composto"] == pytest.approx(
        np.hypot(k + 1, 2)
    )


def test_the_slope_enters_with_the_net_coefficient_not_with_the_old_laps_alone():
    """O erro da inclinação do atacante mexe nas voltas velhas (idade acima da média) e, pela
    extrapolação à idade zero, nas K+1 novas, com sinal oposto: o coeficiente é
    (idade+1 - média) - (K+1)*média."""
    offset = Offset("MEDIUM", "HARD", 0.0, 0.0, 0.0, 6, "pares", 0.0)
    kwargs = {"se_ref": 0.0, "new_slope_sd": 0.0, "offset": offset, "new_compound": "HARD"}
    a = make_side(age=30, mean_age=22, slope_sd=0.01, **kwargs)
    r = make_side(age=30, mean_age=22, slope_sd=0.0, **kwargs)
    loss = PitLoss(21.0, 0.0, 0.0, 21.0, 4, 0, None, 1.0, "corrida")
    k = 6

    coefficient = (30 + 1 - 22) - (k + 1) * 22

    assert _margin_sd(a, r, k, loss)["ritmo do atacante"] == pytest.approx(abs(coefficient) * 0.01)


def test_lap_noise_is_the_square_root_of_the_laps_in_each_car():
    offset = Offset("MEDIUM", "HARD", 0.0, 0.0, 0.0, 6, "pares", 0.0)
    kwargs = {"se_ref": 0.0, "slope_sd": 0.0, "new_slope_sd": 0.0, "offset": offset}
    a = make_side(age=30, mean_age=22, new_compound="HARD", sigma=0.3, **kwargs)
    r = make_side(age=30, mean_age=22, new_compound="HARD", sigma=0.4, **kwargs)
    loss = PitLoss(21.0, 0.0, 0.0, 21.0, 4, 0, None, 1.0, "corrida")

    noise = _margin_sd(a, r, 6, loss)["ruído de volta"]

    assert noise == pytest.approx(np.sqrt(8 * (0.3**2 + 0.4**2)))


# ------------------------------------------------------------------ histerese


def assessment(
    probability: float | None, *, reasons: tuple[str, ...] = (), rival: str = "R"
) -> UndercutAssessment:
    return UndercutAssessment(
        "A", rival, N, 1.0, 1.0, 1.0, probability, K, 21.0, None, None, None, 6, reasons=reasons
    )


def test_the_alert_turns_on_at_the_threshold_and_only_off_below_the_release():
    """0,6 liga, 0,5 desliga: entre os dois o estado anterior se mantém, e a probabilidade que
    oscila em torno do limiar NÃO faz o alerta piscar."""
    tracker = AlertTracker(UndercutConfig(threshold=0.6, release=0.5))
    sequence = [0.55, 0.60, 0.55, 0.52, 0.49, 0.55, 0.58, 0.61, 0.52]

    states = [tracker.update(assessment(p)) for p in sequence]

    assert states == [False, True, True, True, False, False, False, True, True]


def test_without_hysteresis_the_same_sequence_would_blink():
    """O contraponto: limiar e desligamento iguais (sem histerese) trocam de estado 4 vezes numa
    sequência em que a histerese troca 2."""
    plain = AlertTracker(UndercutConfig(threshold=0.6, release=0.6))
    hysteresis = AlertTracker(UndercutConfig(threshold=0.6, release=0.5))
    sequence = [0.58, 0.62, 0.58, 0.62, 0.58, 0.62, 0.45]

    def flips(tracker):
        states = [tracker.update(assessment(p)) for p in sequence]
        return sum(a != b for a, b in zip(states, states[1:], strict=False))

    assert flips(plain) > flips(hysteresis)
    assert flips(hysteresis) == 2  # liga na 2ª, desliga na última


def test_a_hard_rule_turns_the_alert_off_at_once_and_clears_the_state():
    tracker = AlertTracker()
    assert tracker.update(assessment(0.9))
    assert tracker.active() == {("A", "R")}

    assert not tracker.update(assessment(0.9, reasons=("o rival já parou",)))
    assert tracker.active() == set()
    # o estado foi apagado: 0,55 (acima da liberação, abaixo do limiar) não religa
    assert not tracker.update(assessment(0.55))


def test_a_missing_probability_never_alerts():
    tracker = AlertTracker()

    assert not tracker.update(assessment(None))


def test_a_new_rival_is_a_new_situation_and_unseen_pairs_are_forgotten():
    tracker = AlertTracker()
    assert tracker.update(assessment(0.9, rival="R"))

    assert not tracker.update(assessment(0.55, rival="S"))  # outro par: sem o estado do anterior
    on = tracker.update_all([assessment(0.9, rival="S")])
    assert on == {("A", "S")}  # ("A", "R") não foi avaliado nesta volta e é esquecido
    assert tracker.active() == {("A", "S")}


def test_the_config_rejects_a_release_above_the_threshold_and_other_nonsense():
    for bad in (
        {"threshold": 0.5, "release": 0.6},
        {"threshold": 1.0},
        {"release": 0.0},
        {"k_min": 5, "k_max": 3},
        {"k_min": 0},
        {"sd_inflation": 0.0},
    ):
        with pytest.raises(ValueError):
            UndercutConfig(**bad)


# ------------------------------------------------------------------ vazamento do futuro


def corrupt_after(rows: list[LapRow], lap: int) -> list[LapRow]:
    """Troca tudo depois da volta `lap` por lixo: tempos, compostos, status, relógio, paradas."""
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
            position=1,
        )
        if r.lap_number > lap
        else r
        for r in rows
    ]


def test_the_assessment_at_lap_n_ignores_everything_after_n():
    """O teste do futuro corrompido: todas as voltas depois de N viram lixo (inclusive paradas que
    não existiram) e a avaliação de TODOS os pares é idêntica, valor por valor."""
    world = World(total_laps=TOTAL, noise=0.12, seed=5)
    clean = chase_field(gap=3.0, world=world)
    dirty = corrupt_after(clean, N)

    expected = assess_all(build_context(clean, upto_lap=N, total_laps=TOTAL))
    actual = assess_all(build_context(dirty, upto_lap=N, total_laps=TOTAL))

    assert actual == expected
    assert any(a.margin is not None for a in expected)  # o teste não passa por estar tudo vazio


def test_the_leakage_test_would_catch_a_context_that_reads_the_future():
    """A prova de que o teste acima morde: se o contexto recebesse as linhas SEM o corte, a
    avaliação dos mesmos dados mudaria."""
    clean = chase_field(gap=3.0, world=World(total_laps=TOTAL, noise=0.12, seed=5))
    dirty = corrupt_after(clean, N)
    expected = assess_all(build_context(clean, upto_lap=N, total_laps=TOTAL))

    uncut = build_context(dirty, upto_lap=N + 10, total_laps=TOTAL)

    assert assess_all(uncut) != expected


# ------------------------------------------------------------------ as peças


def test_car_ahead_is_the_nearest_car_with_the_smaller_clock(ctx):
    states = car_states(ctx.rows, N)

    assert car_ahead(states, "A") == "R" or states["R"].clock_s < states["A"].clock_s
    assert (
        car_ahead(states, "R") is None
        or states[car_ahead(states, "R")].clock_s < states["R"].clock_s
    )
    assert car_ahead(states, "NINGUEM") is None
    ordered = sorted(states.values(), key=lambda s: s.clock_s)
    for ahead, behind in zip(ordered, ordered[1:], strict=False):
        assert car_ahead(states, behind.driver) == ahead.driver


def test_the_expected_stint_length_comes_from_the_finished_stints_of_the_compound():
    """Os 12 carros lentos pararam nas voltas 12..23 (idade da entrada = a volta), metade deles do
    MEDIUM (os ímpares): as entradas 13, 15, ..., 23, mediana 18."""
    rows = chase_field(world=FLAT)

    length, source = expected_stint_length(rows, N, "MEDIUM", TOTAL)

    assert length == pytest.approx(18.0)
    assert source == "stints concluídos do composto"


def test_the_expected_stint_length_falls_back_and_never_reads_the_future():
    rows = chase_field(world=FLAT)

    early, source_early = expected_stint_length(rows, 5, "MEDIUM", TOTAL)
    other, source_other = expected_stint_length(rows, N, "SOFT", TOTAL)
    after, _ = expected_stint_length(corrupt_after(rows, N), N, "MEDIUM", TOTAL)

    assert early == TOTAL / 2 and "nenhum stint concluído" in source_early
    assert other == pytest.approx(17.0)  # os pares: entradas 12, 14, ..., 22 (SOFT)
    assert source_other == "stints concluídos do composto"
    assert after == pytest.approx(18.0)


def test_the_likely_next_compound_is_the_most_used_after_the_first_stop():
    rows = chase_field(world=FLAT)

    assert likely_next_compound(rows, N, "MEDIUM") == "HARD"
    assert likely_next_compound(rows, N, "HARD") in {"MEDIUM", "SOFT"}
    assert likely_next_compound(rows, 5, "MEDIUM") == "SOFT"  # antes de qualquer parada: o que há
    assert likely_next_compound([], N, "MEDIUM") is None


def test_the_flat_pace_is_the_median_of_the_last_clean_laps(ctx):
    pace = flat_pace(ctx.rows, N)

    assert pace["R"] == pytest.approx(
        88.0 + SLOPE * N - physical_coefficient(TOTAL) * (N - 1), abs=0.3
    )
    assert set(pace) >= {"A", "R", "D00"}


def test_assess_all_assesses_every_car_against_the_one_ahead(ctx):
    results = assess_all(ctx)

    assert [a.attacker for a in results] == sorted(ctx.states)
    by_attacker = {a.attacker: a for a in results}
    first = min(ctx.states.values(), key=lambda s: s.clock_s).driver
    assert by_attacker[first].reasons == ("sem rival à frente",)
    assert by_attacker["A"].rival == "R"
