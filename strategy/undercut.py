"""Alerta de undercut: parar AGORA faz o atacante sair à frente do carro à frente dele?

O UNDERCUT. O atacante (A) está logo atrás do rival (R). Se A para primeiro, perde a perda no pit
(`P`, ~22 s), mas anda as voltas seguintes com pneu novo contra o pneu gasto do rival. Quando R
também parar (e pagar o seu `P`), a conta é: quem sai na frente? É essa MARGEM que se calcula.

A MECÂNICA, num horizonte de `K` voltas (até o rival PROVAVELMENTE parar: `expected_stint_length`):
* A para na volta N+1 (entrada), sai na N+2 (saída: pneu novo e frio) e anda as voltas seguintes no
  composto novo. R fica fora `K` voltas, entra na N+K, sai na N+K+1 e anda a N+K+2 no pneu novo.
  Compara-se o relógio dos dois ao fim da volta N+K+2 (`K + 2` voltas cada), com R já parado.
* A perda `P` já INCLUI o pneu frio da volta de saída: é medida (entrada + saída) contra o ritmo
  normal. Sua média entra nos dois e CANCELA na diferença; sobra a dispersão ENTRE as duas paradas.
* O RETORNO À PISTA. O relógio de A ao fim da volta de saída (`session_time_ms`, o mesmo para todos
  os carros) diz atrás de quem ele volta. Atrás de um carro MAIS LENTO que ele (pneu gasto, ou o
  próprio R), o ganho de ritmo não se realiza: A fica preso, a menos que a vantagem por volta
  chegue a `pass_advantage_s`. A margem usa esse tráfego; `margin_free` mostra a que seria sem ele.

O NÍVEL DE RITMO sai do `PaceModel` (tempo corrigido na idade zero, por piloto e composto, encolhido
para a média do composto). A e R entram pela DIFERENÇA de nível: a evolução da pista aparece nos
dois e se cancela, e o combustível cancela porque os dois usam o mesmo `k`.

A CONFIANÇA: `P(margem > 0) = Phi(margem / desvio)`. O desvio é a propagação LINEAR exata das
incertezas pela soma dos tempos de volta (`_margin_sd`), sem tratar como independentes coisas que
são a mesma estimativa:
  o nível e a inclinação de A (e de R) aparecem nas voltas do pneu velho E, pela extrapolação à
  idade zero, nas do pneu novo, com sinais opostos: o termo de inclinação é o coeficiente líquido;
  a diferença de composto é um erro COMPARTILHADO quando os dois trocam o mesmo par de compostos
  (e só o `tau` de cada piloto é individual);
  a inclinação do pneu novo usa `predictive_sd` (stint futuro), nunca o erro-padrão;
  o ruído de cada volta; a dispersão entre as duas paradas;
  o erro do `k`, medido refazendo TODO o modelo com `k +- sd` (o resultado de comparar pneu velho
  com novo depende dele).
O total é multiplicado por `sd_inflation`, porque a cobertura medida ainda é 87-91%.

QUANDO NÃO ALERTAR (cada motivo vem em `reasons`, e a margem ainda é devolvida para exibir):
  sem rival / sem dados; o rival (ou o atacante) já parou; carros nos boxes; pista não verde nas
  voltas recentes (SC, VSC, vermelha, amarela); previsão com origem "todos os compostos" (68% de
  cobertura, mal calibrada); poucas voltas para recuperar a perda.

HISTERESE. `AlertTracker` liga acima de `threshold` e só desliga abaixo de `release`, para o alerta
não piscar com a probabilidade oscilando em torno do limiar. Uma regra dura desliga na hora e apaga
o estado.

SEM VAZAMENTO. `build_context` corta o futuro como PRIMEIRO passo; tudo o que vem depois (níveis,
perda no pit, ritmo dos outros carros, composto provável) sai de `ctx.rows`. O teste de corrupção
(tests/strategy/test_undercut.py) altera todas as voltas depois de N e exige o mesmo resultado.

APROXIMAÇÕES (não validadas contra a realidade; ver docs/undercut.md): os outros carros ficam fora,
sem parar, no pneu atual, durante o horizonte; um bloqueador por volta (o carro à frente do relógio
de A); ao ficar preso, A segue a `follow_gap_s` do bloqueador até poder passar; o composto seguinte
é o mais provável pelos stints já feitos (a equipe escolhe); sem Safety Car (o Monte Carlo trata do
risco de SC depois).
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from strategy.fuel import (
    FuelVariant,
    coefficient_sd,
    dry_clean_laps,
    laps_until,
)
from strategy.laps import DRY_COMPOUNDS, LapRow
from strategy.pace import DEFAULT_SIGMA, Offset, PaceModel, StintLevel, fit_pace
from strategy.pitloss import PitLoss, estimate_pit_loss
from strategy.priors import DEFAULT_PRIORS, Priors
from strategy.snapshot import MAX_SILENCE_LAPS, CarState, car_states, state_until

# PROVISÓRIO. O fator que faria o intervalo da INCLINAÇÃO de um stint futuro cobrir 95% (o quantil
# 95% de |z| nos 251 stints, com o `k` do modelo, dá 1,29). Vem da calibração da inclinação, não da
# margem do undercut: o backtest recalibra com `strategy.undercut_eval.calibrate_inflation` e deve
# sobrescrever este valor.
DEFAULT_SD_INFLATION = 1.3
GREEN = "1"
FLAT_LAPS = 3  # voltas recentes para o ritmo "plano" de um carro sem modelo
NOT_GREEN_SHARE = 0.5  # fração de carros fora da bandeira verde que vale como "pista não verde"
ALL_COMPOUNDS = "todos os compostos"


@dataclass(frozen=True, slots=True)
class UndercutConfig:
    threshold: float = 0.6  # probabilidade mínima para LIGAR o alerta
    release: float = 0.5  # histerese: um alerta ligado só desliga abaixo disto
    k_min: int = 3  # o rival para daqui a, no mínimo, K voltas
    k_max: int = 10
    min_laps_left: int = 5  # piso de voltas restantes
    pass_advantage_s: float = 0.7  # vantagem (s/volta) com que o atacante passa um carro mais lento
    follow_gap_s: float = 0.7  # distância mantida atrás do carro que bloqueia
    sd_inflation: float = DEFAULT_SD_INFLATION

    def __post_init__(self) -> None:
        if not 0 < self.release <= self.threshold < 1:
            raise ValueError("é preciso 0 < release <= threshold < 1 (a histerese vem de release)")
        if not 1 <= self.k_min <= self.k_max:
            raise ValueError("é preciso 1 <= k_min <= k_max")
        if self.sd_inflation <= 0:
            raise ValueError("sd_inflation deve ser positivo")


# ---------------------------------------------------------------- o estado da corrida na volta N


def car_ahead(states: dict[str, CarState], driver: str) -> str | None:
    """O carro imediatamente à frente: o de maior relógio entre os de relógio menor que o dele, na
    MESMA volta (no estado por instante, um carro uma volta atrás tem relógio menor e não está à
    frente; um que o lapou também não é "o carro à frente" numa disputa)."""
    me = states.get(driver)
    if me is None:
        return None
    ahead = [s for s in states.values() if s.laps_done == me.laps_done and s.clock_s < me.clock_s]
    return max(ahead, key=lambda s: s.clock_s).driver if ahead else None


def expected_stint_length(
    rows: Sequence[LapRow], upto_lap: int, compound: str, total_laps: int
) -> tuple[float, str]:
    """Quantas voltas de idade um stint desse composto costuma ter, pelos stints JÁ CONCLUÍDOS (o
    piloto tem um stint seguinte) na corrida até a volta N. Sem nenhum do composto, usa todos os
    compostos; sem nenhum, metade da corrida (uma parada)."""
    visible = laps_until(rows, upto_lap)
    last: dict[tuple[str, int], LapRow] = {}
    top: dict[str, int] = defaultdict(int)
    for lap in visible:
        if lap.stint is None or lap.tyre_life is None:
            continue
        key = (lap.driver, lap.stint)
        if key not in last or lap.lap_number > last[key].lap_number:
            last[key] = lap
        top[lap.driver] = max(top[lap.driver], lap.stint)
    done = [lap for (driver, stint), lap in last.items() if stint < top[driver]]
    same = [lap.tyre_life for lap in done if lap.compound == compound]
    if same:
        return float(median(same)), "stints concluídos do composto"
    if done:
        return float(
            median(lap.tyre_life for lap in done)
        ), "stints concluídos (todos os compostos)"
    return total_laps / 2, "metade da corrida (nenhum stint concluído)"


def likely_next_compound(rows: Sequence[LapRow], upto_lap: int, current: str) -> str | None:
    """O composto mais provável do próximo stint: o mais usado nos stints depois da primeira parada
    (e que não seja o atual); sem nenhum, o composto de seco com mais voltas na corrida. Heurística:
    a regra de dois compostos obriga a trocar, mas qual deles é escolha da equipe."""
    visible = laps_until(rows, upto_lap)
    after_stop: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for lap in visible:
        if lap.compound not in DRY_COMPOUNDS or lap.compound == current:
            continue
        total[lap.compound] += 1
        if lap.stint is not None and lap.stint >= 2:
            after_stop[lap.compound] += 1
    pool = after_stop or total
    return max(pool, key=lambda c: (pool[c], c)) if pool else None


def flat_pace(rows: Sequence[LapRow], upto_lap: int) -> dict[str, float]:
    """O ritmo "plano" de cada piloto: a mediana das últimas voltas limpas. Serve para projetar
    os carros que não têm modelo (só entram como possível bloqueio)."""
    recent: dict[str, list[float]] = defaultdict(list)
    for lap in sorted(dry_clean_laps(laps_until(rows, upto_lap)), key=lambda x: x.lap_number):
        recent[lap.driver].append(lap.lap_time_ms / 1000)
    return {driver: float(median(times[-FLAT_LAPS:])) for driver, times in recent.items()}


# ---------------------------------------------------------------- o contexto de uma volta


@dataclass(slots=True)
class LapContext:
    """Tudo o que se ajusta UMA vez por volta e serve a todos os pares atacante/rival."""

    rows: list[LapRow]  # só até a volta N
    upto_lap: int
    total_laps: int
    config: UndercutConfig
    priors: Priors
    pace: PaceModel
    pace_up: PaceModel  # o mesmo modelo com k + sd
    pace_down: PaceModel  # o mesmo modelo com k - sd
    kappa_sd: float
    pit_loss: PitLoss | None
    states: dict[str, CarState]
    flat: dict[str, float]
    as_of_ms: int | None = None  # o instante do estado (None = por contagem de voltas)

    @property
    def laps_left(self) -> int:
        return self.total_laps - self.upto_lap


def build_context(
    rows: Sequence[LapRow],
    *,
    upto_lap: int,
    total_laps: int,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    priors: Priors = DEFAULT_PRIORS,
    config: UndercutConfig | None = None,
    as_of_ms: int | None = None,
    max_silence_laps: float = MAX_SILENCE_LAPS,
) -> LapContext:
    """O contexto da volta `upto_lap`. Com `as_of_ms` o estado é o do INSTANTE (só as voltas
    completadas até ele, cada piloto na sua última volta: o que um sistema ao vivo saberia); sem
    ele,
    é o estado por contagem de voltas (cada piloto na volta N dele), que dá ao retardatário
    informação de depois do instante e fica para os testes de mecânica e o diagnóstico."""
    if as_of_ms is None:
        visible = laps_until(rows, upto_lap)  # PRIMEIRO passo: nada do futuro passa daqui
        states = car_states(visible, upto_lap)
    else:
        snapshot = state_until(rows, as_of_ms, max_silence_laps=max_silence_laps)
        visible = [r for r in snapshot.rows if r.lap_number <= upto_lap]
        states = snapshot.cars
    pace = fit_pace(
        visible, upto_lap=upto_lap, total_laps=total_laps, variant=variant, priors=priors
    )
    kappa_sd = coefficient_sd(pace.degradation.fuel)

    def shifted(delta: float) -> PaceModel:
        if kappa_sd < 1e-9:
            return pace
        return fit_pace(
            visible,
            upto_lap=upto_lap,
            total_laps=total_laps,
            variant=variant,
            priors=priors,
            kappa=pace.kappa + delta,
        )

    return LapContext(
        rows=list(visible),
        upto_lap=upto_lap,
        total_laps=total_laps,
        config=config or UndercutConfig(),
        priors=priors,
        pace=pace,
        pace_up=shifted(kappa_sd),
        pace_down=shifted(-kappa_sd),
        kappa_sd=kappa_sd,
        pit_loss=estimate_pit_loss(
            visible, upto_lap=upto_lap, kappa=pace.kappa, prior=priors.pit_loss
        ),
        states=states,
        flat=flat_pace(visible, upto_lap),
        as_of_ms=as_of_ms,
    )


# ---------------------------------------------------------------- o resultado


@dataclass(frozen=True, slots=True)
class UndercutAssessment:
    attacker: str
    rival: str | None
    lap: int
    margin: float | None  # s; + = o atacante termina o horizonte à frente do rival (com o tráfego)
    margin_free: float | None  # s, a mesma margem sem o tráfego
    margin_sd: float | None  # desvio total, já inflado
    probability: float | None  # P(margem > 0)
    horizon_k: int | None
    pit_loss: float | None
    rejoin_gap_s: float | None  # a que distância (s) atrás de outro carro A volta à pista
    rejoin_behind: str | None  # e atrás de quem (None = volta à frente de todos)
    blocked_by: str | None  # quem o prende depois (None = ninguém)
    laps_to_recover: int | None
    components: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()  # por que NÃO se alerta (vazio = pode alertar)
    compounds: tuple[str | None, str | None, str | None, str | None] = (None, None, None, None)
    kappa: float | None = None
    kappa_sd: float | None = None

    @property
    def eligible(self) -> bool:
        """Sem regra dura e com probabilidade. O limiar e a histerese são do `AlertTracker`."""
        return not self.reasons and self.probability is not None


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------- um lado (atacante ou rival)


@dataclass(frozen=True, slots=True)
class _Side:
    state: CarState
    cur: StintLevel
    new_compound: str
    offset: Offset
    slope_new: float
    slope_new_sd: float
    sigma: float
    sources: tuple[str, str]  # origem da inclinação do stint atual e do novo

    @property
    def age(self) -> int:
        assert self.state.age is not None
        return self.state.age

    @property
    def level0_new(self) -> float:
        return self.cur.level0 + self.offset.delta


def _side(pace: PaceModel, state: CarState, new_compound: str | None) -> tuple[_Side | None, str]:
    cur = pace.current(state.driver)
    if cur is None or state.stint is None or cur.stint != state.stint or state.age is None:
        return None, f"sem ritmo atual de {state.driver}"
    if new_compound is None:
        return None, f"sem composto provável para {state.driver}"
    offset = pace.offset(cur.compound, new_compound)
    slope = pace.degradation.slope(state.driver, new_compound)
    if offset is None or slope is None:
        return None, f"sem diferença de ritmo {cur.compound}->{new_compound}"
    sigma = cur.slope.sigma_lap or pace.degradation.sigma_lap or DEFAULT_SIGMA
    side = _Side(
        state, cur, new_compound, offset, slope.slope, slope.predictive_sd, float(sigma),
        (cur.slope.source, slope.source),
    )  # fmt: skip
    return side, ""


def _old_times(side: _Side, n: int, count: int, kappa: float) -> np.ndarray:
    """Os tempos previstos das próximas `count` voltas (N+1..N+count), no pneu ATUAL."""
    steps = np.arange(1, count + 1)
    return side.cur.level0 + side.cur.slope.slope * (side.age + steps) - kappa * (n + steps - 1)


def _new_times(side: _Side, n: int, out_lap: int, count: int, kappa: float) -> np.ndarray:
    """Os tempos previstos de `count` voltas no pneu NOVO, de idade 1 a `count`. A de idade 1 é a
    volta N+`out_lap` (a de saída dos boxes); a perda `P` é somada à parte."""
    ages = np.arange(1, count + 1)
    return side.level0_new + side.slope_new * ages - kappa * (n + out_lap + ages - 2)


# ---------------------------------------------------------------- a projeção determinística


@dataclass(frozen=True, slots=True)
class _Projection:
    margin: float
    margin_free: float
    rejoin_gap_s: float | None
    rejoin_behind: str | None
    blocked_by: str | None
    a: _Side
    r: _Side


def _others(
    ctx: LapContext, pace: PaceModel, skip: set[str], horizon: int
) -> dict[str, np.ndarray]:
    """O relógio projetado (voltas 0..H) dos OUTROS carros, supondo que ficam fora no pneu atual. O
    índice 0 é o relógio ao fim da volta N; um carro que ainda está na volta anterior (estado por
    instante) primeiro anda as `lag` voltas que lhe faltam, pelo modelo, para chegar à N."""
    out = {}
    n = ctx.upto_lap
    for driver, state in ctx.states.items():
        if driver in skip or state.laps_done > n:
            continue  # quem já lapou o grupo não disputa posição com ele
        lag = n - state.laps_done
        steps = np.arange(1, lag + horizon + 1)
        cur = pace.current(driver)
        if cur is not None and state.stint == cur.stint and state.age is not None:
            laps = state.laps_done + steps
            times = cur.level0 + cur.slope.slope * (state.age + steps) - pace.kappa * (laps - 1)
        elif driver in ctx.flat:
            times = np.full(lag + horizon, ctx.flat[driver])
        else:
            continue
        cumulative = state.clock_s + np.cumsum(times)
        start = cumulative[lag - 1] if lag > 0 else state.clock_s
        out[driver] = np.concatenate([[start], cumulative[lag:]])
    return out


def _project(
    ctx: LapContext,
    pace: PaceModel,
    a_state: CarState,
    r_state: CarState,
    new_a: str | None,
    new_r: str | None,
    k: int,
) -> tuple[_Projection | None, str]:
    sa, why = _side(pace, a_state, new_a)
    if sa is None:
        return None, why
    sr, why = _side(pace, r_state, new_r)
    if sr is None:
        return None, why
    loss = ctx.pit_loss
    assert loss is not None
    cfg, n, kappa, horizon = ctx.config, ctx.upto_lap, pace.kappa, k + 2

    # --- atacante: entra na N+1, sai na N+2 (pneu novo, frio) e anda as voltas seguintes no novo.
    a_old = _old_times(sa, n, 1, kappa)[0]
    a_new = _new_times(sa, n, 2, k + 1, kappa)  # idades 1..k+1, nas voltas N+2..N+k+2
    free = np.full(horizon + 1, np.nan)  # relógio de A ao fim de cada volta; [1] não é usado
    free[0] = a_state.clock_s
    free[2] = a_state.clock_s + a_old + a_new[0] + loss.median
    for j in range(3, horizon + 1):
        free[j] = free[j - 1] + a_new[j - 2]

    # --- rival: fica fora k voltas, entra na N+k, sai na N+k+1 e anda a N+k+2 no novo.
    r_old = _old_times(sr, n, k, kappa)
    r_new = _new_times(sr, n, k + 1, 2, kappa)
    r_path = np.empty(horizon + 1)
    r_path[0] = r_state.clock_s
    r_path[1 : k + 1] = r_state.clock_s + np.cumsum(r_old)
    r_path[k + 1] = r_path[k] + r_new[0] + loss.median
    r_path[k + 2] = r_path[k + 1] + r_new[1]

    # --- o retorno à pista e o tráfego, pelo relógio dos outros carros
    paths = _others(ctx, pace, {a_state.driver, r_state.driver}, horizon)
    paths[r_state.driver] = r_path
    rejoin_gap, rejoin_behind = None, None
    ahead_at_rejoin = [d for d, p in paths.items() if p[2] < free[2]]
    if ahead_at_rejoin:
        rejoin_behind = max(ahead_at_rejoin, key=lambda d: paths[d][2])
        rejoin_gap = float(free[2] - paths[rejoin_behind][2])

    # A não termina a volta à frente de NENHUM carro à frente que não consiga passar (a vantagem de
    # ritmo sobre ele é menor que `pass_advantage_s`): o piso é o mais alto entre eles. Olhar só o
    # carro mais próximo erra quando ele é passável (o rival em volta de box) e o de trás o prende.
    clock = free.copy()
    blocked_by = None
    for j in range(3, horizon + 1):
        clock[j] = clock[j - 1] + a_new[j - 2]
        floors = {
            d: p[j] + cfg.follow_gap_s
            for d, p in paths.items()
            if p[j - 1] < clock[j - 1] and (p[j] - p[j - 1]) - a_new[j - 2] < cfg.pass_advantage_s
        }
        if floors:
            holder, floor = max(floors.items(), key=lambda item: item[1])
            if clock[j] < floor:
                clock[j] = floor
                blocked_by = blocked_by or holder

    return (
        _Projection(
            margin=float(r_path[horizon] - clock[horizon]),
            margin_free=float(r_path[horizon] - free[horizon]),
            rejoin_gap_s=rejoin_gap,
            rejoin_behind=rejoin_behind,
            blocked_by=blocked_by,
            a=sa,
            r=sr,
        ),
        "",
    )


# ---------------------------------------------------------------- a incerteza da margem


def _margin_sd(a: _Side, r: _Side, k: int, loss: PitLoss) -> dict[str, float]:
    """A incerteza da margem (R - A), por componente, SEM o `k` e SEM a inflação.

    A margem é linear nos parâmetros, então a propagação é exata. Para cada lado, com `L` o nível no
    ritmo atual (erro `se_ref`), `e` a inclinação do pneu atual, `d` a diferença de composto e `v` a
    inclinação do pneu novo, o nível na idade zero é `L - idade_média * e`, e a soma dos tempos é
        (n_velhas + n_novas) * L
      + (soma(idades_velhas - idade_média) - n_novas * idade_média) * e     <- coeficiente LÍQUIDO
      + n_novas * d  +  soma(idades_novas) * v.
    Tratar nível e inclinação como independentes e somar em quadratura (a aproximação óbvia) erra,
    porque a mesma inclinação mexe nas voltas velhas e, pela extrapolação à idade zero, nas novas.
    """
    n_new_a, n_new_r = k + 1, 2
    old_a = np.array([a.age + 1], dtype=float)
    old_r = r.age + np.arange(1, k + 1, dtype=float)

    def own(side: _Side, old: np.ndarray, n_new: int) -> float:
        mean_age = side.cur.mean_age
        level = (len(old) + n_new) * side.cur.se_ref
        slope = (float(np.sum(old - mean_age)) - n_new * mean_age) * side.cur.slope.predictive_sd
        new_slope = n_new * (n_new + 1) / 2 * side.slope_new_sd
        driver_offset = n_new * side.offset.tau
        return float(np.sqrt(level**2 + slope**2 + new_slope**2 + driver_offset**2))

    def shared_sd(side: _Side) -> float:
        return float(np.hypot(side.offset.se, side.offset.race_sd))

    same_pair = (a.cur.compound, a.new_compound) == (r.cur.compound, r.new_compound)
    if same_pair:  # o mesmo erro da diferença de composto nos dois: só o líquido de voltas conta
        compound = abs(n_new_r - n_new_a) * shared_sd(a)
    else:
        compound = float(np.hypot(n_new_a * shared_sd(a), n_new_r * shared_sd(r)))
    return {
        "ritmo do atacante": own(a, old_a, n_new_a),
        "ritmo do rival": own(r, old_r, n_new_r),
        "diferença de composto": float(compound),
        "ruído de volta": float(np.sqrt((k + 2) * (a.sigma**2 + r.sigma**2))),
        "dispersão entre paradas": float(np.sqrt(2) * loss.sd),
    }


# ---------------------------------------------------------------- recuperar a perda


def _laps_to_recover(ctx: LapContext, a: _Side, loss: float) -> int | None:
    """Em quantas voltas (contando a de entrada e a de saída) o pneu novo repõe o que a parada
    custou, contra continuar no gasto. None se não repõe até o fim da corrida."""
    kappa, n = ctx.pace.kappa, ctx.upto_lap
    after = ctx.laps_left - 2
    if after <= 0:
        return None
    steps = np.arange(1, after + 1)
    stay = a.cur.level0 + a.cur.slope.slope * (a.age + 2 + steps) - kappa * (n + 1 + steps)
    new = a.level0_new + a.slope_new * (1 + steps) - kappa * (n + 1 + steps)
    deficit = loss + _new_times(a, n, 2, 1, kappa)[0] - _old_times(a, n, 2, kappa)[1]
    reached = np.flatnonzero(np.cumsum(stay - new) >= deficit)
    return int(reached[0]) + 3 if reached.size else None


# ---------------------------------------------------------------- a avaliação de um par


def _hard_rules(ctx: LapContext, a: CarState, r: CarState, sources: Sequence[str]) -> list[str]:
    reasons = []
    if r.stint is not None and a.stint is not None and r.stint > a.stint:
        reasons.append("o rival já parou")
    elif r.stint is not None and a.stint is not None and a.stint > r.stint:
        reasons.append("o atacante já parou")
    if r.pit_in or r.pit_out or a.pit_in or a.pit_out:
        reasons.append("um dos carros está nos boxes")
    recent = [
        lap.track_status
        for lap in ctx.rows
        if lap.driver in (a.driver, r.driver) and lap.lap_number >= ctx.upto_lap - 1
    ]
    now = [s.status for s in ctx.states.values()]
    mostly_not_green = bool(now) and sum(s != GREEN for s in now) / len(now) >= NOT_GREEN_SHARE
    if any(s != GREEN for s in recent) or mostly_not_green:
        reasons.append("pista não verde nas voltas recentes")
    if ALL_COMPOUNDS in sources:
        reasons.append(f'previsão com origem "{ALL_COMPOUNDS}"')
    return reasons


def assess_undercut(
    ctx: LapContext,
    attacker: str,
    rival: str | None = None,
    *,
    new_compound: str | None = None,
    rival_new_compound: str | None = None,
    horizon_k: int | None = None,
) -> UndercutAssessment:
    """A margem esperada, a probabilidade e se se pode alertar, para `attacker` parar agora (na
    volta N+1). Sem `rival`, é o carro à frente dele; sem compostos, os mais prováveis."""
    cfg = ctx.config
    a_state = ctx.states.get(attacker)
    rival = rival or car_ahead(ctx.states, attacker)
    r_state = ctx.states.get(rival) if rival else None
    lap = ctx.upto_lap

    def blocked(*why: str) -> UndercutAssessment:
        return UndercutAssessment(
            attacker, rival, lap, None, None, None, None, None, None, None, None, None, None,
            reasons=why,
        )  # fmt: skip

    if a_state is None or r_state is None:
        return blocked("sem rival à frente")
    if ctx.pit_loss is None:
        return blocked("sem perda no pit (nem paradas nem prior)")
    old_a, old_r = a_state.compound, r_state.compound
    if old_a is None or old_r is None:
        return blocked("sem composto atual")

    new_a = new_compound or likely_next_compound(ctx.rows, lap, old_a)
    new_r = rival_new_compound or likely_next_compound(ctx.rows, lap, old_r)
    if horizon_k is None:
        length, _ = expected_stint_length(ctx.rows, lap, old_r, ctx.total_laps)
        horizon_k = round(length - (r_state.age or 0))
    k = int(min(max(horizon_k, cfg.k_min), cfg.k_max, max(ctx.laps_left - 2, 1)))

    base, why = _project(ctx, ctx.pace, a_state, r_state, new_a, new_r, k)
    if base is None:
        return blocked(why)
    up, _ = _project(ctx, ctx.pace_up, a_state, r_state, new_a, new_r, k)
    down, _ = _project(ctx, ctx.pace_down, a_state, r_state, new_a, new_r, k)
    kappa_effect = abs(up.margin - down.margin) / 2 if up and down else 0.0

    components = _margin_sd(base.a, base.r, k, ctx.pit_loss) | {"erro do k": float(kappa_effect)}
    sd_total = float(np.sqrt(sum(v**2 for v in components.values()))) * cfg.sd_inflation
    probability = normal_cdf(base.margin / sd_total) if sd_total > 0 else None

    reasons = _hard_rules(ctx, a_state, r_state, (*base.a.sources, *base.r.sources))
    recover = _laps_to_recover(ctx, base.a, ctx.pit_loss.median)
    if recover is None or ctx.laps_left < max(recover, cfg.min_laps_left, k + 2):
        reasons.append("poucas voltas para recuperar a perda")

    return UndercutAssessment(
        attacker=attacker,
        rival=rival,
        lap=lap,
        margin=base.margin,
        margin_free=base.margin_free,
        margin_sd=sd_total,
        probability=probability,
        horizon_k=k,
        pit_loss=ctx.pit_loss.median,
        rejoin_gap_s=base.rejoin_gap_s,
        rejoin_behind=base.rejoin_behind,
        blocked_by=base.blocked_by,
        laps_to_recover=recover,
        components=components,
        reasons=tuple(reasons),
        compounds=(old_a, new_a, old_r, new_r),
        kappa=ctx.pace.kappa,
        kappa_sd=ctx.kappa_sd,
    )


def assess_all(ctx: LapContext) -> list[UndercutAssessment]:
    """Cada piloto contra o carro à frente dele (os pares que o consumidor avalia a cada volta)."""
    return [assess_undercut(ctx, driver) for driver in sorted(ctx.states)]


# ---------------------------------------------------------------- histerese


Pair = tuple[str, str | None]


class AlertTracker:
    """O estado do alerta de cada par (atacante, rival), com histerese.

    Liga quando a probabilidade chega a `threshold` e só desliga abaixo de `release`; uma regra dura
    (o rival parou, SC...) desliga na hora e apaga o estado, para um alerta velho não reaparecer.
    O par é a chave: se o atacante passa a andar atrás de OUTRO carro, é uma situação nova."""

    def __init__(self, config: UndercutConfig | None = None) -> None:
        self.config = config or UndercutConfig()
        self._on: set[Pair] = set()

    def update(self, assessment: UndercutAssessment) -> bool:
        key = (assessment.attacker, assessment.rival)
        probability = assessment.probability
        if not assessment.eligible:
            self._on.discard(key)
            return False
        assert probability is not None
        if key in self._on:
            if probability >= self.config.release:
                return True
            self._on.discard(key)
            return False
        if probability >= self.config.threshold:
            self._on.add(key)
            return True
        return False

    def update_all(self, assessments: Sequence[UndercutAssessment]) -> set[Pair]:
        """Uma volta inteira: devolve os pares em alerta e esquece os que não foram avaliados (o
        piloto abandonou, ou passou a andar atrás de outro)."""
        seen = {(a.attacker, a.rival) for a in assessments}
        self._on &= seen
        for assessment in assessments:
            self.update(assessment)
        return set(self._on)

    def active(self) -> set[Pair]:
        return set(self._on)
