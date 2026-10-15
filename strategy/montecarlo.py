"""Monte Carlo da corrida: probabilidade de vitória e de pódio de cada piloto, na volta N.

`python -m strategy.montecarlo` não existe: quem chama é o consumidor de estratégia (processo
separado, nunca o event loop da API). A conta, em uma frase: 5.000 corridas do que falta (da volta N
até a última), em arrays pilotos x simulações, em que cada simulação sorteia os parâmetros incertos,
o ruído de cada volta, se haverá Safety Car, quem abandona e quando cada carro para, e o vencedor é
quem cruza a linha primeiro.

O QUE UMA SIMULAÇÃO SORTEIA (cada coisa num stream próprio, ver `make_streams`):
* parâmetros: o nível de ritmo e a inclinação de cada piloto (com o erro que o modelo declara), a
  diferença entre compostos (parte compartilhada entre pilotos + a de cada um), a perda no pit;
* ruído de volta (`sigma_lap` de cada piloto), volta a volta;
* Safety Car: a taxa (`lambda`, da posteriori Gamma: a incerteza da taxa entra na incerteza da
  vitória), quantos SCs faltam, quando começam, quanto duram, a razão de ritmo, o espaçamento e o
  custo da parada sob SC (todos de `strategy.history`, sem a corrida avaliada);
* abandono: a probabilidade (Beta) e a volta de cada piloto;
* o composto seguinte de cada parada (da contagem histórica, respeitando a regra de dois compostos).

AS REGRAS, e o que é heurística (declarado no README):
* PARADAS DOS OUTROS CARROS. O carro para quando o ganho de trocar agora, em tempo, até o fim da
  corrida, supera a perda no pit (mais uma folga de decisão sorteada, `pit_decision_sd`): ganho =
  soma de (pneu atual - pneu novo) nas voltas que faltam - perda. Se ainda não usou dois compostos
  de seco, para no mais tardar `force_laps_left` voltas antes do fim. HEURÍSTICA: equipes decidem
  por rivais, tráfego e pelo SC, que isto só vê pelo ganho de tempo.
* TRÁFEGO. A mesma regra do undercut (`UndercutConfig`): um carro não termina a volta à frente do
  que vinha à frente dele a menos que sua vantagem de ritmo na volta chegue a `pass_advantage_s`;
  senão fica a `follow_gap_s` atrás. HEURÍSTICA, não medida; só o carro imediatamente à frente.
* SAFETY CAR (só o SC; VSC e vermelha ficam de fora): a ordem se congela, o líder anda a `rho` vezes
  o ritmo verde, os intervalos entre carros do pelotão se comprimem a `gap` (os de mais de meia
  volta, retardatários, ficam), a parada custa `cost_ratio` da verde, e o pneu desgasta `sc_wear`
  da volta verde. `rho`, `gap` e `cost_ratio` são MEDIDOS; `sc_wear` é HEURÍSTICA.
* ABANDONO: um carro que abandona fica de fora da pista (não bloqueia ninguém); a classificação é
por
  voltas completadas e depois pelo relógio, como a ordem na pista. Abandono não causa SC aqui
  (independentes), o que é uma simplificação.

REPRODUTIBILIDADE E COMPARAÇÃO. Os números aleatórios vêm de `SeedSequence([ano, etapa, volta])` e
de
um stream independente por componente. Cada stream consome SEMPRE a mesma quantidade de números,
qualquer que seja o modelo (o `k` físico, o estimado ou o encolhimento mudam valores, não a
quantidade sorteada; o multiplicador da taxa de SC age DEPOIS do sorteio, e a contagem de SCs sai da
inversão da CDF de Poisson, não de `rng.poisson`, que gasta números de forma variável). Assim as
variantes usam os MESMOS números aleatórios, e a diferença entre elas é sinal, não ruído de
simulação.

ERRO DE MONTE CARLO. Cada probabilidade vem com `sqrt(p(1-p)/n)`.

SEM VAZAMENTO. Tudo sai de `prepare`, que corta as voltas depois de N como primeiro passo.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from strategy.fuel import FuelVariant, laps_until
from strategy.history import SimPriors
from strategy.laps import DRY_COMPOUNDS, LapRow
from strategy.pace import DEFAULT_SIGMA, PaceModel, fit_pace
from strategy.pitloss import PitLoss, estimate_pit_loss
from strategy.priors import DEFAULT_PRIORS, Priors
from strategy.undercut import CarState, UndercutConfig, car_states, flat_pace

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
STREAMS = ("params", "pit", "compound", "sc", "retirement", "lap_noise")
SC_CODE = "4"
BIG = 1e6  # s: separa quem completou mais voltas na classificação
PODIUM_SIZE = 3
DEFAULT_SLOPE = 0.05  # s/volta de idade, só para quem não tem modelo de composto nenhum
MIN_FLAT_SE = 0.1


@dataclass(frozen=True, slots=True)
class SimConfig:
    n_sims: int = 5000
    sc_rate_multiplier: float = 1.0  # análise de sensibilidade: a taxa de SC em 0,5x, 1x, 2x
    retire_rate_multiplier: float = 1.0  # idem para a probabilidade de abandono
    sc_wear: float = 0.3  # HEURÍSTICA: fração do desgaste de uma volta verde numa volta sob SC
    pit_decision_sd: float = 3.0  # HEURÍSTICA: s de folga aleatória na decisão de parar
    max_stops: int = 3  # paradas por carro, a partir da volta N
    force_laps_left: int = 3  # quem ainda não usou dois compostos para até esta volta do fim
    min_stint_laps: int = 5  # sem parada voluntária com o pneu mais novo que isto
    max_sc: int = 3  # SCs sorteados no máximo, além de um já em curso
    pseudo_events: float = 4.0  # peso da distribuição uniforme na do início de SC e de abandono
    start_jitter_laps: int = 3  # espalha o início sorteado dos eventos históricos
    traffic: UndercutConfig = field(default_factory=UndercutConfig)  # as regras do undercut


class SimulationCancelled(Exception):
    """A simulação foi cancelada (chegou a volta seguinte) e o resultado é descartado."""


# ---------------------------------------------------------------- os carros


@dataclass(frozen=True, slots=True)
class DriverSpec:
    """Um piloto como o simulador o vê na volta N: onde está e o que o modelo diz dele."""

    driver: str
    clock_s: float
    compound: int  # índice em COMPOUNDS
    age: int  # idade do pneu na volta N
    used: tuple[bool, bool, bool]  # compostos de seco já usados
    pit_now: bool  # entrou nos boxes na volta N: troca de pneu no começo da N+1
    inlap_excess: float  # o quanto a volta de entrada passou do ritmo: já está no relógio
    levels: tuple[
        float, float, float
    ]  # nível (idade zero) por composto; nan = sem diferença medida
    slopes: tuple[float, float, float]
    level_se: float  # o erro do nível no ritmo atual
    mean_age: float
    slope_sd: float  # incerteza da inclinação do pneu atual
    new_slope_sd: tuple[float, float, float]
    offset_shared_sd: tuple[float, float, float]  # da diferença entre compostos, igual para todos
    offset_tau: tuple[float, float, float]  # e a de cada piloto
    sigma: float  # ruído de uma volta
    source: str  # "modelo", "troca recente" ou "plano"


@dataclass(frozen=True, slots=True)
class SimInput:
    """Tudo o que a simulação da volta N precisa, tirado SÓ das voltas até N."""

    year: int
    round: int
    upto_lap: int
    total_laps: int
    drivers: tuple[DriverSpec, ...]
    kappa: float
    pit_loss: PitLoss
    sim_priors: SimPriors
    sc_elapsed: int  # voltas que o SC em curso já durou (0 = pista livre na volta N)
    without_model: tuple[str, ...]  # pilotos na pista sem nenhum ritmo: ficam de fora
    variant: FuelVariant

    @property
    def laps_left(self) -> int:
        return self.total_laps - self.upto_lap


def _dry_index(compound: str | None) -> int | None:
    return COMPOUNDS.index(compound) if compound in DRY_COMPOUNDS else None  # type: ignore[arg-type]


def _used_compounds(rows: Sequence[LapRow]) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for r in rows:
        if r.compound in DRY_COMPOUNDS:
            used.setdefault(r.driver, set()).add(r.compound)  # type: ignore[arg-type]
    return used


def _sc_elapsed(rows: Sequence[LapRow], upto_lap: int) -> int:
    """Há quantas voltas o SC está em curso na volta N (metade dos carros com o código "4")."""
    by_lap: dict[int, list[bool]] = {}
    for r in rows:
        by_lap.setdefault(r.lap_number, []).append(SC_CODE in (r.track_status or ""))
    elapsed = 0
    lap = upto_lap
    while lap in by_lap and sum(by_lap[lap]) >= 0.5 * len(by_lap[lap]):
        elapsed += 1
        lap -= 1
    return elapsed


def _spec(
    driver: str,
    state: CarState,
    pace: PaceModel,
    flat: dict[str, float],
    used: set[str],
    upto_lap: int,
) -> DriverSpec | None:
    cidx = _dry_index(state.compound)
    if cidx is None or state.age is None:
        return None
    model = pace.degradation
    age = max(state.age, 1)
    cur = pace.current(driver)
    own_slope = model.slope(driver, COMPOUNDS[cidx])
    sigma = (
        (cur.slope.sigma_lap if cur else None) or (own_slope.sigma_lap if own_slope else None)
        or model.sigma_lap or DEFAULT_SIGMA
    )  # fmt: skip

    if cur is not None and cur.stint == state.stint:
        level, slope, level_se = cur.level0, cur.slope.slope, cur.se_ref
        mean_age, slope_sd, source = cur.mean_age, cur.slope.predictive_sd, "modelo"
    elif cur is not None and own_slope is not None:
        # parou há pouco: o stint novo ainda não tem nível. O nível dele é o do anterior mais a
        # diferença entre os compostos.
        offset = pace.offset(cur.compound, COMPOUNDS[cidx])
        if offset is None:
            return _flat_spec(driver, state, pace, flat, used, upto_lap, cidx, age, sigma)
        level, slope, level_se = cur.level0 + offset.delta, own_slope.slope, cur.se_ref
        mean_age, slope_sd, source = cur.mean_age, own_slope.predictive_sd, "troca recente"
    else:
        return _flat_spec(driver, state, pace, flat, used, upto_lap, cidx, age, sigma)
    return _finish_spec(
        driver,
        state,
        pace,
        used,
        cidx,
        age,
        level,
        slope,
        level_se,
        mean_age,
        slope_sd,
        sigma,
        source,
    )


def _flat_spec(
    driver: str,
    state: CarState,
    pace: PaceModel,
    flat: dict[str, float],
    used: set[str],
    upto_lap: int,
    cidx: int,
    age: int,
    sigma: float,
) -> DriverSpec | None:
    """Sem nível de ritmo: parte do tempo recente do próprio piloto, com a inclinação do
    composto."""
    if driver not in flat:
        return None
    est = pace.degradation.slope(driver, COMPOUNDS[cidx])
    slope = est.slope if est is not None else DEFAULT_SLOPE
    slope_sd = est.predictive_sd if est is not None else 0.03
    level = flat[driver] + pace.kappa * (upto_lap - 1) - slope * age
    level_se = max(sigma / math.sqrt(3), MIN_FLAT_SE)  # três voltas de referência
    return _finish_spec(
        driver,
        state,
        pace,
        used,
        cidx,
        age,
        level,
        slope,
        level_se,
        float(age),
        slope_sd,
        sigma,
        "plano",
    )


def _finish_spec(
    driver, state, pace, used, cidx, age, level, slope, level_se, mean_age, slope_sd, sigma, source
) -> DriverSpec:
    levels = [math.nan] * 3
    slopes = [math.nan] * 3
    new_sd = [0.0] * 3
    shared = [0.0] * 3
    tau = [0.0] * 3
    for k, name in enumerate(COMPOUNDS):
        if k == cidx:
            levels[k], slopes[k] = level, slope
            continue
        offset = pace.offset(COMPOUNDS[cidx], name)
        est = pace.degradation.slope(driver, name)
        if offset is None or est is None:
            continue
        levels[k] = level + offset.delta
        slopes[k], new_sd[k] = est.slope, est.predictive_sd
        shared[k] = math.hypot(offset.se, offset.race_sd)
        tau[k] = offset.tau
    return DriverSpec(
        driver=driver,
        clock_s=state.clock_s,
        compound=cidx,
        age=age,
        used=tuple(name in used for name in COMPOUNDS),  # type: ignore[arg-type]
        pit_now=state.pit_in,
        inlap_excess=0.0,
        levels=tuple(levels),  # type: ignore[arg-type]
        slopes=tuple(slopes),  # type: ignore[arg-type]
        level_se=float(level_se),
        mean_age=float(mean_age),
        slope_sd=float(slope_sd),
        new_slope_sd=tuple(new_sd),  # type: ignore[arg-type]
        offset_shared_sd=tuple(shared),  # type: ignore[arg-type]
        offset_tau=tuple(tau),  # type: ignore[arg-type]
        sigma=float(sigma),
        source=source,
    )


def prepare(
    rows: Sequence[LapRow],
    *,
    year: int,
    round: int,
    upto_lap: int,
    total_laps: int,
    sim_priors: SimPriors,
    priors: Priors = DEFAULT_PRIORS,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
) -> SimInput:
    """O estado da corrida na volta N e o modelo de cada piloto, só com o que se sabia nela."""
    visible = laps_until(rows, upto_lap)  # PRIMEIRO passo: nada do futuro passa daqui
    pace = fit_pace(
        visible, upto_lap=upto_lap, total_laps=total_laps, variant=variant, priors=priors
    )
    states = car_states(visible, upto_lap)
    flat = flat_pace(visible, upto_lap)
    used = _used_compounds(visible)
    loss = estimate_pit_loss(visible, upto_lap=upto_lap, kappa=pace.kappa, prior=priors.pit_loss)
    if loss is None:
        loss = PitLoss(priors.pit_loss.mean, priors.pit_loss.stop_sd, priors.pit_loss.between_sd,
                       None, 0, 0, None, 0.0, "prior")  # fmt: skip
    lap_times = {
        r.driver: r.lap_time_ms / 1000
        for r in visible
        if r.lap_number == upto_lap and r.lap_time_ms is not None
    }
    specs: list[DriverSpec] = []
    missing: list[str] = []
    for driver in sorted(states):
        spec = _spec(driver, states[driver], pace, flat, used.get(driver, set()), upto_lap)
        if spec is None:
            missing.append(driver)
            continue
        if spec.pit_now and driver in lap_times:
            # a volta de entrada já está no relógio com a parte da perda que cabe a ela (o excesso
            # sobre o ritmo): na volta de saída só falta o resto, senão a perda conta duas vezes
            normal = (
                spec.levels[spec.compound]
                + spec.slopes[spec.compound] * spec.age
                - pace.kappa * (upto_lap - 1)
            )
            spec = replace(spec, inlap_excess=max(lap_times[driver] - normal, 0.0))
        specs.append(spec)
    return SimInput(
        year=year,
        round=round,
        upto_lap=upto_lap,
        total_laps=total_laps,
        drivers=tuple(specs),
        kappa=pace.kappa,
        pit_loss=loss,
        sim_priors=sim_priors,
        sc_elapsed=_sc_elapsed(visible, upto_lap),
        without_model=tuple(missing),
        variant=variant,
    )


# ---------------------------------------------------------------- os números aleatórios


def make_streams(year: int, round: int, lap: int) -> dict[str, np.random.Generator]:
    """Um stream independente por componente, todos derivados de (corrida, volta). A mesma tripla dá
    os mesmos números, e dois componentes nunca compartilham números."""
    children = np.random.SeedSequence([year, round, lap]).spawn(len(STREAMS))
    return {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(STREAMS, children, strict=True)
    }


def poisson_counts(u: np.ndarray, rate: np.ndarray, cap: int) -> np.ndarray:
    """Quantos eventos de um Poisson(`rate`), pela inversão da CDF a partir de uma uniforme `u`.
    Gasta exatamente um número por simulação, qualquer que seja a taxa (rng.poisson não gasta), e é
    o que deixa o multiplicador de taxa da sensibilidade usar os mesmos números."""
    pmf = np.exp(-rate)
    cdf = pmf.copy()
    count = np.zeros(rate.shape, dtype=int)
    for k in range(cap):
        count += u > cdf
        pmf = pmf * rate / (k + 1)
        cdf = cdf + pmf
    return count


def beyond_mass(fractions: Sequence[float], total_laps: int, upto_lap: int, weight: float) -> float:
    """A fração dos eventos que ainda faltam depois da volta N, `1 - F(N)`, da mistura: o histórico
    (com `weight`) e o uniforme (com `1 - weight`)."""
    if total_laps <= upto_lap:
        return 0.0
    uniform = (total_laps - upto_lap) / total_laps
    if not fractions:
        return uniform
    empirical = float(np.mean([f * total_laps > upto_lap for f in fractions]))
    return weight * empirical + (1.0 - weight) * uniform


def mixture_weight(n_events: int, pseudo: float) -> float:
    return n_events / (n_events + pseudo) if n_events + pseudo > 0 else 0.0


def sample_starts(
    u_comp: np.ndarray,
    u_pos: np.ndarray,
    u_jit: np.ndarray,
    fractions: Sequence[float],
    *,
    total_laps: int,
    upto_lap: int,
    weight: float,
    jitter: int,
) -> np.ndarray:
    """A volta de início de cada evento, condicionada a ser depois de N, da mistura do histórico
    (com
    o início espalhado em +- `jitter` voltas) e do uniforme. Cada evento gasta três uniformes."""
    first = upto_lap + 1
    span = max(total_laps - upto_lap, 1)
    uniform = first + np.minimum((u_pos * span).astype(int), span - 1)
    eligible = np.array(sorted(f * total_laps for f in fractions if f * total_laps > upto_lap))
    if eligible.size == 0:
        return uniform
    m_emp = weight * eligible.size / len(fractions)
    m_uni = (1.0 - weight) * span / total_laps
    pick_emp = u_comp < m_emp / (m_emp + m_uni)
    chosen = eligible[np.minimum((u_pos * eligible.size).astype(int), eligible.size - 1)]
    start = np.rint(chosen + (u_jit - 0.5) * 2 * jitter).astype(int)
    start = np.clip(start, first, total_laps)
    return np.where(pick_emp, start, uniform)


@dataclass(slots=True)
class Draws:
    """Todos os números aleatórios de UMA volta de decisão, gerados de uma vez em ordem fixa."""

    streams: dict[str, np.random.Generator]
    level_err: np.ndarray  # (S, n)
    slope_err: np.ndarray
    offset_shared: np.ndarray  # (S, 3, 3)
    offset_tau_z: np.ndarray  # (S, n, 3)
    new_slope_z: np.ndarray  # (S, n, 3)
    loss_shared_z: np.ndarray  # (S,)
    tie: np.ndarray  # (S, n) desempate de relógios iguais
    loss_stop_z: np.ndarray  # (S, n, max_stops + 1)
    pit_eps: np.ndarray  # (S, n)
    u_next: np.ndarray  # (S, n, max_stops + 1)
    sc_rate_gamma: np.ndarray  # (S,)
    sc_u_count: np.ndarray  # (S,)
    sc_u_comp: np.ndarray  # (S, max_sc)
    sc_u_pos: np.ndarray
    sc_u_jit: np.ndarray
    sc_u_dur: np.ndarray
    sc_u_active: np.ndarray  # (S,) a duração restante de um SC em curso
    sc_u_ratio: np.ndarray  # (S,)
    sc_u_gap: np.ndarray
    sc_z_cost: np.ndarray
    ret_rate_beta: np.ndarray  # (S,)
    ret_u: np.ndarray  # (S, n)
    ret_u_comp: np.ndarray
    ret_u_pos: np.ndarray
    ret_u_jit: np.ndarray


def draw_all(inp: SimInput, config: SimConfig) -> Draws:
    """Gera os sorteios. A ordem e as formas dependem só de (S, n, nº de SCs, nº de paradas): nunca
    de
    um valor do modelo, que é o que mantém as variantes alinhadas."""
    s, n = config.n_sims, len(inp.drivers)
    streams = make_streams(inp.year, inp.round, inp.upto_lap)
    prm, pit, cmp_, sc, ret = (
        streams[k] for k in ("params", "pit", "compound", "sc", "retirement")
    )
    stops = config.max_stops + 2  # +1: quem já entrou nos boxes na volta N conta como uma parada
    sp, rp = inp.sim_priors.sc, inp.sim_priors.retirement
    return Draws(
        streams=streams,
        level_err=prm.standard_normal((s, n)),
        slope_err=prm.standard_normal((s, n)),
        offset_shared=prm.standard_normal((s, 3, 3)),
        offset_tau_z=prm.standard_normal((s, n, 3)),
        new_slope_z=prm.standard_normal((s, n, 3)),
        loss_shared_z=prm.standard_normal(s),
        tie=prm.random((s, n)),
        loss_stop_z=pit.standard_normal((s, n, stops)),
        pit_eps=pit.standard_normal((s, n)),
        u_next=cmp_.random((s, n, stops)),
        sc_rate_gamma=sc.gamma(sp.events + 0.5, 1.0 / max(sp.races, 1), size=s),
        sc_u_count=sc.random(s),
        sc_u_comp=sc.random((s, config.max_sc)),
        sc_u_pos=sc.random((s, config.max_sc)),
        sc_u_jit=sc.random((s, config.max_sc)),
        sc_u_dur=sc.random((s, config.max_sc)),
        sc_u_active=sc.random(s),
        sc_u_ratio=sc.random(s),
        sc_u_gap=sc.random(s),
        sc_z_cost=sc.standard_normal(s),
        ret_rate_beta=ret.beta(
            rp.retirements + 0.5, max(rp.starters - rp.retirements, 0) + 0.5, size=s
        ),
        ret_u=ret.random((s, n)),
        ret_u_comp=ret.random((s, n)),
        ret_u_pos=ret.random((s, n)),
        ret_u_jit=ret.random((s, n)),
    )


# ---------------------------------------------------------------- o resultado


@dataclass(frozen=True, slots=True)
class DriverForecast:
    driver: str
    win: float
    podium: float
    win_error: float  # erro de Monte Carlo, sqrt(p(1-p)/n)
    podium_error: float
    expected_stops: float  # paradas médias da volta N até o fim
    retire: float  # probabilidade de abandonar até o fim
    source: str


@dataclass(frozen=True, slots=True)
class RaceForecast:
    year: int
    round: int
    lap: int
    total_laps: int
    n_sims: int
    drivers: tuple[DriverForecast, ...]
    sc_probability: float  # P(há volta sob SC de N+1 até o fim)
    sc_multiplier: float
    duration_s: float
    without_model: tuple[str, ...]
    variant: str
    notes: tuple[str, ...] = ()

    def win_probabilities(self) -> dict[str, float]:
        return {d.driver: d.win for d in self.drivers}


def mc_error(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


# ---------------------------------------------------------------- a simulação


def sc_schedule(
    count: np.ndarray,
    starts: np.ndarray,
    durations: np.ndarray,
    *,
    upto_lap: int,
    laps: int,
    remaining: np.ndarray,
) -> np.ndarray:
    """Quais das `laps` voltas que faltam (a partir da N+1) rodam sob SC, por simulação.

    `count[s]` SCs novos, o e-ésimo começando na volta `starts[s, e]` (da N+1 em diante) e durando
    `durations[s, e]` voltas; mais o SC que já estava em curso na volta N, que ainda dura
    `remaining[s]` voltas (0 = nenhum)."""
    lap_index = np.arange(laps)
    mask = np.zeros((count.shape[0], laps), dtype=bool)
    for e in range(starts.shape[1]):
        begin = starts[:, e] - upto_lap - 1  # índice da primeira volta sob SC (0 = a N+1)
        mask |= (
            (count > e)[:, None]
            & (lap_index >= begin[:, None])
            & (lap_index < (begin + durations[:, e])[:, None])
        )
    return mask | (lap_index[None, :] < remaining[:, None])


def remaining_sc_laps(u: np.ndarray, durations: np.ndarray, elapsed: int) -> np.ndarray:
    """Quantas voltas ainda faltam a um SC que já durou `elapsed`: a duração histórica, dado que
    passou de `elapsed`, menos `elapsed`. Se nenhuma histórica foi tão longa, mais uma volta."""
    if elapsed <= 0:
        return np.zeros(u.shape, dtype=int)
    left = durations[durations > elapsed] - elapsed
    pool = left if left.size else np.array([1])
    return pool[np.minimum((u * pool.size).astype(int), pool.size - 1)]


def retirement_laps(
    u: np.ndarray, p_after: np.ndarray, starts: np.ndarray, total_laps: int
) -> np.ndarray:
    """A ÚLTIMA volta que cada piloto completa: quem abandona na volta `start` completou a anterior;
    `total_laps + 1` quer dizer que não abandona."""
    return np.where(u < p_after, starts - 1, total_laps + 1)


def hold_in_traffic(
    tentative: np.ndarray, lap_time: np.ndarray, traffic: UndercutConfig
) -> np.ndarray:
    """O relógio ao fim da volta com o tráfego, carros em ordem de largada na volta (a frente
    primeiro). A regra é a do undercut: um carro não termina à frente do que vinha na frente dele
    se a vantagem de ritmo na volta for menor que `pass_advantage_s`; fica a `follow_gap_s`."""
    held = tentative.copy()
    for rank in range(1, held.shape[1]):
        floor = held[:, rank - 1] + traffic.follow_gap_s
        advantage = lap_time[:, rank - 1] - lap_time[:, rank]
        hold = (
            (tentative[:, rank] < floor)
            & (advantage < traffic.pass_advantage_s)
            & np.isfinite(tentative[:, rank])
        )
        held[:, rank] = np.where(hold, floor, tentative[:, rank])
    return held


def compress_under_sc(
    prev: np.ndarray,
    leader_base_lap: np.ndarray,
    rho: np.ndarray,
    gap_s: np.ndarray,
    penalty: np.ndarray,
) -> np.ndarray:
    """O relógio ao fim de uma volta sob SC, carros em ordem. A ordem congela; o líder anda a `rho`
    vezes a volta verde dele; cada um dos de trás fica a no máximo `gap_s` do da frente, salvo os
    que estão a mais de meia volta do da frente (retardatários), que mantêm a distância. A troca de
    pneu (`penalty`) custa depois da compressão."""
    leader_lap = leader_base_lap * rho
    out = np.empty_like(prev)
    out[:, 0] = prev[:, 0] + leader_lap
    for rank in range(1, prev.shape[1]):
        with np.errstate(invalid="ignore"):  # inf - inf entre dois que abandonaram
            behind = prev[:, rank] - prev[:, rank - 1]
        spacing = np.where(behind < leader_lap / 2, np.minimum(behind, gap_s), behind)
        out[:, rank] = np.where(np.isfinite(prev[:, rank]), out[:, rank - 1] + spacing, np.inf)
    return np.where(np.isfinite(out), out + penalty, np.inf)


def _sorted(order: np.ndarray, array: np.ndarray) -> np.ndarray:
    """array[s, order[s, i]]: a linha de cada simulação na ordem dada."""
    return np.take_along_axis(array, order, axis=1)


def _take(array: np.ndarray, index: np.ndarray) -> np.ndarray:
    """array[s, n, index[s, n]]"""
    return np.take_along_axis(array, index[..., None], axis=2)[..., 0]


def _next_compound(
    counts: np.ndarray,
    current: np.ndarray,
    used: np.ndarray,
    available: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """O composto da próxima parada: a contagem histórica a partir do composto atual, só entre os
    disponíveis; antes de usar dois compostos, só os ainda não usados (a regra de dois
    compostos)."""
    two = used.sum(axis=2) >= 2
    mask = available[None, :, :] & (two[..., None] | (np.arange(3) != current[..., None]))
    weight = counts[current] * mask
    empty = weight.sum(axis=2) <= 0
    weight = np.where(empty[..., None], mask.astype(float), weight)
    empty = weight.sum(axis=2) <= 0
    weight = np.where(empty[..., None], np.eye(3)[current], weight)
    cdf = np.cumsum(weight, axis=2) / weight.sum(axis=2, keepdims=True)
    return np.minimum((u[..., None] > cdf).sum(axis=2), 2)


def simulate(
    inp: SimInput,
    config: SimConfig | None = None,
    *,
    draws: Draws | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RaceForecast:
    """Corre o que falta da corrida `n_sims` vezes e devolve as probabilidades."""
    config = config or SimConfig()
    started = time.perf_counter()
    n = len(inp.drivers)
    s = config.n_sims
    laps = inp.laps_left
    if n == 0:
        raise ValueError("nenhum piloto com ritmo para simular")
    draws = draws or draw_all(inp, config)
    stops_cap = config.max_stops
    traffic = config.traffic
    sp, rp, comp = inp.sim_priors.sc, inp.sim_priors.retirement, inp.sim_priors.compounds

    # ---- parâmetros por simulação: nível e inclinação de cada composto, por piloto
    spec = inp.drivers
    cur0 = np.array([d.compound for d in spec])
    level = np.array([d.levels for d in spec])  # (n, 3)
    slope = np.array([d.slopes for d in spec])
    available = ~np.isnan(level)
    level_se = np.array([d.level_se for d in spec])
    mean_age = np.array([d.mean_age for d in spec])
    slope_sd = np.array([d.slope_sd for d in spec])
    new_sd = np.array([d.new_slope_sd for d in spec])
    shared_sd = np.array([d.offset_shared_sd for d in spec])
    tau = np.array([d.offset_tau for d in spec])
    sigma = np.array([d.sigma for d in spec])
    level0_err = draws.level_err * level_se - mean_age * draws.slope_err * slope_sd  # (S, n)
    cur_slope_err = draws.slope_err * slope_sd
    shared = draws.offset_shared[:, cur0, :] * shared_sd[None]  # (S, n, 3): do composto atual a k
    eye = np.eye(3, dtype=bool)[cur0]  # (n, 3): o próprio composto atual
    lv = np.where(available[None], level[None] + level0_err[..., None], np.inf)
    lv = np.where(eye[None], lv, lv + shared + draws.offset_tau_z * tau[None])
    sl = np.where(available[None], slope[None], 0.0)
    sl = np.where(eye[None], sl + cur_slope_err[..., None], sl + draws.new_slope_z * new_sd[None])

    # ---- a perda no pit e o que o SC faz
    loss = inp.pit_loss
    loss_shared = loss.median + draws.loss_shared_z * loss.se  # (S,)
    loss_draw = np.maximum(loss_shared[:, None, None] + draws.loss_stop_z * loss.sd, 5.0)
    rate = (
        draws.sc_rate_gamma * config.sc_rate_multiplier
    )  # SCs por corrida, uma taxa por simulação
    w_sc = mixture_weight(sp.events, config.pseudo_events)
    remaining_mass = beyond_mass(sp.start_fractions, inp.total_laps, inp.upto_lap, w_sc)
    n_new = poisson_counts(draws.sc_u_count, rate * remaining_mass, config.max_sc)
    starts = sample_starts(
        draws.sc_u_comp, draws.sc_u_pos, draws.sc_u_jit, sp.start_fractions,
        total_laps=inp.total_laps, upto_lap=inp.upto_lap, weight=w_sc,
        jitter=config.start_jitter_laps,
    )  # fmt: skip
    durations = np.array(sp.durations) if sp.durations else np.array([4])
    dur = durations[np.minimum((draws.sc_u_dur * durations.size).astype(int), durations.size - 1)]
    sc_mask = sc_schedule(
        n_new,
        starts,
        dur,
        upto_lap=inp.upto_lap,
        laps=laps,
        remaining=remaining_sc_laps(draws.sc_u_active, durations, inp.sc_elapsed),
    )
    ratios = np.array(sp.lap_ratios)
    gaps = np.array(sp.gaps)
    rho = ratios[np.minimum((draws.sc_u_ratio * ratios.size).astype(int), ratios.size - 1)]
    sc_gap = gaps[np.minimum((draws.sc_u_gap * gaps.size).astype(int), gaps.size - 1)]
    cost_mean, cost_sd = sp.cost_ratio
    cost_ratio = np.clip(cost_mean + draws.sc_z_cost * cost_sd, 0.05, 1.0)

    # ---- abandonos: a volta da última volta completada (L + 1 = termina a corrida)
    w_ret = mixture_weight(rp.retirements, config.pseudo_events)
    mass_ret = beyond_mass(rp.fractions, inp.total_laps, inp.upto_lap, w_ret)
    rate_r = np.minimum(draws.ret_rate_beta[:, None] * config.retire_rate_multiplier, 0.999)
    p_after = rate_r * mass_ret / (1.0 - rate_r * (1.0 - mass_ret))
    retire_start = sample_starts(
        draws.ret_u_comp, draws.ret_u_pos, draws.ret_u_jit, rp.fractions,
        total_laps=inp.total_laps, upto_lap=inp.upto_lap, weight=w_ret,
        jitter=config.start_jitter_laps,
    )  # fmt: skip
    retire_lap = retirement_laps(draws.ret_u, p_after, retire_start, inp.total_laps)

    # ---- o estado que anda de volta em volta
    counts = np.zeros((3, 3)) + 0.5  # suavização: nenhuma troca é impossível
    for (a, b), c in comp.counts.items():
        if a in COMPOUNDS and b in COMPOUNDS:
            counts[COMPOUNDS.index(a), COMPOUNDS.index(b)] += c
    clock = np.broadcast_to(np.array([d.clock_s for d in spec]), (s, n)).copy()
    cur = np.broadcast_to(cur0, (s, n)).copy()
    used = np.broadcast_to(np.array([d.used for d in spec]), (s, n, 3)).copy()
    used[:, np.arange(n), cur0] = True
    age = np.broadcast_to(np.array([float(d.age) for d in spec]), (s, n)).copy()
    cols = np.arange(n)[None, :]
    stops_done = np.zeros((s, n), dtype=int)  # paradas da volta N em diante (indexa os sorteios)
    nxt = _next_compound(counts, cur, used, available, draws.u_next[:, :, 0])
    pending = np.zeros((s, n))  # a perda de uma troca que já começou (entrou nos boxes na volta N)
    pit_now = np.array([d.pit_now for d in spec])
    if pit_now.any():
        idx = np.flatnonzero(pit_now)
        cur[:, idx] = nxt[:, idx]
        age[:, idx] = 0.0
        used[:, idx, :] |= np.eye(3, dtype=bool)[cur[:, idx]]
        stops_done[:, idx] = 1
        excess = np.array([d.inlap_excess for d in spec])[idx]
        pending[:, idx] = np.maximum(loss_draw[:, idx, 0] - excess, 0.0)
        nxt = _next_compound(counts, cur, used, available, draws.u_next[:, :, 1])
    stop_count = stops_done.copy()
    rows_ix = np.arange(s)[:, None]
    noise = draws.streams["lap_noise"]

    for j in range(1, laps + 1):
        lap_no = inp.upto_lap + j
        if should_stop is not None and should_stop():
            raise SimulationCancelled
        z = noise.standard_normal((s, n))  # sempre (S, n): o consumo não depende de quem vive
        alive = retire_lap >= lap_no  # completa esta volta
        under_sc = sc_mask[:, j - 1]
        age_now = age + np.where(under_sc, config.sc_wear, 1.0)[:, None]
        l_cur, s_cur = _take(lv, cur), _take(sl, cur)
        base_lap = l_cur + s_cur * age_now - inp.kappa * (lap_no - 1) + z * sigma

        # ---- a decisão de parar: o ganho de trocar agora até o fim contra a perda (sob SC, mais
        # barata)
        left = inp.total_laps - lap_no  # voltas depois desta
        slot = np.minimum(stops_done, stops_cap + 1)
        eff_loss = loss_draw[rows_ix, cols, slot]
        eff_loss = np.where(under_sc[:, None], eff_loss * cost_ratio[:, None], eff_loss)
        l_nxt, s_nxt = _take(lv, nxt), _take(sl, nxt)
        r = float(left)
        gain = (
            r * (l_cur - l_nxt) + s_cur * (r * age_now + r * (r + 1) / 2) - s_nxt * r * (r + 1) / 2
            - eff_loss
        )  # fmt: skip
        two = used.sum(axis=2) >= 2
        wants = (gain > draws.pit_eps * config.pit_decision_sd) & (age_now > config.min_stint_laps)
        forced = ~two & (left <= config.force_laps_left)
        can = alive & (stops_done < stops_cap) & (left >= 1) & (pending == 0)
        pit = can & (wants | forced)
        penalty = np.where(pit, eff_loss, 0.0) + pending  # a troca custa na volta em que acontece
        pending = np.zeros((s, n))

        # ---- o relógio. Quem abandona nesta volta já fica no fim da fila (relógio infinito).
        prev = np.where(alive, clock, np.inf)
        # relógios iguais ao microssegundo: quem fica na frente é sorteado, não o primeiro da lista
        order = np.argsort(prev + draws.tie * 1e-6, axis=1, kind="stable")
        lap_sorted = _sorted(order, base_lap + penalty)
        tent_sorted = _sorted(order, np.where(alive, prev + base_lap + penalty, np.inf))
        new_sorted = hold_in_traffic(tent_sorted, lap_sorted, traffic)
        if under_sc.any():
            # sob SC: o líder anda a rho vezes o ritmo; a troca de pneu custa depois da compressão
            sc_sorted = compress_under_sc(
                _sorted(order, prev),
                _sorted(order, base_lap)[:, 0],
                rho,
                sc_gap,
                _sorted(order, penalty),
            )
            new_sorted = np.where(under_sc[:, None], sc_sorted, new_sorted)
        new_clock = np.empty_like(clock)
        np.put_along_axis(new_clock, order, new_sorted, axis=1)
        clock = np.where(alive, new_clock, clock)

        # ---- depois da volta: quem parou troca de pneu; o desgaste anda
        age = np.where(pit, 0.0, age_now)
        if pit.any():
            used |= pit[..., None] & np.eye(3, dtype=bool)[nxt]
            cur = np.where(pit, nxt, cur)
            stops_done = stops_done + pit
            stop_count = stop_count + pit
            planned = _next_compound(
                counts,
                cur,
                used,
                available,
                draws.u_next[rows_ix, cols, np.minimum(stops_done, stops_cap + 1)],
            )
            nxt = np.where(pit, planned, nxt)

    # ---- a classificação: voltas completadas, depois o relógio (a ordem na pista)
    done = np.minimum(retire_lap, inp.total_laps)
    key = (inp.total_laps - done) * BIG + clock
    rank = np.argsort(np.argsort(key, axis=1, kind="stable"), axis=1, kind="stable")
    win = (rank == 0).mean(axis=0)
    podium = (rank < PODIUM_SIZE).mean(axis=0)
    retired = (retire_lap <= inp.total_laps).mean(axis=0)
    out = tuple(
        DriverForecast(
            driver=d.driver,
            win=float(win[i]),
            podium=float(podium[i]),
            win_error=mc_error(float(win[i]), s),
            podium_error=mc_error(float(podium[i]), s),
            expected_stops=float(stop_count[:, i].mean()),
            retire=float(retired[i]),
            source=d.source,
        )
        for i, d in enumerate(spec)
    )
    return RaceForecast(
        year=inp.year,
        round=inp.round,
        lap=inp.upto_lap,
        total_laps=inp.total_laps,
        n_sims=s,
        drivers=out,
        sc_probability=float(sc_mask.any(axis=1).mean()),
        sc_multiplier=config.sc_rate_multiplier,
        duration_s=time.perf_counter() - started,
        without_model=inp.without_model,
        variant=inp.variant.value,
    )


def sensitivity(
    inp: SimInput,
    config: SimConfig | None = None,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict[float, RaceForecast]:
    """A mesma simulação com a taxa de SC multiplicada: usa EXATAMENTE os mesmos números aleatórios
    (o multiplicador age depois do sorteio), então a diferença entre as linhas é o efeito da
    taxa."""
    config = config or SimConfig()
    draws = draw_all(inp, config)
    return {
        m: simulate(inp, replace(config, sc_rate_multiplier=m), draws=draws) for m in multipliers
    }
