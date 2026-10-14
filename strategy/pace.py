"""Nível de ritmo: o que falta à degradação para comparar um pneu novo com um gasto.

O modelo de degradação só tem INCLINAÇÕES (quanto o pneu piora por volta). Para dizer "o atacante,
com
pneu novo, anda quanto mais rápido que o rival com o pneu gasto" falta o NÍVEL: em que tempo cada
um anda. Aqui:

    tempo(idade, volta) = nivel0 + inclinacao * idade - k * (volta - 1)

`nivel0` é o tempo CORRIGIDO do combustível na idade zero, um por STINT (cada jogo de pneus tem o
seu
nível: o piloto, a carga, a pista naquele momento). É a mediana de `tempo_corrigido - inclinacao *
idade` nas voltas limpas do stint, e é ENCOLHIDO para a média do composto na corrida (com o peso
`B = t^2 / (t^2 + se^2)` da degradação): um stint de 3 voltas não manda sozinho.

A EVOLUÇÃO DA PISTA E A DIFERENÇA ENTRE OS DOIS CARROS
A pista melhora volta a volta (borracha, temperatura) e isso entra em `k`. Como o atacante e o rival
andam na MESMA pista nas MESMAS voltas, o que importa é a DIFERENÇA entre os dois, e nela a evolução
da pista aparece nos dois lados e se cancela. Por isso o nível de um piloto num composto que ele
nunca
usou NÃO é a média desse composto (a média mistura pilotos mais rápidos e mais lentos): é o nível
dele no composto atual MAIS a diferença entre os dois compostos, medida DENTRO de cada piloto que
rodou os dois (o piloto cancela o próprio ritmo). Sem pilotos suficientes com os dois compostos, cai
para a diferença entre as médias dos compostos, com uma incerteza maior.

INCERTEZA. Um nível tem duas partes: o ruído de amostragem (`se_ref`, do ritmo no meio do stint) e a
incerteza da inclinação, que pesa mais longe da idade média do stint: a variância da SOMA de tempos
em
idades `a` é `(n * se_ref)^2 + (soma(a - idade_media))^2 * sd_inclinacao^2`, a fórmula de predição
da
regressão (usar o nível na idade ZERO somaria um erro de extrapolação que a regressão não tem,
porque o nível e a inclinação estimados andam em sentidos contrários).

SEM VAZAMENTO DO FUTURO: `fit_pace` só usa voltas até `upto_lap`, e a degradação e o `k` também.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from strategy.degradation import (
    DegradationModel,
    SlopeEstimate,
    estimate_tau,
    fit_degradation,
    random_effects_mean,
    shrink_slope,
)
from strategy.fuel import FuelVariant, correct_for_fuel, dry_clean_laps, laps_until
from strategy.laps import LapRow
from strategy.priors import DEFAULT_PRIORS, Priors

MIN_LEVEL_LAPS = 3  # voltas limpas mínimas de um stint para ter nível
MIN_PAIRS = 3  # pilotos com os dois compostos para medir a diferença DENTRO do piloto
DEFAULT_SIGMA = 0.3  # s: ruído de volta de reserva (a mediana das corridas é 0,32)
Z_NONE = float("inf")  # "não consegui estimar": sem encolhimento


@dataclass(frozen=True, slots=True)
class StintLevel:
    driver: str
    compound: str
    stint: int
    level0: float  # s corrigidos na idade zero, DEPOIS do encolhimento
    raw_level0: float  # o que só este stint diz
    se_ref: float  # incerteza do nível no ritmo atual (idade média), com o encolhimento
    mean_age: float
    n_laps: int
    first_lap: int
    last_lap: int
    slope: SlopeEstimate  # a degradação com que o nível foi tirado (e que prevê a partir dele)
    weight: float  # peso do dado do stint (1 = só ele, 0 = só a média do composto)

    def predict(self, age: float, lap: int, kappa: float) -> float:
        """O tempo de volta previsto (s), na idade `age`, na volta `lap` da corrida."""
        return self.level0 + self.slope.slope * age - kappa * (lap - 1)

    def sd_of_sum(self, ages: Sequence[float], *, slope_sd: float | None = None) -> float:
        """Desvio da SOMA dos tempos previstos nessas idades (nível e inclinação; o ruído de volta
        entra à parte). `slope_sd` troca a incerteza da inclinação (o padrão é a de um stint futuro,
        `predictive_sd`)."""
        sd_slope = self.slope.predictive_sd if slope_sd is None else slope_sd
        a = np.asarray(ages, dtype=float)
        level_part = len(a) * self.se_ref
        slope_part = float(np.sum(a - self.mean_age)) * sd_slope
        return float(np.hypot(level_part, slope_part))


@dataclass(frozen=True, slots=True)
class CompoundLevel:
    compound: str
    mean0: float  # nível médio do composto na corrida (idade zero)
    se: float
    n_stints: int


@dataclass(frozen=True, slots=True)
class Offset:
    """O quanto o nível (idade zero) de `to` difere do de `from`, em segundos (+ = `to` mais
    lento)."""

    from_compound: str
    to_compound: str
    delta: float
    se: float  # incerteza da diferença média
    tau: float  # variação da diferença entre pilotos (o próximo stint terá a sua)
    n_pairs: int
    source: Literal["pares", "médias", "igual"]
    # O quanto esta diferença varia ENTRE corridas (de `Priors.offset_sd`): o viés por corrida que o
    # erro-padrão não vê. Medido: o erro-padrão por corrida é de 0,05 a 0,17 s e a diferença entre
    # corridas espalha ~0,4 s (docs/perda-no-pit.md).
    race_sd: float = 0.0

    @property
    def predictive_sd(self) -> float:
        return float(np.sqrt(self.se**2 + self.tau**2 + self.race_sd**2))


@dataclass(frozen=True, slots=True)
class PaceModel:
    upto_lap: int
    degradation: DegradationModel
    levels: dict[tuple[str, int], StintLevel]  # (piloto, stint) -> nível
    compound_levels: dict[str, CompoundLevel]
    tau_level: float  # variação do nível entre pilotos/stints (inf = não estimada)
    offsets: dict[tuple[str, str], Offset]

    @property
    def kappa(self) -> float:
        return self.degradation.fuel.kappa

    def current(self, driver: str) -> StintLevel | None:
        """O nível do stint mais recente do piloto com nível estimado (o que ele está correndo)."""
        mine = [lvl for (d, _), lvl in self.levels.items() if d == driver]
        return max(mine, key=lambda lvl: lvl.stint) if mine else None

    def offset(self, from_compound: str, to_compound: str) -> Offset | None:
        if from_compound == to_compound:
            return Offset(from_compound, to_compound, 0.0, 0.0, 0.0, 0, "igual")
        return self.offsets.get((from_compound, to_compound))


def _sigma(model: DegradationModel, slope: SlopeEstimate) -> float:
    return slope.sigma_lap or model.sigma_lap or DEFAULT_SIGMA


PRIOR_DOF = 4  # o ruído global da corrida vale como estas voltas de dado, ao moderar o do stint


def _moderated_sigma(residual: np.ndarray, pooled: float) -> float:
    """O ruído de volta de UM stint, moderado: mistura a dispersão do próprio stint com a global da
    corrida, ponderadas pelos graus de liberdade (a moderação de variância dos modelos empíricos de
    Bayes). Só a global ignora um stint que ANDOU com muito ruído (4 voltas e 2 s de dispersão
    recebiam o erro de um stint tranquilo); só a do stint engana com 4 voltas (a dispersão de 4
    números é quase um palpite). Com 20+ voltas o stint manda e o efeito some."""
    dof = len(residual) - 1
    if dof < 1:
        return pooled
    own = float(np.std(residual, ddof=1))
    return float(np.sqrt((PRIOR_DOF * pooled**2 + dof * own**2) / (PRIOR_DOF + dof)))


def _stint_levels(
    laps: Sequence[LapRow], model: DegradationModel
) -> list[tuple[StintLevel, float]]:
    """Os níveis BRUTOS de cada stint e a incerteza dele na idade zero (para o encolhimento)."""
    stints: dict[tuple[str, int], list[LapRow]] = defaultdict(list)
    for lap in laps:
        stints[(lap.driver, lap.stint)].append(lap)
    kappa = model.fuel.kappa
    out = []
    for (driver, stint), group in sorted(stints.items()):
        if len(group) < MIN_LEVEL_LAPS:
            continue
        compound = group[0].compound
        slope = model.slope(driver, compound)
        if slope is None:
            continue
        age = np.array([lap.tyre_life for lap in group], dtype=float)
        seconds = np.array([lap.lap_time_ms for lap in group], dtype=float) / 1000
        corrected = correct_for_fuel(np.array([lap.lap_number for lap in group]), seconds, kappa)
        residual = corrected - slope.slope * age
        raw = float(np.median(residual))
        se_ref = _moderated_sigma(residual, _sigma(model, slope)) / np.sqrt(len(group))
        se0 = float(np.hypot(se_ref, float(age.mean()) * slope.se))
        level = StintLevel(
            driver, compound, stint, raw, raw, se_ref, float(age.mean()), len(group),
            min(lap.lap_number for lap in group), max(lap.lap_number for lap in group), slope, 1.0,
        )  # fmt: skip
        out.append((level, se0))
    return out


def _offsets(
    stints: list[StintLevel],
    ses: dict[tuple[str, int], float],
    compound_levels: dict[str, CompoundLevel],
    tau_level: float,
    race_sd: float,
) -> dict[tuple[str, str], Offset]:
    offsets: dict[tuple[str, str], Offset] = {}
    by_driver: dict[str, dict[str, list[StintLevel]]] = defaultdict(lambda: defaultdict(list))
    for lvl in stints:
        by_driver[lvl.driver][lvl.compound].append(lvl)

    def mean_and_se(group: list[StintLevel]) -> tuple[float, float]:
        values = [lvl.level0 for lvl in group]
        sds = [ses[(lvl.driver, lvl.stint)] for lvl in group]
        return float(np.mean(values)), float(np.sqrt(np.sum(np.square(sds))) / len(group))

    for a in sorted(compound_levels):
        for b in sorted(compound_levels):
            if a == b:
                continue
            diffs, sds = [], []
            for per_compound in by_driver.values():
                if a in per_compound and b in per_compound:
                    (ma, sa), (mb, sb) = mean_and_se(per_compound[a]), mean_and_se(per_compound[b])
                    diffs.append(mb - ma)
                    sds.append(float(np.hypot(sa, sb)))
            if len(diffs) >= MIN_PAIRS:
                arr, se_arr = np.array(diffs), np.maximum(np.array(sds), 1e-6)
                tau, _ = estimate_tau(arr, se_arr, ["x"] * len(arr), float(np.std(arr, ddof=1)))
                delta, se = random_effects_mean(arr, se_arr, tau)
                offsets[(a, b)] = Offset(a, b, delta, se, tau, len(arr), "pares", race_sd)
            else:
                ca, cb = compound_levels[a], compound_levels[b]
                tau = tau_level if np.isfinite(tau_level) else 0.0
                se = float(np.hypot(ca.se, cb.se))
                offsets[(a, b)] = Offset(
                    a, b, cb.mean0 - ca.mean0, se, tau, len(diffs), "médias", race_sd
                )
    return offsets


def fit_pace(
    rows: Sequence[LapRow],
    *,
    upto_lap: int,
    total_laps: int,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    priors: Priors = DEFAULT_PRIORS,
    kappa: float | None = None,
) -> PaceModel:
    """O nível de ritmo por stint, as médias por composto e as diferenças entre compostos, na volta
    `upto_lap`, só com o que se sabia nela."""
    visible = laps_until(rows, upto_lap)  # nada do futuro passa daqui
    model = fit_degradation(
        visible,
        upto_lap=upto_lap,
        total_laps=total_laps,
        variant=variant,
        priors=priors,
        kappa=kappa,
    )
    raw = _stint_levels(dry_clean_laps(visible), model)
    if not raw:
        return PaceModel(upto_lap, model, {}, {}, Z_NONE, {})

    values = np.array([lvl.raw_level0 for lvl, _ in raw])
    se0 = np.array([s for _, s in raw])
    strata = [lvl.compound for lvl, _ in raw]
    tau_level, source = estimate_tau(values, se0, strata, Z_NONE)

    compound_levels: dict[str, CompoundLevel] = {}
    for compound in sorted(set(strata)):
        idx = [i for i, c in enumerate(strata) if c == compound]
        mean, se = random_effects_mean(
            values[idx], se0[idx], tau_level if np.isfinite(tau_level) else 0.0
        )
        compound_levels[compound] = CompoundLevel(compound, mean, se, len(idx))

    levels: dict[tuple[str, int], StintLevel] = {}
    ses: dict[tuple[str, int], float] = {}
    for lvl, s0 in raw:
        prior = compound_levels[lvl.compound]
        if np.isfinite(tau_level):
            level0, se_post, weight = shrink_slope(
                lvl.raw_level0, s0, prior.mean0, prior.se, tau_level
            )
            se_ref = float(lvl.se_ref * np.sqrt(weight) + (1 - weight) * tau_level)
        else:
            level0, weight, se_ref = lvl.raw_level0, 1.0, lvl.se_ref
            se_post = s0
        key = (lvl.driver, lvl.stint)
        levels[key] = StintLevel(
            lvl.driver, lvl.compound, lvl.stint, level0, lvl.raw_level0, se_ref, lvl.mean_age,
            lvl.n_laps, lvl.first_lap, lvl.last_lap, lvl.slope, weight,
        )  # fmt: skip
        ses[key] = se_post
    offsets = _offsets(list(levels.values()), ses, compound_levels, tau_level, priors.offset_sd)
    return PaceModel(upto_lap, model, levels, compound_levels, tau_level, offsets)
