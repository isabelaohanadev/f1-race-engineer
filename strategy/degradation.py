"""Modelo de degradação do pneu (`python -m strategy.degradation`): a inclinação do tempo de volta
contra a idade do pneu, por composto e por piloto, com a incerteza dita em DUAS partes.

O QUE É A "DEGRADAÇÃO"
Quantos segundos a volta fica mais lenta a cada volta de idade do pneu (s por volta de idade),
depois de corrigido o combustível (`strategy.fuel`). Para cada stint ajusta-se uma reta do tempo
corrigido contra a idade; o NÍVEL do stint é irrelevante (cada um tem o seu: piloto, composto,
pista), só a inclinação importa. É `fit_within`: um intercepto por stint e uma inclinação, com
Huber.

HIERARQUIA (a ideia central, e a resposta de "por que encolhimento")
Há três níveis, do mais geral ao mais específico: a corrida inteira, o composto, e o par
(piloto, composto). Um piloto com 8 voltas num composto tem uma inclinação com erro enorme; a média
do composto na corrida (de ~20 pilotos) é estável, mas não é "dele". O encolhimento
(normal-normal, "efeitos aleatórios") pesa as duas coisas pela precisão de cada uma:

    inclinação_do_piloto = prior + B * (observada - prior),   B = t^2 / (t^2 + se^2)

Com pouco dado, `se` é grande, `B` é pequeno e vale o prior (a média do composto); com muito dado,
`B` tende a 1 e vale o que o piloto fez. `t` é a variação REAL da inclinação entre pilotos/stints
da mesma corrida, estimada dos próprios dados (DerSimonian-Laird, a mesma conta da meta-análise de
efeitos aleatórios). Sem encolhimento, as voltas de um piloto num composto raro mandariam
sozinhas, e o erro-padrão delas é grande demais para isso.

AS DUAS INCERTEZAS (e por que misturá-las é o erro clássico)
* `se`: a incerteza do PARÂMETRO, o quanto a inclinação é incerta por termos poucas voltas e ruído.
  Cai com mais dados. Mede só ruído de amostragem e supõe voltas independentes.
* `sigma_lap`: o RUÍDO de uma volta em torno da reta. Não cai com mais dados (mais voltas só o medem
  melhor) e é o que a simulação soma a cada volta futura.
Um terceiro número vem à parte, `tau`, a variação da inclinação ENTRE pilotos/stints da corrida. O
erro-padrão não a vê: ele diz "conheço bem a inclinação desse stint", não "o PRÓXIMO stint vai ter
essa inclinação". Para prever um stint futuro (o undercut), a incerteza é
`predictive_sd = sqrt(se^2 + tau^2)`, e usar só `se` deixaria o alerta confiante demais. É
conservadora: `tau` mistura diferença entre pilotos e entre stints do mesmo piloto, e o piloto já
tem parte da primeira no próprio dado.

ÚLTIMO RECURSO: se ninguém rodou o composto ainda, vale a média dos outros, e a incerteza soma o
quanto os compostos diferem entre si (com um piso, porque dois compostos conhecidos podem ser
parecidos por acaso). É o caso mais difícil de prever: ver a cobertura por tipo de previsão em
docs/degradacao.md, e o campo `source`, que deixa quem usa a previsão (o undercut) desconfiar dela.

SEM VAZAMENTO DO FUTURO: `fit_degradation` corta as voltas depois de `upto_lap` ANTES de qualquer
outra coisa (e o coeficiente de combustível, idem), então o modelo da volta N só conhece o que se
sabia na volta N. O teste corrompe todas as voltas depois de N e confere que nada muda.
"""

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np

from db.session import get_session_factory
from strategy.fuel import (
    FuelCoefficient,
    FuelVariant,
    correct_for_fuel,
    dry_clean_laps,
    fuel_coefficient,
    laps_until,
)
from strategy.laps import LapRow, load_race_laps
from strategy.population import split_population
from strategy.priors import DEFAULT_PRIORS, Priors
from strategy.regression import fit_within
from strategy.tables import pct, render_markdown, render_text

MIN_STINT_LAPS = 3  # um stint com menos voltas não ensina nada sobre inclinação (e o resíduo é 0)
MIN_UNIT_LAPS = 6  # um (piloto, composto) com menos voltas não ganha inclinação própria
MIN_DF_TAU = 3  # graus de liberdade mínimos para estimar a variação entre pilotos/stints
MIN_SIGMA_LAPS = 100  # menos resíduos que isto: o ruído do composto usa o da corrida
SE_FLOOR = 1e-4  # um ajuste perfeito não pode dar erro-padrão zero (peso infinito)
# Os valores de reserva (variação entre pilotos/stints quando a corrida ainda não tem dados para
# estimá-la, e o piso da diferença entre compostos no último recurso) vêm de `strategy.priors`: o
# padrão abaixo usa as 21 corridas todas, e um backtest passa `priors=` sem a corrida avaliada.
TAU_FALLBACK = DEFAULT_PRIORS.tau
BETWEEN_COMPOUNDS_FLOOR = DEFAULT_PRIORS.between_compounds
Z95 = 1.96
FUTURE_FRACTION = 0.5  # a calibração prevê os stints que começam depois de metade da corrida

Source = Literal["piloto", "composto", "todos os compostos"]


# ---------------------------------------------------------------- peças puras


def random_effects_mean(slopes: np.ndarray, ses: np.ndarray, tau: float) -> tuple[float, float]:
    """Média ponderada por 1/(se^2 + tau^2) e o seu erro-padrão, `1/sqrt(soma dos pesos)`."""
    weights = 1.0 / (ses**2 + tau**2)
    return float(np.sum(weights * slopes) / np.sum(weights)), float(1.0 / np.sqrt(np.sum(weights)))


def estimate_tau(
    slopes: np.ndarray, ses: np.ndarray, strata: Sequence[str], fallback: float = TAU_FALLBACK
) -> tuple[float, Literal["estimado", "padrão"]]:
    """Variação REAL da inclinação entre unidades (pilotos/stints), por DerSimonian-Laird.

    Se todas as unidades tivessem a mesma inclinação, as diferenças observadas seriam só ruído de
    amostragem e `Q = soma(w * (b - media)^2)` valeria ~ os graus de liberdade. O que passa disso é
    variação verdadeira: `tau^2 = (Q - gl) / C`. As médias são POR COMPOSTO (estratos), porque os
    compostos têm degradações diferentes e isso não é heterogeneidade."""
    labels = np.asarray(strata)
    weights = 1.0 / ses**2
    q = c = 0.0
    df = 0
    for stratum in sorted(set(strata)):
        mask = labels == stratum
        w, b = weights[mask], slopes[mask]
        if mask.sum() < 2:
            continue
        mean = np.sum(w * b) / np.sum(w)
        q += float(np.sum(w * (b - mean) ** 2))
        c += float(np.sum(w) - np.sum(w**2) / np.sum(w))
        df += int(mask.sum()) - 1
    if df < MIN_DF_TAU or c <= 0:
        return fallback, "padrão"
    return float(np.sqrt(max(0.0, (q - df) / c))), "estimado"


def shrink_slope(
    observed: float, se: float, prior_mean: float, prior_se: float, tau: float
) -> tuple[float, float, float]:
    """`(inclinação, erro-padrão, peso do dado do piloto)` do piloto encolhido para o prior.

    `B = tau^2 / (tau^2 + se^2)` é o peso do que o piloto fez. O erro-padrão do resultado soma o do
    dado dele (ponderado) e o do prior (ponderado pelo que sobrou): `B*se^2 + (1-B)^2 * prior_se^2`.
    """
    weight = tau**2 / (tau**2 + se**2)
    slope = prior_mean + weight * (observed - prior_mean)
    return slope, float(np.sqrt(weight * se**2 + (1 - weight) ** 2 * prior_se**2)), weight


# ---------------------------------------------------------------- o modelo


@dataclass(frozen=True, slots=True)
class SlopeEstimate:
    """A degradação de um (piloto, composto) e TODAS as incertezas, cada uma com o seu nome."""

    driver: str
    compound: str
    slope: float  # s por volta de idade, com o combustível corrigido
    se: float  # incerteza do PARÂMETRO (amostragem), já com o encolhimento
    tau: float  # variação da inclinação entre pilotos/stints desta corrida
    predictive_sd: float  # sqrt(se^2 + tau^2): a incerteza de um stint FUTURO
    sigma_lap: float | None  # o RUÍDO de uma volta (não é incerteza do parâmetro)
    n_laps: int  # voltas deste piloto neste composto
    weight: float  # peso do dado do piloto (1 = só ele, 0 = só o prior)
    source: Source


@dataclass(frozen=True, slots=True)
class CompoundPrior:
    compound: str
    mean: float  # a inclinação típica do composto nesta corrida
    se: float
    n_units: int  # (piloto, composto) que entraram
    n_laps: int


@dataclass(frozen=True, slots=True)
class DegradationModel:
    upto_lap: int
    fuel: FuelCoefficient
    sigma_lap: float | None  # ruído de volta da corrida
    sigma_by_compound: dict[str, float]
    tau: float
    tau_source: Literal["estimado", "padrão"]
    compounds: dict[str, CompoundPrior]
    pooled: CompoundPrior | None  # todos os compostos juntos: o último recurso
    # O quanto a degradação difere ENTRE compostos (desvio das médias dos compostos). Só importa no
    # último recurso: prever um composto que ninguém rodou ainda pela média dos outros.
    between_compounds: float | None
    units: dict[tuple[str, str], SlopeEstimate]
    between_floor: float = BETWEEN_COMPOUNDS_FLOOR  # o piso usado no último recurso (de `Priors`)

    def slope(self, driver: str, compound: str) -> SlopeEstimate | None:
        """A degradação a usar para `driver` em `compound`: o dado dele (encolhido), senão a média
        do composto na corrida, senão a de todos os compostos, senão None (corrida sem dados)."""
        own = self.units.get((driver, compound))
        if own is not None:
            return own
        prior = self.compounds.get(compound)
        if prior is not None:
            return self._from_prior(driver, compound, prior, extra=0.0, source="composto")
        if self.pooled is not None:
            # Ninguém rodou este composto ainda: a média dos OUTROS compostos é um chute, e o erro
            # dele inclui o quanto os compostos diferem entre si (SOFT degrada mais que HARD). Com
            # um composto só conhecido, não há como medir essa diferença: assume-se que o composto
            # que falta pode diferir do conhecido por algo do tamanho dele (|média|).
            spread = max(
                self.between_compounds
                if self.between_compounds is not None
                else abs(self.pooled.mean),
                self.between_floor,
            )
            return self._from_prior(
                driver, compound, self.pooled, extra=spread, source="todos os compostos"
            )
        return None

    def _from_prior(
        self, driver: str, compound: str, prior: CompoundPrior, *, extra: float, source: Source
    ) -> SlopeEstimate:
        return SlopeEstimate(
            driver=driver,
            compound=compound,
            slope=prior.mean,
            se=float(np.sqrt(prior.se**2 + extra**2)),
            tau=self.tau,
            predictive_sd=float(np.sqrt(prior.se**2 + extra**2 + self.tau**2)),
            sigma_lap=self.sigma_by_compound.get(compound, self.sigma_lap),
            n_laps=0,
            weight=0.0,
            source=source,
        )


@dataclass(frozen=True, slots=True)
class _Unit:
    driver: str
    compound: str
    slope: float
    se: float
    n_laps: int


def _units(laps: Sequence[LapRow], kappa: float) -> tuple[list[_Unit], dict[str, list[float]]]:
    """Ajusta cada (piloto, composto) e junta os resíduos (já corrigidos pelos graus de liberdade)
    por composto, para estimar o ruído de volta."""
    by_unit: dict[tuple[str, str], list[LapRow]] = defaultdict(list)
    for lap in laps:
        by_unit[(lap.driver, lap.compound)].append(lap)

    units: list[_Unit] = []
    residuals: dict[str, list[float]] = defaultdict(list)
    for (driver, compound), group in sorted(by_unit.items()):
        stint_sizes: dict[int, int] = defaultdict(int)
        for lap in group:
            stint_sizes[lap.stint] += 1
        group = [lap for lap in group if stint_sizes[lap.stint] >= MIN_STINT_LAPS]
        if len(group) < MIN_UNIT_LAPS:
            continue
        age = np.array([lap.tyre_life for lap in group], dtype=float)
        seconds = np.array([lap.lap_time_ms for lap in group], dtype=float) / 1000
        corrected = correct_for_fuel(np.array([lap.lap_number for lap in group]), seconds, kappa)
        fit = fit_within(age, corrected, [lap.stint for lap in group])
        if fit.slope is None or fit.se is None:
            continue
        units.append(_Unit(driver, compound, fit.slope, max(fit.se, SE_FLOOR), len(group)))
        # O ajuste "encolhe" os resíduos (ele escolheu a reta que os minimiza): desfaz isso.
        dof = fit.n_obs - fit.n_groups - 1
        if dof > 0:
            residuals[compound] += list(fit.residuals * np.sqrt(fit.n_obs / dof))
    return units, residuals


def _robust_sd(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(1.4826 * np.median(np.abs(arr - np.median(arr))))


def fit_degradation(
    rows: Sequence[LapRow],
    *,
    upto_lap: int,
    total_laps: int,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    priors: Priors = DEFAULT_PRIORS,
    kappa: float | None = None,
) -> DegradationModel:
    """O modelo de degradação da corrida na volta `upto_lap`, só com o que se sabia nela.

    `priors`: as constantes de reserva (use `leave_one_out` num backtest). `kappa`: força o `k`
    (para medir o quanto o resultado depende dele); o padrão é o da variante, estimado só com o
    que se sabia na volta."""
    visible = laps_until(rows, upto_lap)  # PRIMEIRO passo: nada do futuro passa daqui
    laps = dry_clean_laps(visible)
    fuel = fuel_coefficient(visible, upto_lap=upto_lap, total_laps=total_laps, variant=variant)
    if kappa is not None:
        fuel = replace(fuel, kappa=kappa)
    units, residuals = _units(laps, fuel.kappa)

    all_residuals = [value for values in residuals.values() for value in values]
    sigma_race = _robust_sd(all_residuals) if all_residuals else None
    sigma_by_compound = {
        compound: _robust_sd(values)
        for compound, values in residuals.items()
        if len(values) >= MIN_SIGMA_LAPS
    }

    if not units:
        return DegradationModel(
            upto_lap,
            fuel,
            sigma_race,
            sigma_by_compound,
            priors.tau,
            "padrão",
            {},
            None,
            None,
            {},
            priors.between_compounds,
        )

    slopes = np.array([u.slope for u in units])
    ses = np.array([u.se for u in units])
    tau, tau_source = estimate_tau(slopes, ses, [u.compound for u in units], priors.tau)

    compounds: dict[str, CompoundPrior] = {}
    for compound in sorted({u.compound for u in units}):
        mine = [u for u in units if u.compound == compound]
        mean, se = random_effects_mean(
            np.array([u.slope for u in mine]), np.array([u.se for u in mine]), tau
        )
        compounds[compound] = CompoundPrior(
            compound, mean, se, len(mine), sum(u.n_laps for u in mine)
        )
    pooled_mean, pooled_se = random_effects_mean(slopes, ses, tau)
    pooled = CompoundPrior(
        "todos", pooled_mean, pooled_se, len(units), sum(u.n_laps for u in units)
    )

    estimates: dict[tuple[str, str], SlopeEstimate] = {}
    for u in units:
        prior = compounds[u.compound]
        slope, se, weight = shrink_slope(u.slope, u.se, prior.mean, prior.se, tau)
        estimates[(u.driver, u.compound)] = SlopeEstimate(
            driver=u.driver,
            compound=u.compound,
            slope=slope,
            se=se,
            tau=tau,
            predictive_sd=float(np.sqrt(se**2 + tau**2)),
            sigma_lap=sigma_by_compound.get(u.compound, sigma_race),
            n_laps=u.n_laps,
            weight=weight,
            source="piloto",
        )
    means = [prior.mean for prior in compounds.values()]
    between = float(np.std(means, ddof=1)) if len(means) >= 2 else None
    return DegradationModel(
        upto_lap,
        fuel,
        sigma_race,
        sigma_by_compound,
        tau,
        tau_source,
        compounds,
        pooled,
        between,
        estimates,
        priors.between_compounds,
    )


# ---------------------------------------------------------------- calibração com stints futuros


@dataclass(frozen=True, slots=True)
class FutureStint:
    """Um stint que começou DEPOIS do corte, com a inclinação que ele realmente teve."""

    race: str
    driver: str
    compound: str
    realized: float
    realized_se: float
    predicted: SlopeEstimate

    def z(self, sd: float) -> float:
        """Quantos desvios a inclinação real ficou da prevista (somando o erro da medida dela)."""
        return (self.realized - self.predicted.slope) / float(np.sqrt(sd**2 + self.realized_se**2))

    @property
    def z_parameter_only(self) -> float:
        return self.z(self.predicted.se)

    @property
    def z_predictive(self) -> float:
        return self.z(self.predicted.predictive_sd)


def future_stints(
    label: str,
    rows: Sequence[LapRow],
    *,
    total_laps: int,
    fraction: float = FUTURE_FRACTION,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    kappa_from: Literal["modelo", "fim"] = "modelo",
) -> list[FutureStint]:
    """Os stints que começam depois de `fraction` da corrida, com a previsão do modelo cortado ali.

    `kappa_from="modelo"` mede a inclinação real com o MESMO `k` do modelo: isola a variação entre
    stints (o que `tau` deve cobrir). `"fim"` usa o `k` da corrida inteira, e então o erro do `k`
    também entra: é a pergunta realista, "e se o `k` da volta N estava errado?"."""
    cut = round(fraction * total_laps)
    model = fit_degradation(rows, upto_lap=cut, total_laps=total_laps, variant=variant)
    if kappa_from == "modelo":
        kappa = model.fuel.kappa
    else:
        kappa = fuel_coefficient(
            rows, upto_lap=total_laps, total_laps=total_laps, variant=variant
        ).kappa

    laps = dry_clean_laps(rows)
    stints: dict[tuple[str, int], list[LapRow]] = defaultdict(list)
    for lap in laps:
        stints[(lap.driver, lap.stint)].append(lap)
    out = []
    for (driver, _), group in sorted(stints.items()):
        if min(lap.lap_number for lap in group) <= cut or len(group) < MIN_UNIT_LAPS:
            continue  # começou antes do corte (o modelo já viu parte dele) ou é curto demais
        compound = group[0].compound
        age = np.array([lap.tyre_life for lap in group], dtype=float)
        seconds = np.array([lap.lap_time_ms for lap in group], dtype=float) / 1000
        corrected = correct_for_fuel(np.array([lap.lap_number for lap in group]), seconds, kappa)
        fit = fit_within(age, corrected, [0] * len(group))
        predicted = model.slope(driver, compound)
        if fit.slope is None or fit.se is None or predicted is None:
            continue
        out.append(
            FutureStint(label, driver, compound, fit.slope, max(fit.se, SE_FLOOR), predicted)
        )
    return out


def coverage(stints: Sequence[FutureStint], which: Literal["param", "predictive"]) -> float | None:
    """Fração dos stints cuja inclinação real caiu dentro do intervalo de 95% previsto."""
    if not stints:
        return None
    key = (lambda s: s.z_parameter_only) if which == "param" else (lambda s: s.z_predictive)
    return sum(abs(key(s)) <= Z95 for s in stints) / len(stints)


# ---------------------------------------------------------------- relatório


@dataclass(slots=True)
class RaceModelReport:
    label: str
    total_laps: int
    model: DegradationModel
    future: list[FutureStint]
    future_end_k: list[FutureStint]
    by_variant: dict[FuelVariant, DegradationModel]


def analyze_race(label: str, rows: Sequence[LapRow], total_laps: int) -> RaceModelReport:
    by_variant = {
        variant: fit_degradation(rows, upto_lap=total_laps, total_laps=total_laps, variant=variant)
        for variant in FuelVariant
    }
    return RaceModelReport(
        label=label,
        total_laps=total_laps,
        model=by_variant[FuelVariant.ENCOLHIMENTO],
        future=future_stints(label, rows, total_laps=total_laps),
        future_end_k=future_stints(label, rows, total_laps=total_laps, kappa_from="fim"),
        by_variant=by_variant,
    )


def load_reports(
    session, year: int | None = None
) -> tuple[list[RaceModelReport], list[RaceModelReport], list[str]]:
    """(backtest principal, corridas à parte com o motivo no rótulo, rótulos)."""
    main, separate = split_population(session, year)

    def run(refs):
        return [
            analyze_race(r.label, load_race_laps(session, r.race_id), r.total_laps) for r in refs
        ]

    return run(main), run(separate), [f"{r.label}: {r.reason}" for r in separate]


COMPOUND_ORDER = ("HARD", "MEDIUM", "SOFT")


def _cell(model: DegradationModel, compound: str) -> str:
    prior = model.compounds.get(compound)
    if prior is None:
        return "-"
    return f"{prior.mean:+.3f} ± {prior.se:.3f} ({prior.n_units})"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def race_table(reports: Sequence[RaceModelReport]) -> list[list[str]]:
    rows = [
        [
            "corrida",
            "k (s/volta)",
            "peso do k estimado",
            "ruído de volta σ",
            "variação entre pilotos/stints τ",
            *[f"{c} (± erro-padrão, unidades)" for c in COMPOUND_ORDER],
        ]
    ]
    for r in reports:
        m = r.model
        rows.append(
            [
                r.label,
                _fmt(m.fuel.kappa),
                _fmt(m.fuel.weight, 2),
                _fmt(m.sigma_lap),
                _fmt(m.tau) + ("" if m.tau_source == "estimado" else " (padrão)"),
                *[_cell(m, c) for c in COMPOUND_ORDER],
            ]
        )
    return rows


def variant_table(reports: Sequence[RaceModelReport]) -> list[list[str]]:
    rows = [["variante do k", *[f"{c}: mediana entre corridas" for c in COMPOUND_ORDER]]]
    for variant in FuelVariant:
        cells = []
        for compound in COMPOUND_ORDER:
            values = [
                r.by_variant[variant].compounds[compound].mean
                for r in reports
                if compound in r.by_variant[variant].compounds
            ]
            cells.append(f"{median(values):+.3f}" if values else "-")
        rows.append([variant.value, *cells])
    return rows


def calibration_table(reports: Sequence[RaceModelReport]) -> list[list[str]]:
    def row(name: str, stints: Sequence[FutureStint]) -> list[str]:
        param, predictive = coverage(stints, "param"), coverage(stints, "predictive")
        return [
            name,
            str(len(stints)),
            "-" if param is None else pct(param),
            "-" if predictive is None else pct(predictive),
        ]

    rows = [["stints futuros", "n", "cobertura só com o erro-padrão", "cobertura com a variação τ"]]
    same_k = [s for r in reports for s in r.future]
    end_k = [s for r in reports for s in r.future_end_k]
    rows.append(row("`k` do modelo (isola a variação entre stints)", same_k))
    rows.append(row("`k` da corrida inteira (inclui o erro do `k`)", end_k))
    for source in ("piloto", "composto", "todos os compostos"):
        rows.append(
            row(
                f"  só previsões por {source} (k do modelo)",
                [s for s in same_k if s.predicted.source == source],
            )
        )
    return rows


def _summary(reports: Sequence[RaceModelReport]) -> list[str]:
    if not reports:
        return ["- Nenhuma corrida no backtest principal."]
    sigmas = [r.model.sigma_lap for r in reports if r.model.sigma_lap is not None]
    taus = [r.model.tau for r in reports if r.model.tau_source == "estimado"]
    same_k = [s for r in reports for s in r.future]
    end_k = [s for r in reports for s in r.future_end_k]
    lines = [
        f"- {len(reports)} corridas do backtest principal, modelo ajustado com a corrida inteira "
        "e a variante `encolhimento` do `k`.",
        f"- Ruído de volta (σ), mediana entre corridas: {median(sigmas):.3f} s. É a dispersão de "
        "UMA volta em torno da reta; não cai com mais dados.",
        "- Variação da inclinação entre pilotos/stints (τ), mediana: "
        f"{median(taus):.3f} s/volta de idade ({len(taus)} de {len(reports)} corridas com dados "
        f"para estimá-la; as demais usam o padrão {TAU_FALLBACK}).",
    ]
    for name, stints in (("`k` do modelo", same_k), ("`k` da corrida inteira", end_k)):
        param, predictive = coverage(stints, "param"), coverage(stints, "predictive")
        if param is not None and predictive is not None:
            lines.append(
                f"- Stints futuros ({name}, n={len(stints)}): o intervalo de 95% só com o "
                f"erro-padrão cobre a inclinação real em {pct(param)}; com a variação τ, em "
                f"{pct(predictive)}. O alvo é 95%."
            )
    return lines


MARKDOWN_METHOD = """\
## Método

- **Dados**: voltas limpas (status exatamente `1`, sem box, sem volta 1) e de pneu de pista seca,
  só até a volta N. O tempo é corrigido do combustível com a variante do `k` escolhida.
- **Por (piloto, composto)**: reta do tempo corrigido contra a idade, com um intercepto por stint e
  Huber (`fit_within`). Precisa de pelo menos 6 voltas, em stints de pelo menos 3.
- **Variação entre pilotos/stints τ**: DerSimonian-Laird sobre as inclinações dessas unidades, com
  médias por composto. Se sobram menos de 3 graus de liberdade, usa o padrão (0,03).
- **Encolhimento**: a inclinação do piloto vai para a média do composto na corrida com peso
  `B = τ² / (τ² + se²)`. Sem dado do piloto naquele composto, vale a média do composto; sem dado do
  composto, a de todos os compostos juntos; sem nada, não há modelo.
- **Três números de incerteza, separados**: `se` (parâmetro; cai com dados), `σ` (ruído de uma
  volta; não cai) e `τ` (variação entre pilotos/stints). `predictive_sd = sqrt(se² + τ²)` é a de
  um stint FUTURO, a que o undercut deve usar.
- **Calibração**: para os stints que começam depois de 50% da corrida, o intervalo previsto com o
  modelo cortado em 50% cobre a inclinação que o stint teve de fato? Compara o intervalo só com o
  erro-padrão e o intervalo com `τ`. Soma o erro da própria medida do stint.
"""


def build_report(
    main: Sequence[RaceModelReport],
    separate: Sequence[RaceModelReport],
    skipped: Sequence[str],
    *,
    markdown: bool,
    generated_on: date | None = None,
) -> str:
    summary = "\n".join(_summary(main))
    if not markdown:
        return (
            f"{summary}\n\nPOR CORRIDA (corrida inteira)\n{render_text(race_table(main))}\n\n"
            f"CALIBRAÇÃO COM STINTS FUTUROS\n{render_text(calibration_table(main))}\n\n"
            f"VARIANTES DO k\n{render_text(variant_table(main))}\n\n"
            f"À PARTE (fora da nota principal): {'; '.join(skipped) or 'nenhuma'}\n"
            + (render_text(race_table(separate)) + "\n" if separate else "")
        )
    day = (generated_on or date.today()).isoformat()
    return (
        "# Modelo de degradação do pneu\n\n"
        "Gerado por `python -m strategy.degradation --markdown docs/degradacao.md` em "
        f"{day}. A regra está em `strategy/degradation.py`.\n\n"
        f"## Resumo\n\n{summary}\n\n{MARKDOWN_METHOD}\n"
        "## Por corrida, com a corrida inteira\n\n"
        "Inclinação do composto na corrida (s por volta de idade, `+` = o pneu fica mais lento), "
        "± erro-padrão, e entre parênteses quantos (piloto, composto) entraram.\n\n"
        f"{render_markdown(race_table(main))}\n\n"
        "## Calibração: o intervalo cobre o stint que ainda não aconteceu?\n\n"
        f"{render_markdown(calibration_table(main))}\n\n"
        "## O `k` muda a degradação: as três variantes\n\n"
        "Mediana, entre as corridas, da inclinação do composto. Cada +0,01 s/volta no `k` soma "
        "+0,01 s/volta à inclinação (ver `docs/combustivel.md`); o backtest decide a variante.\n\n"
        f"{render_markdown(variant_table(main))}\n\n"
        "## Reportadas à parte (fora da nota principal)\n\n"
        "Pista mista ou molhada: o modelo ajusta só as voltas de pneu de seco, e nelas a pista "
        "está secando ou molhada, então os números abaixo **não medem a qualidade do modelo**. "
        "Estão "
        f"aqui para serem vistos, não para entrar em nenhuma média.\n\n"
        f"Corridas: {'; '.join(skipped) or 'nenhuma'}.\n\n"
        + (f"{render_markdown(race_table(separate))}\n" if separate else "")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.degradation",
        description="Ajusta o modelo de degradação em cada corrida e confere a calibração.",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="grava o relatório neste arquivo .md"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        main_reports, separate_reports, skipped = load_reports(session, args.year)
    if not main_reports and not separate_reports:
        print("Nenhuma corrida no banco. Ingira com: python -m ingestion --year 2024 --round 10")
        return 1
    print(build_report(main_reports, separate_reports, skipped, markdown=False))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            build_report(main_reports, separate_reports, skipped, markdown=True), encoding="utf-8"
        )
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
