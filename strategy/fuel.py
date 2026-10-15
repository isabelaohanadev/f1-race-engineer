"""Correção de combustível (`python -m strategy.fuel`): estimado x físico e sensibilidade.

POR QUE CORRIGIR O COMBUSTÍVEL (a pergunta de entrevista)
O carro larga com ~110 kg de combustível e termina com quase nada. Cada kg custa décimos de segundo
por volta, então a volta N é mais rápida que a volta 1 só por estar mais leve. O pneu faz o oposto:
envelhece e fica mais lento. Os dois efeitos acontecem juntos, com sinais contrários e do mesmo
tamanho: sem corrigir, o desgaste que a regressão enxerga é o desgaste real MENOS o ganho de
combustível. Medir "o pneu perdeu 0,03 s por volta" quando na verdade perdeu 0,09 e o carro ganhou
0,06 por ficar leve faria o undercut parecer três vezes menos vantajoso do que é.

A CORREÇÃO. `tempo_corrigido = tempo + k * (volta - 1)`: o tempo que a volta teria com o combustível
da volta 1. `k` são os segundos que o carro ganha por volta ao queimar combustível. A constante da
volta 1 não importa (só a inclinação), e é a mesma convenção que a simulação usa ao projetar voltas.

DE ONDE VEM `k`
1. FÍSICO (suposição): 110 kg (limite do regulamento) queimados linearmente ao longo da corrida, a
   ~0,03 s por kg (ordem de grandeza típica de engenharia de corrida, 0,025 a 0,035; NÃO medida
   aqui). Dá `k = 0,03 * 110 / voltas`: ~0,058 s/volta com 57 voltas, ~0,042 com 78 (Mônaco), ~0,075
   com 44.
2. ESTIMADO, junto com a degradação, nos stints de vários pilotos da mesma corrida.

POR QUE O ESTIMADO NÃO SAI DE UM AJUSTE COM INTERCEPTO POR STINT
Dentro de um stint, `idade = volta - c` (c constante), então idade e volta andam juntas. Se cada
stint tem o seu intercepto, `k * volta` vira `k * idade + constante` e é ABSORVIDO pela inclinação
da idade: só se identifica `degradação - k`, nunca os dois (e daí a sensibilidade ser exatamente 1
para 1). O que separa os dois é comparar stints que começam em voltas diferentes (a idade 10 de
um piloto acontece na volta 12, a de outro na volta 35). Por isso o ajuste conjunto usa
intercepto POR PILOTO (e por composto), e não por stint:
`tempo = piloto + composto + degradação_composto * idade - k * volta`.

O QUE O `k` ESTIMADO TAMBÉM CONTÉM (e por que pode divergir do físico)
Tudo o que melhora o tempo com a volta da corrida e não é pneu: além do combustível, a pista que
"borracha" (fica mais aderente) e a temperatura. Ele mede a TENDÊNCIA DE RITMO POR VOLTA DA CORRIDA,
que é o que se precisa tirar para isolar o pneu, mas não é só combustível. Assume também que o nível
de ritmo de um piloto é o mesmo entre stints (intercepto por piloto) e que a degradação é linear na
idade.

QUANDO USAR O ESTIMADO (critério FIXADO ANTES de olhar os resultados, para não ajustá-lo ao que sai)
Só se for identificável, PRECISO (erro-padrão <= metade do físico: uma estimativa mais incerta que o
próprio efeito não informa nada) e PARECIDO com o físico (razão entre 0,5 e 2). Em qualquer outro
caso, usa o físico e diz por quê. Com dados só até a volta N (sem vazamento do futuro) o estimador
muitas vezes ainda não se identifica: então usa o físico, que não depende dos dados.

ENCOLHIMENTO CONTÍNUO (a troca de fonte sem salto)
A regra acima TROCA de físico para estimado num ponto da corrida, e a troca é um degrau em `k` que
passa 1 para 1 para a degradação. O encolhimento a substitui por uma média ponderada:
`k = w * estimado + (1 - w) * físico`, com `w` entre 0 e 1 crescendo com a quantidade de dados
(voltas até N). `w` é o peso de precisão da média normal-normal, `w = t^2 / (t^2 + se^2)`, em
que `t` é o quanto se aceita que o `k` real da corrida se afaste do físico (a priori, 50% dele)
e `se` é o erro-padrão do estimado. Sem estimativa (não identificável), `se` é infinito e
`w = 0`: é o físico, e a função é contínua nesse ponto (não há "ligar"; o `w` sobe de 0
conforme os dados chegam).

TRÊS COISAS QUE A PRIMEIRA VERSÃO ERROU (medido nas 21 corridas secas; o critério acima não mudou)
1. Com o erro-padrão por cluster, o encolhimento ficou MENOS suave que a regra binária (maior
   passo entre voltas de 0,066 s/volta, em mediana, contra 0,024; 0,449 em Singapura). Cedo na
   corrida o estimador devolve valores absurdos (até 19x o físico) com erro-padrão pequeno, e
   qualquer `w > 0` deixa passar um estimador sem limite: o portão de razão (0,5x a 2x) da regra
   binária protegia disso sem que eu tivesse notado. Agora o estimado entra LIMITADO a esse mesmo
   intervalo (um teto contínuo de influência, não uma chave liga-desliga) e o `se` é o MAIOR entre
   o erro-padrão por cluster e o do jackknife por piloto (o quanto o estimado muda quando se tira
   um piloto por vez): uma estimativa que depende de 2 pilotos tem jackknife grande e peso quase
   nulo, mesmo com o erro-padrão clássico pequeno.
2. Mesmo assim o encolhimento por precisão herdava o degrau: a identificação pode chegar de uma
   volta para outra (numa corrida sintética, o `k` não existe na volta 15 e tem erro-padrão de
   0,0008 na 16, quando o primeiro piloto completa um segundo stint), e `w` por precisão vai de 0
   a 1 junto.
   Continuidade em `(estimado, erro-padrão)` não basta se as entradas pulam. O que cresce de forma
   gradual são as PARADAS já vistas (cada uma é uma comparação de stints): `w = dados * precisão`,
   com `dados = min(1, paradas / pilotos)`. Nasce perto de 0 e só chega a 1 quando, em média, cada
   piloto já parou uma vez; é monótono em N por construção.
3. `w` como um todo cresce com os dados em tendência, mas NÃO é monótono volta a volta (a precisão
   oscila). Não finjo o contrário: a tendência é testada e a fração de voltas em que `w` cai é
   reportada.

TRÊS VARIANTES PARA O BACKTEST: `fisico` (sempre o físico), `estimado` (a regra binária) e
`encolhimento`. Os números do backtest (etapa 8) decidem qual vira o padrão; nenhuma é "a certa" por
construção, e o critério da regra binária não é apertado aqui.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np

from db.session import get_session_factory
from strategy.cleaning import clean_laps
from strategy.laps import DRY_COMPOUNDS, LapRow, load_race_laps
from strategy.population import split_population
from strategy.regression import fit_linear, one_hot
from strategy.tables import pct, render_markdown, render_text

# ---------------------------------------------------------------- a suposição física

FUEL_START_KG = 110.0  # limite de combustível do regulamento da F1 (2024)
# Efeito de 1 kg no tempo de volta. SUPOSIÇÃO de engenharia (0,025 a 0,035 conforme o circuito), não
# medida com estes dados: é o que a comparação com o estimado testa.
FUEL_S_PER_KG = 0.03

# ---------------------------------------------------------------- o critério (fixado a priori)

MAX_RATIO = 2.0  # o estimado só vale se estiver entre MAX_RATIO vezes e 1/MAX_RATIO do físico
MAX_REL_SE = 0.5  # ... e com erro-padrão de no máximo metade do físico
MIN_LAPS = 60  # abaixo disto o ajuste (piloto + composto + inclinações + volta) não se sustenta
MIN_STINTS = 6
MIN_DRIVERS = 4
# Encolhimento contínuo: o quanto se aceita (1 desvio, a priori) que o k real se afaste do físico.
PRIOR_REL_SD = 0.5
MIN_JACKKNIFE = 4  # com menos leave-one-out identificáveis, o jackknife não diz nada: peso 0
FIRST_SERIES_LAP = 8  # a série volta a volta começa aqui (antes disso não há o que estimar)
SENSITIVITY_MULTIPLIERS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
PROGRESS_FRACTIONS = (0.25, 0.5, 0.75)
# Um composto com menos voltas que isto não tem inclinação que se leia (o ajuste devolve, é ruído).
MIN_COMPOUND_LAPS = 30


def physical_coefficient(total_laps: int) -> float:
    """Segundos ganhos por volta ao queimar combustível, pela suposição física."""
    if total_laps < 1:
        raise ValueError(f"total_laps deve ser >= 1, recebi {total_laps}")
    return FUEL_S_PER_KG * FUEL_START_KG / total_laps


def correct_for_fuel(lap_number, lap_time_s, kappa: float):
    """O tempo que a volta teria com o combustível da volta 1. Aceita escalar ou array numpy."""
    return lap_time_s + kappa * (np.asarray(lap_number) - 1)


def dry_clean_laps(rows: Sequence[LapRow]) -> list[LapRow]:
    """Voltas que representam o ritmo E são de pista seca: o que o modelo de pneu consome."""
    return [lap for lap in clean_laps(rows) if lap.compound in DRY_COMPOUNDS]


def laps_until(rows: Sequence[LapRow], lap: int) -> list[LapRow]:
    """Só o que se sabe na volta `lap`: nada depois dela entra (sem vazamento do futuro)."""
    return [row for row in rows if row.lap_number <= lap]


# ---------------------------------------------------------------- estimativa conjunta


@dataclass(frozen=True, slots=True)
class FuelEstimate:
    kappa: float | None  # s/volta; None se não identificável
    se: float | None
    n_laps: int
    n_stints: int
    n_drivers: int
    note: str  # "" se estimou; senão, por que não deu

    @property
    def identified(self) -> bool:
        return self.kappa is not None and self.se is not None


def _require_clean(laps: Sequence[LapRow]) -> None:
    for lap in laps:
        if None in (lap.lap_time_ms, lap.compound, lap.tyre_life, lap.stint):
            raise ValueError(f"volta suja chegou ao estimador: {lap}; passe dry_clean_laps()")


def estimate_fuel_coefficient(laps: Sequence[LapRow]) -> FuelEstimate:
    """Estima `k` junto com a degradação por composto (ver o topo do módulo). `laps` já limpas."""
    _require_clean(laps)
    n_stints = len({(lap.driver, lap.stint) for lap in laps})
    n_drivers = len({lap.driver for lap in laps})

    def unknown(note: str) -> FuelEstimate:
        return FuelEstimate(None, None, len(laps), n_stints, n_drivers, note)

    if len(laps) < MIN_LAPS or n_stints < MIN_STINTS or n_drivers < MIN_DRIVERS:
        return unknown(f"poucos dados ({len(laps)} voltas, {n_stints} stints, {n_drivers} pilotos)")

    drivers = [lap.driver for lap in laps]
    compounds = np.array([lap.compound for lap in laps])
    age = np.array([lap.tyre_life for lap in laps], dtype=float)
    race_lap = np.array([lap.lap_number for lap in laps], dtype=float)
    seconds = np.array([lap.lap_time_ms for lap in laps], dtype=float) / 1000

    driver_cols, _ = one_hot(drivers)
    compound_cols, _ = one_hot(list(compounds), drop_first=True)
    age_cols = np.column_stack([age * (compounds == c) for c in sorted(set(compounds))])
    X = np.column_stack([driver_cols, compound_cols, age_cols, race_lap])
    fit = fit_linear(X, seconds, clusters=drivers)

    last = X.shape[1] - 1
    if not fit.identified[last] or not np.isfinite(fit.se[last]):
        return unknown(
            "a volta da corrida não se separa da idade (stints começam nas mesmas voltas)"
        )
    return FuelEstimate(
        -float(fit.coef[last]), float(fit.se[last]), len(laps), n_stints, n_drivers, ""
    )


# ---------------------------------------------------------------- a escolha


@dataclass(frozen=True, slots=True)
class FuelChoice:
    kappa: float
    source: Literal["estimado", "físico"]
    physical: float
    estimate: FuelEstimate
    reason: str

    @property
    def ratio(self) -> float | None:
        """estimado / físico (None se não houve estimativa)."""
        return None if self.estimate.kappa is None else self.estimate.kappa / self.physical


def choose_fuel_coefficient(estimate: FuelEstimate, physical: float) -> FuelChoice:
    """Aplica o critério do topo do módulo: estimado se identificável, preciso e parecido."""

    def physical_because(reason: str) -> FuelChoice:
        return FuelChoice(physical, "físico", physical, estimate, reason)

    if not estimate.identified:
        return physical_because(estimate.note or "não identificável")
    assert estimate.kappa is not None and estimate.se is not None
    if estimate.se > MAX_REL_SE * physical:
        return physical_because(
            f"impreciso (erro-padrão {estimate.se:.3f} > {MAX_REL_SE:g} x físico {physical:.3f})"
        )
    ratio = estimate.kappa / physical
    if not (1 / MAX_RATIO <= ratio <= MAX_RATIO):
        return physical_because(f"diverge do físico (razão {ratio:.2f}, fora de 0,5 a 2)")
    return FuelChoice(estimate.kappa, "estimado", physical, estimate, "parecido e preciso")


# ---------------------------------------------------------------- encolhimento contínuo


class FuelVariant(StrEnum):
    """As três formas de obter `k` que o backtest compara (etapa 8)."""

    FISICO = "fisico"
    ESTIMADO = "estimado"  # a regra binária: estimado se identificável, preciso e parecido
    ENCOLHIMENTO = "encolhimento"


def jackknife_se(laps: Sequence[LapRow]) -> float | None:
    """Erro-padrão do `k` estimado pelo jackknife por piloto: refaz a estimativa tirando UM piloto
    por vez e mede o quanto ela muda, `sqrt((g-1)/g * soma((k_i - media)^2))`. Diferente do
    erro-padrão por cluster, que com poucos stints cedo na corrida sai pequeno demais, o jackknife
    explode quando a estimativa depende de poucos pilotos. None se menos de MIN_JACKKNIFE
    retiradas se identificam."""
    estimates = []
    for driver in sorted({lap.driver for lap in laps}):
        estimate = estimate_fuel_coefficient([lap for lap in laps if lap.driver != driver])
        if estimate.identified and estimate.kappa is not None:
            estimates.append(estimate.kappa)
    if len(estimates) < MIN_JACKKNIFE:
        return None
    values = np.array(estimates)
    g = len(values)
    return float(np.sqrt((g - 1) / g * np.sum((values - values.mean()) ** 2)))


def pit_stops_seen(rows: Sequence[LapRow]) -> int:
    """Paradas nos boxes já vistas: voltas de saída (a largada pelos boxes, na volta 1, não conta).
    Cada parada é uma comparação de stints, e é ela que identifica o `k`: é a QUANTIDADE DE DADOS
    que cresce de forma gradual, uma parada de cada vez, e nunca diminui com mais voltas."""
    return sum(1 for row in rows if row.pit_out and row.lap_number > 1)


def data_weight(rows: Sequence[LapRow]) -> float:
    """De 0 a 1 conforme as paradas já vistas: peso cheio quando, em média, cada piloto já parou uma
    vez. Cresce monotonamente com os dados até a volta N."""
    drivers = len({row.driver for row in rows})
    return min(1.0, pit_stops_seen(rows) / drivers) if drivers else 0.0


def shrink_coefficient(
    kappa_hat: float | None, se: float | None, physical: float, *, data: float = 1.0
) -> tuple[float, float]:
    """`(k, w)` com `k = w * estimado_limitado + (1 - w) * físico` e `w = data * precisão`.

    * `precisão = t^2 / (t^2 + se^2)`: o quanto se confia no estimado, dado o seu erro-padrão.
    * `data` (0 a 1, ver `data_weight`): a quantidade de dados. É ela que impede o degrau: a
      identificação pode chegar de uma volta para outra (o estimador passa de inexistente a preciso
      numa volta só), e o peso por precisão iria de 0 a 1 junto; ponderado pelas paradas já vistas,
      `w` só sobe conforme elas chegam.

    Função PURA e CONTÍNUA em `(kappa_hat, se, data)`: sem estimativa ou com `se` enorme, `w` tende
    a 0 e `k` ao físico. O estimado é limitado a 1/MAX_RATIO..MAX_RATIO do físico (o mesmo intervalo
    do critério), o que também é contínuo: nenhum valor, por absurdo que seja, move `k` para além
    desse intervalo."""
    if kappa_hat is None or se is None or not np.isfinite(se):
        return physical, 0.0
    prior_var = (PRIOR_REL_SD * physical) ** 2
    weight = data * prior_var / (prior_var + se**2)
    bounded = min(max(kappa_hat, physical / MAX_RATIO), physical * MAX_RATIO)
    return weight * bounded + (1 - weight) * physical, weight


@dataclass(frozen=True, slots=True)
class FuelCoefficient:
    kappa: float
    variant: FuelVariant
    physical: float
    estimate: FuelEstimate
    weight: float  # peso do estimado: 0 = só o físico, 1 = só o estimado (limitado)
    reason: str
    se: float | None = None  # o erro-padrão efetivo do estimado (None se não há estimativa)


def fuel_coefficient(
    rows: Sequence[LapRow], *, upto_lap: int, total_laps: int, variant: FuelVariant
) -> FuelCoefficient:
    """O `k` a usar na volta `upto_lap`, só com o que se sabe até ela.

    SEM VAZAMENTO DO FUTURO por construção: o primeiro passo é cortar as voltas depois de
    `upto_lap`, antes de limpar, estimar ou pesar qualquer coisa. Receber a corrida inteira e
    confiar que o chamador cortou seria a porta para o vazamento."""
    laps = dry_clean_laps(laps_until(rows, upto_lap))
    physical = physical_coefficient(total_laps)
    if variant is FuelVariant.FISICO:
        unused = FuelEstimate(None, None, len(laps), 0, 0, "variante física: não se estima")
        return FuelCoefficient(physical, variant, physical, unused, 0.0, "físico por definição")

    estimate = estimate_fuel_coefficient(laps)
    if variant is FuelVariant.ESTIMADO:
        choice = choose_fuel_coefficient(estimate, physical)
        weight = 1.0 if choice.source == "estimado" else 0.0
        return FuelCoefficient(
            choice.kappa, variant, physical, estimate, weight, choice.reason, estimate.se
        )

    if not estimate.identified:
        return FuelCoefficient(physical, variant, physical, estimate, 0.0, estimate.note)
    assert estimate.se is not None
    jackknife = jackknife_se(laps)
    if jackknife is None:
        reason = "a estimativa depende de poucos pilotos (jackknife sem retiradas suficientes)"
        return FuelCoefficient(physical, variant, physical, estimate, 0.0, reason)
    data = data_weight(laps_until(rows, upto_lap))
    kappa, weight = shrink_coefficient(
        estimate.kappa, max(estimate.se, jackknife), physical, data=data
    )
    reason = f"peso do estimado {weight:.2f} (dados {data:.2f})"
    return FuelCoefficient(
        kappa, variant, physical, estimate, weight, reason, max(estimate.se, jackknife)
    )


def coefficient_sd(coefficient: FuelCoefficient) -> float:
    """O quanto o `k` usado pode estar errado, em s por volta: mistura a incerteza do físico (a
    priori, 50% dele) e a do estimado, ponderadas pelo peso. Sem estimativa (peso 0) é a do físico;
    com o estimado dominando, é o erro-padrão dele. É aproximada (a média de duas fontes
    independentes), e é o que o undercut usa para medir o quanto o resultado depende do `k`."""
    prior_sd = PRIOR_REL_SD * coefficient.physical
    se = coefficient.se if coefficient.se is not None else prior_sd
    return float(np.hypot((1 - coefficient.weight) * prior_sd, coefficient.weight * se))


def coefficient_series(
    rows: Sequence[LapRow],
    *,
    total_laps: int,
    variant: FuelVariant,
    first_lap: int = FIRST_SERIES_LAP,
) -> list[FuelCoefficient]:
    """O `k` da variante em cada volta de `first_lap` ao fim, só com dados até aquela volta."""
    return [
        fuel_coefficient(rows, upto_lap=n, total_laps=total_laps, variant=variant)
        for n in range(first_lap, total_laps + 1)
    ]


def max_step(series: Sequence[FuelCoefficient]) -> float:
    """O maior salto de `k` entre voltas consecutivas (0 se há menos de duas voltas)."""
    return max(
        (abs(b.kappa - a.kappa) for a, b in zip(series, series[1:], strict=False)), default=0.0
    )


def weight_drops(series: Sequence[FuelCoefficient], tolerance: float = 0.01) -> int:
    """Quantas vezes o peso do estimado CAIU de uma volta para a seguinte em mais de `tolerance`
    (oscilações de quarta casa decimal perto de w = 1 não são quedas, são ruído numérico)."""
    return sum(
        1 for a, b in zip(series, series[1:], strict=False) if b.weight < a.weight - tolerance
    )


# ---------------------------------------------------------------- degradação e sensibilidade


@dataclass(frozen=True, slots=True)
class CompoundSlope:
    compound: str
    slope: float | None  # s por volta de idade, já com o combustível corrigido; None = não se lê
    se: float | None
    n_laps: int


def compound_slopes(laps: Sequence[LapRow], kappa: float) -> list[CompoundSlope]:
    """Degradação do composto na corrida: inclinação do tempo corrigido contra a idade, com um
    intercepto por stint (cada jogo de pneus tem o seu nível de ritmo). É o ajuste "da corrida" que
    a etapa seguinte usa como prior; aqui serve à análise de sensibilidade."""
    _require_clean(laps)
    if not laps:
        return []
    age = np.array([lap.tyre_life for lap in laps], dtype=float)
    race_lap = np.array([lap.lap_number for lap in laps], dtype=float)
    seconds = np.array([lap.lap_time_ms for lap in laps], dtype=float) / 1000
    corrected = correct_for_fuel(race_lap, seconds, kappa)
    compounds = np.array([lap.compound for lap in laps])
    drivers = [lap.driver for lap in laps]

    stint_cols, _ = one_hot([(lap.driver, lap.stint) for lap in laps])
    names = sorted(set(compounds))
    age_cols = np.column_stack([age * (compounds == c) for c in names])
    fit = fit_linear(np.column_stack([stint_cols, age_cols]), corrected, clusters=drivers)

    first = stint_cols.shape[1]
    slopes = []
    for i, name in enumerate(names):
        n_laps = int((compounds == name).sum())
        # `lstsq` devolve um número para QUALQUER coluna, inclusive as que o dado não determina
        # (um composto com 1 volta chegou a dar +32 s/volta): só se lê o que é identificável e tem
        # voltas suficientes.
        readable = (
            bool(fit.identified[first + i])
            and np.isfinite(fit.se[first + i])
            and n_laps >= MIN_COMPOUND_LAPS
        )
        slopes.append(
            CompoundSlope(
                name,
                float(fit.coef[first + i]) if readable else None,
                float(fit.se[first + i]) if readable else None,
                n_laps,
            )
        )
    return slopes


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    multiplier: float
    kappa: float
    slopes: dict[str, CompoundSlope]


def sensitivity(
    laps: Sequence[LapRow],
    kappa_physical: float,
    multipliers: Sequence[float] = SENSITIVITY_MULTIPLIERS,
) -> list[SensitivityPoint]:
    """A degradação estimada de cada composto para `k` = múltiplos do físico (de 0, sem correção,
    a 2x). Dentro do stint `volta = idade + constante`, então somar `k*volta` ao tempo soma `k` à
    inclinação da idade EXATAMENTE: cada +0,01 s/volta de `k` vira +0,01 s/volta de degradação, em
    todo composto. Os testes conferem a identidade; os números mostram o tamanho dela nos dados."""
    return [
        SensitivityPoint(
            m,
            m * kappa_physical,
            {s.compound: s for s in compound_slopes(laps, m * kappa_physical)},
        )
        for m in multipliers
    ]


# ---------------------------------------------------------------- uma corrida


@dataclass(slots=True)
class RaceFuelReport:
    label: str
    total_laps: int
    n_laps: int  # voltas limpas e secas
    physical: float
    full: FuelChoice  # a regra binária com a corrida inteira
    progress: dict[float, FuelChoice]  # a regra binária com dados só até cada fração
    sensitivity: list[SensitivityPoint]
    chosen: dict[str, CompoundSlope]  # a degradação com o k que a regra escolheu
    full_shrink: FuelCoefficient  # o encolhimento com a corrida inteira
    shrink_progress: dict[float, FuelCoefficient]  # ... e com dados só até cada fração
    # k volta a volta (cada um só com dados até a sua volta); só com `with_series`, que custa
    # ~20 reajustes por volta (o jackknife).
    rule_series: list[FuelCoefficient] | None = None
    shrink_series: list[FuelCoefficient] | None = None


def analyze_race(
    label: str, rows: Sequence[LapRow], total_laps: int, *, with_series: bool = False
) -> RaceFuelReport:
    laps = dry_clean_laps(rows)
    physical = physical_coefficient(total_laps)
    progress = {
        fraction: choose_fuel_coefficient(
            estimate_fuel_coefficient(laps_until(laps, round(fraction * total_laps))), physical
        )
        for fraction in PROGRESS_FRACTIONS
    }
    full = choose_fuel_coefficient(estimate_fuel_coefficient(laps), physical)

    def shrink(upto: int) -> FuelCoefficient:
        return fuel_coefficient(
            rows, upto_lap=upto, total_laps=total_laps, variant=FuelVariant.ENCOLHIMENTO
        )

    series = {}
    if with_series:
        for variant in (FuelVariant.ESTIMADO, FuelVariant.ENCOLHIMENTO):
            series[variant] = coefficient_series(rows, total_laps=total_laps, variant=variant)
    return RaceFuelReport(
        label=label,
        total_laps=total_laps,
        n_laps=len(laps),
        physical=physical,
        full=full,
        progress=progress,
        sensitivity=sensitivity(laps, physical),
        chosen={s.compound: s for s in compound_slopes(laps, full.kappa)},
        full_shrink=shrink(total_laps),
        shrink_progress={f: shrink(round(f * total_laps)) for f in PROGRESS_FRACTIONS},
        rule_series=series.get(FuelVariant.ESTIMADO),
        shrink_series=series.get(FuelVariant.ENCOLHIMENTO),
    )


def load_reports(
    session, year: int | None = None, *, with_series: bool = False
) -> tuple[list[RaceFuelReport], list[str]]:
    """Relatório das corridas do backtest principal (secas) e, à parte, as que ficaram de fora."""
    main, separate = split_population(session, year)
    reports = [
        analyze_race(
            r.label, load_race_laps(session, r.race_id), r.total_laps, with_series=with_series
        )
        for r in main
    ]
    return reports, [f"{r.label}: {r.reason}" for r in separate]


# ---------------------------------------------------------------- relatório


def _num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _estimate_text(choice: FuelChoice) -> str:
    est = choice.estimate
    return "-" if not est.identified else f"{est.kappa:.3f} ± {est.se:.3f}"


def full_table(reports: Sequence[RaceFuelReport]) -> list[list[str]]:
    rows = [
        [
            "corrida",
            "voltas",
            "voltas úteis",
            "físico (s/volta)",
            "estimado ± erro-padrão",
            "razão",
            "usa",
            "motivo",
        ]
    ]
    for r in reports:
        c = r.full
        rows.append(
            [
                r.label,
                str(r.total_laps),
                str(r.n_laps),
                _num(r.physical),
                _estimate_text(c),
                _num(c.ratio, 2),
                c.source,
                c.reason,
            ]
        )
    return rows


def progress_table(reports: Sequence[RaceFuelReport]) -> list[list[str]]:
    header = ["corrida"]
    for f in PROGRESS_FRACTIONS:
        header += [f"regra até {pct(f)}", f"encolhimento até {pct(f)} (k/físico, peso)"]
    rows = [header]
    for r in reports:
        cells = []
        for f in PROGRESS_FRACTIONS:
            rule, shrunk = r.progress[f], r.shrink_progress[f]
            cells += [
                f"{rule.source} ({_num(rule.ratio, 2)}x)"
                if rule.ratio is not None
                else "físico (-)",
                f"{shrunk.kappa / r.physical:.2f}x (w {shrunk.weight:.2f})",
            ]
        rows.append([r.label, *cells])
    return rows


def smoothness_table(reports: Sequence[RaceFuelReport]) -> list[list[str]]:
    header = [
        "corrida",
        "maior passo, regra (s/volta)",
        "maior passo, encolhimento",
        "voltas em que o peso caiu",
        "k final / físico, regra",
        "k final / físico, encolhimento",
    ]
    rows = [header]
    for r in reports:
        if r.rule_series is None or r.shrink_series is None:
            continue
        rows.append(
            [
                r.label,
                _num(max_step(r.rule_series)),
                _num(max_step(r.shrink_series)),
                f"{weight_drops(r.shrink_series)} de {len(r.shrink_series) - 1}",
                f"{r.rule_series[-1].kappa / r.physical:.2f}x",
                f"{r.shrink_series[-1].kappa / r.physical:.2f}x",
            ]
        )
    return rows


def sensitivity_table(reports: Sequence[RaceFuelReport]) -> list[list[str]]:
    header = ["corrida", "composto", "voltas"] + [
        f"{m:g}x" + (" (físico)" if m == 1.0 else " (sem correção)" if m == 0.0 else "")
        for m in SENSITIVITY_MULTIPLIERS
    ]
    rows = [header]
    for r in reports:
        compounds = sorted({c for p in r.sensitivity for c in p.slopes})
        for compound in compounds:
            points = [p.slopes.get(compound) for p in r.sensitivity]
            n_laps = next((s.n_laps for s in points if s), 0)
            rows.append(
                [r.label, compound, str(n_laps)]
                + ["-" if s is None or s.slope is None else f"{s.slope:+.3f}" for s in points]
            )
    return rows


def _readable_slopes(reports: Sequence[RaceFuelReport], multiplier: float) -> list[float]:
    values = []
    for r in reports:
        for point in r.sensitivity:
            if point.multiplier == multiplier:
                values += [x.slope for x in point.slopes.values() if x.slope is not None]
    return values


def _sensitivity_summary(reports: Sequence[RaceFuelReport]) -> list[str]:
    """O que a sensibilidade mostra, em números calculados dos próprios dados."""
    deviation = 0.0  # |variação da degradação - variação de k|, no pior caso
    for r in reports:
        base = {x.compound: x.slope for x in r.sensitivity[0].slopes.values()}
        for point in r.sensitivity[1:]:
            for x in point.slopes.values():
                if x.slope is not None and base.get(x.compound) is not None:
                    deviation = max(deviation, abs((x.slope - base[x.compound]) - point.kappa))
    chosen = [x.slope for r in reports for x in r.chosen.values() if x.slope is not None]
    raw, physical = _readable_slopes(reports, 0.0), _readable_slopes(reports, 1.0)
    half, one_and_half = _readable_slopes(reports, 0.5), _readable_slopes(reports, 1.5)
    n = len(physical)
    if not n:
        return []

    def negative(values: list[float]) -> str:
        return f"{sum(v < 0 for v in values)} de {len(values)}"

    return [
        f"- Sensibilidade: somar `k * volta` ao tempo soma `k` à inclinação da idade. Desvio "
        f"máximo entre a variação medida da degradação e a variação de `k`, em todas as linhas: "
        f"{deviation:.1e} s/volta (a identidade vale; a incerteza do combustível passa 1 para 1).",
        f"- Degradação mediana (linhas com {MIN_COMPOUND_LAPS}+ voltas): sem correção "
        f"{median(raw):+.3f}; `k` a 0,5x {median(half):+.3f}; `k` físico {median(physical):+.3f}; "
        f"`k` a 1,5x {median(one_and_half):+.3f} s por volta de idade.",
        f"- Linhas com degradação NEGATIVA (o pneu 'melhora' com a idade, o que não é desgaste): "
        f"sem correção {negative(raw)}; `k` físico {negative(physical)}; `k` escolhido "
        f"{negative(chosen)}.",
    ]


def _summary(reports: Sequence[RaceFuelReport], skipped: Sequence[str]) -> list[str]:
    if not reports:
        return ["- Nenhuma corrida seca para analisar."]
    full = [r.full for r in reports]
    identified = [c for c in full if c.estimate.identified]
    ratios = [c.ratio for c in identified if c.ratio is not None]
    used = sum(c.source == "estimado" for c in full)
    lines = [
        f"- {len(reports)} corridas secas analisadas (reportadas à parte, fora da nota principal: "
        f"{'; '.join(skipped) if skipped else 'nenhuma'}).",
        f"- Físico: de {min(r.physical for r in reports):.3f} a "
        f"{max(r.physical for r in reports):.3f} s/volta, conforme o número de voltas.",
        f"- Com a corrida inteira, o coeficiente é identificável em {len(identified)} de "
        f"{len(reports)} corridas; o critério escolhe o estimado em {used} e o físico em "
        f"{len(reports) - used}.",
    ]
    if ratios:
        lines.append(
            f"- Razão estimado/físico entre as identificáveis: mediana {median(ratios):.2f}, de "
            f"{min(ratios):.2f} a {max(ratios):.2f}."
        )
        lines.append(
            "- Erro-padrão mediano do estimado: "
            f"{median(c.estimate.se for c in identified if c.estimate.se is not None):.3f} s/volta "
            f"(físico mediano: {median(r.physical for r in reports):.3f})."
        )
    reasons: dict[str, int] = {}
    for c in full:
        if c.source == "físico":
            key = c.reason.split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        lines.append(
            "- Por que o físico, quando foi o escolhido: "
            + "; ".join(f"{k} em {n}" for k, n in sorted(reasons.items()))
            + "."
        )
    lines += _sensitivity_summary(reports)
    for fraction in PROGRESS_FRACTIONS:
        n = sum(r.progress[fraction].source == "estimado" for r in reports)
        weights = [r.shrink_progress[fraction].weight for r in reports]
        lines.append(
            f"- Com dados só até {pct(fraction)} da corrida: a regra usa o estimado em {n} de "
            f"{len(reports)}; no encolhimento o peso do estimado tem mediana {median(weights):.2f}."
        )
    lines += _smoothness_summary(reports)
    return lines


def _smoothness_summary(reports: Sequence[RaceFuelReport]) -> list[str]:
    with_series = [r for r in reports if r.rule_series and r.shrink_series]
    if not with_series:
        return []
    rule = [max_step(r.rule_series) for r in with_series]
    shrunk = [max_step(r.shrink_series) for r in with_series]
    drops = sum(weight_drops(r.shrink_series) for r in with_series)
    laps = sum(len(r.shrink_series) - 1 for r in with_series)
    margin = 0.001  # diferenças menores que 1 milésimo de s/volta contam como empate
    better = sum(sh < ru - margin for ru, sh in zip(rule, shrunk, strict=True))
    worse = [
        r.label for r, ru, sh in zip(with_series, rule, shrunk, strict=True) if sh > ru + margin
    ]
    return [
        "- Maior passo de `k` entre voltas consecutivas, por corrida (s/volta): regra binária com "
        f"mediana {median(rule):.3f} e máximo {max(rule):.3f}; encolhimento com mediana "
        f"{median(shrunk):.3f} e máximo {max(shrunk):.3f}. O degrau da troca de fonte some, mas o "
        "`k` estimado ainda oscila a cada volta que chega: o encolhimento não o torna constante.",
        f"- O encolhimento tem o maior passo menor que o da regra em {better} de "
        f"{len(with_series)} "
        f"corridas, empata (diferença < {margin}) em {len(with_series) - better - len(worse)} e é "
        f"PIOR em {len(worse)}" + (f" ({'; '.join(worse)})" if worse else "") + ". Pior não é "
        "necessariamente defeito: onde o estimado diverge muito do físico, a regra nunca sai do "
        "físico (passo 0) e o encolhimento caminha até o teto de 2x.",
        f"- O peso do estimado caiu de uma volta para a seguinte em {drops} de {laps} voltas "
        f"({drops / laps:.0%}): `w` sobe em tendência, não volta a volta.",
    ]


MARKDOWN_METHOD = """\
## Método

- **Correção**: `tempo_corrigido = tempo + k * (volta - 1)`, o tempo que a volta teria com o
  combustível da volta 1. `k` = segundos que o carro ganha por volta ao queimar combustível.
- **Físico**: `k = 0,03 s/kg * 110 kg / voltas`. A constante de 0,03 s/kg é uma **suposição** de
  engenharia (0,025 a 0,035 conforme o circuito), não medida com estes dados.
- **Estimado**: `tempo = piloto + composto + degradação_do_composto * idade - k * volta`, com
  mínimos quadrados robustos (Huber) e erro-padrão por cluster de piloto, nas voltas limpas e secas.
  O intercepto é **por piloto e não por stint**: com intercepto por stint, `k` não se separa da
  degradação (dentro do stint `volta = idade + constante`). O `k` estimado contém tudo o que melhora
  o ritmo com a volta da corrida, não só o combustível: a pista que borracha, a temperatura.
- **Critério** (fixado antes de olhar os resultados): o estimado só é usado se for identificável,
  preciso (erro-padrão de no máximo metade do físico) e parecido (razão de 0,5 a 2 em relação ao
  físico). Senão usa o físico e a coluna `motivo` diz por quê.
- **Encolhimento contínuo**: `k = w * estimado_limitado + (1 - w) * físico`, com
  `w = t^2 / (t^2 + se^2)`, `t` = 50% do físico (a priori) e `se` = o maior entre o erro-padrão por
  cluster e o do jackknife por piloto (o quanto o estimado muda quando se tira um piloto). Sem
  estimativa, `w = 0`. O estimado é limitado a 0,5x..2x do físico (os mesmos limites do critério,
  que não foi apertado). A primeira versão usava só o erro-padrão por cluster e ficou **menos
  suave** que a regra (cedo na corrida o estimador devolve valores absurdos com erro-padrão
  pequeno); o jackknife e o limite corrigem isso. Só usa voltas até a volta N.
- **Variantes do backtest**: `fisico`, `estimado` (a regra binária) e `encolhimento`. Os números do
  backtest decidem o padrão.
- **Sensibilidade**: a degradação de cada composto (inclinação do tempo corrigido contra a idade,
  intercepto por stint) para `k` de 0 a 2 vezes o físico. Dentro do stint
  `volta = idade + constante`, então **cada +0,01 s/volta de `k` soma exatamente +0,01 s/volta à
  degradação estimada, em todos os compostos**: a incerteza do combustível passa 1 para 1 para a da
  degradação, e a ordem entre compostos não muda.
"""


def build_report(
    reports: Sequence[RaceFuelReport],
    skipped: Sequence[str],
    *,
    markdown: bool,
    generated_on: date | None = None,
) -> str:
    summary = "\n".join(_summary(reports, skipped))
    if not markdown:
        return (
            f"{summary}\n\n"
            f"ESTIMADO x FÍSICO (corrida inteira)\n{render_text(full_table(reports))}\n\n"
            f"REGRA E ENCOLHIMENTO COM DADOS ATÉ CADA FRAÇÃO DA CORRIDA\n"
            f"{render_text(progress_table(reports))}\n\n"
            + (
                f"SUAVIDADE VOLTA A VOLTA\n{render_text(smoothness_table(reports))}\n\n"
                if len(smoothness_table(reports)) > 1
                else ""
            )
            + f"SENSIBILIDADE DA DEGRADAÇÃO (s por volta de idade) AO COEFICIENTE\n"
            f"{render_text(sensitivity_table(reports))}\n"
        )
    day = (generated_on or date.today()).isoformat()
    return (
        "# Combustível: estimado x físico e sensibilidade da degradação\n\n"
        "Gerado por `python -m strategy.fuel --markdown docs/combustivel.md` em "
        f"{day}. A regra está em `strategy/fuel.py`.\n\n"
        f"## Resumo\n\n{summary}\n\n{MARKDOWN_METHOD}\n"
        "## Estimado x físico, com a corrida inteira\n\n"
        f"{render_markdown(full_table(reports))}\n\n"
        "## Regra e encolhimento com dados só até cada fração da corrida (sem o futuro)\n\n"
        f"{render_markdown(progress_table(reports))}\n\n"
        + (
            "## Suavidade volta a volta: o maior salto de `k` entre voltas consecutivas\n\n"
            f"{render_markdown(smoothness_table(reports))}\n\n"
            if len(smoothness_table(reports)) > 1
            else ""
        )
        + "## Sensibilidade: degradação (s por volta de idade) para `k` de 0 a 2x o físico\n\n"
        f"{render_markdown(sensitivity_table(reports))}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.fuel",
        description="Compara o coeficiente de combustível estimado com o físico e mostra a "
        "sensibilidade da degradação a ele.",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="grava o relatório neste arquivo .md"
    )
    parser.add_argument(
        "--no-series",
        action="store_true",
        help="pula a suavidade volta a volta (custa ~20 reajustes por volta; minutos)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        reports, skipped = load_reports(session, args.year, with_series=not args.no_series)
    if not reports:
        print(
            "Nenhuma corrida seca no banco. Ingira com: python -m ingestion --year 2024 --round 10"
        )
        return 1
    print(build_report(reports, skipped, markdown=False))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_report(reports, skipped, markdown=True), encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
