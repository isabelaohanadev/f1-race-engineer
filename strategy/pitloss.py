"""Perda no pit: quanto uma parada custa em segundos, medida na PRÓPRIA corrida.

`perda = (volta de entrada + volta de saída) - 2 x ritmo normal` do piloto naquele momento. O ritmo
normal é a média entre a mediana das voltas limpas logo ANTES da entrada e a das logo DEPOIS da
saída
(a de antes é com o pneu velho, a de depois com o novo: a média é o ritmo "no meio"). Os tempos
entram
corrigidos do combustível, como em todo o resto, e só com voltas limpas (status exatamente "1").

PARADAS EM BANDEIRA VERDE, SEPARADAS DAS DEMAIS. Sob safety car ou VSC o pelotão anda devagar, então
parar custa bem menos; misturar as duas puxaria a mediana para baixo e o undercut pareceria mais
barato do que é. Uma parada é "verde" se a volta de entrada E a de saída têm status exatamente "1".

DUAS INCERTEZAS, de novo (como na degradação):
* `sd`: a dispersão ENTRE paradas (equipe mais rápida ou mais lenta, tráfego na saída). É a de UMA
  parada futura. No undercut, atacante e rival pagam paradas independentes, e a incerteza da
  MEDIANA é a mesma para os dois (cancela na diferença), então só `sd` importa para a margem.
* `se`: a incerteza da mediana, por ter poucas paradas.

POUCAS PARADAS: encolhe para um prior (a mediana das outras corridas, `strategy.priors`), com o
mesmo
peso normal-normal da degradação. Nunca com uma corrida POSTERIOR do mesmo circuito (ver priors.py).
Sem prior e sem paradas, não há estimativa.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

import numpy as np

from strategy.cleaning import clean_laps
from strategy.fuel import correct_for_fuel, laps_until
from strategy.laps import LapRow

REFERENCE_WINDOW = 3  # voltas limpas de cada lado usadas como "ritmo normal"
MIN_REFERENCE_LAPS = 2  # de cada lado; com menos, a parada não é medida
MIN_STOPS = 2  # paradas verdes mínimas para ter uma estimativa própria da corrida
SD_FLOOR = 0.5  # s: a dispersão entre paradas nunca é tratada como menor que isto
MAD_TO_SD = 1.4826
MEDIAN_SE_FACTOR = 1.2533  # sqrt(pi/2): erro-padrão da mediana relativo ao da média, normal


@dataclass(frozen=True, slots=True)
class PitStop:
    driver: str
    in_lap: int
    out_lap: int
    loss: float  # s, contra o ritmo normal ao redor
    green: bool  # volta de entrada E de saída com status exatamente "1"


def find_pit_stops(rows: Sequence[LapRow], *, kappa: float, upto_lap: int) -> list[PitStop]:
    """Todas as paradas MEDÍVEIS até `upto_lap`: precisam das duas voltas com tempo e de ao menos
    MIN_REFERENCE_LAPS voltas limpas de cada lado (portanto, uma parada só entra quando o ritmo
    DEPOIS dela já foi observado; nada do futuro)."""
    visible = laps_until(rows, upto_lap)
    by_key = {(lap.driver, lap.lap_number): lap for lap in visible}
    corrected = {
        (lap.driver, lap.lap_number): float(
            correct_for_fuel(lap.lap_number, lap.lap_time_ms / 1000, kappa)
        )
        for lap in clean_laps(visible)
    }

    def corrected_time(lap: LapRow) -> float:
        return float(correct_for_fuel(lap.lap_number, lap.lap_time_ms / 1000, kappa))

    stops = []
    for (driver, lap_number), lap in sorted(by_key.items()):
        out = by_key.get((driver, lap_number + 1))
        if not lap.pit_in or out is None or not out.pit_out:
            continue
        if lap.lap_time_ms is None or out.lap_time_ms is None:
            continue
        before = [
            corrected[(driver, n)]
            for n in range(lap_number - REFERENCE_WINDOW, lap_number)
            if (driver, n) in corrected
        ]
        after = [
            corrected[(driver, n)]
            for n in range(lap_number + 2, lap_number + 2 + REFERENCE_WINDOW)
            if (driver, n) in corrected
        ]
        if len(before) < MIN_REFERENCE_LAPS or len(after) < MIN_REFERENCE_LAPS:
            continue
        reference = (median(before) + median(after)) / 2
        loss = (corrected_time(lap) - reference) + (corrected_time(out) - reference)
        green = lap.track_status == "1" and out.track_status == "1"
        stops.append(PitStop(driver, lap_number, lap_number + 1, loss, green))
    return stops


@dataclass(frozen=True, slots=True)
class PitLossPrior:
    """O que se sabia ANTES desta corrida sobre a perda no pit."""

    mean: float  # s, a mediana das outras corridas
    between_sd: float  # s, o quanto a perda mediana varia entre corridas
    stop_sd: float  # s, a dispersão típica entre paradas dentro de uma corrida


@dataclass(frozen=True, slots=True)
class PitLoss:
    median: float  # s, a perda usada (depois do encolhimento)
    sd: float  # s, dispersão ENTRE paradas: a de uma parada futura
    se: float  # s, incerteza da mediana
    raw_median: float | None  # s, só desta corrida (None se não há paradas verdes)
    n_green: int
    n_other: int  # paradas sob SC/VSC/vermelha ou amarela (não entram na estimativa)
    other_median: float | None  # só informativo: o que custam as paradas fora da bandeira verde
    weight: float  # peso dos dados da corrida (1 = só eles, 0 = só o prior)
    source: Literal["corrida", "encolhido", "prior"]


def estimate_pit_loss(
    rows: Sequence[LapRow],
    *,
    upto_lap: int,
    kappa: float,
    prior: PitLossPrior | None = None,
) -> PitLoss | None:
    """A perda no pit na volta `upto_lap`, só com as paradas já observadas e concluídas até ela."""
    stops = find_pit_stops(rows, kappa=kappa, upto_lap=upto_lap)
    green = np.array([s.loss for s in stops if s.green])
    other = [s.loss for s in stops if not s.green]
    other_median = float(np.median(other)) if other else None
    n_green = len(green)

    if n_green < MIN_STOPS:
        if prior is None:
            return None
        return PitLoss(
            median=prior.mean,
            sd=prior.stop_sd,
            se=prior.between_sd,
            raw_median=float(np.median(green)) if n_green else None,
            n_green=n_green,
            n_other=len(other),
            other_median=other_median,
            weight=0.0,
            source="prior",
        )

    raw = float(np.median(green))
    sd = max(MAD_TO_SD * float(np.median(np.abs(green - raw))), SD_FLOOR)
    se = MEDIAN_SE_FACTOR * sd / np.sqrt(n_green)
    if prior is None:
        return PitLoss(raw, sd, se, raw, n_green, len(other), other_median, 1.0, "corrida")
    weight = prior.between_sd**2 / (prior.between_sd**2 + se**2)
    return PitLoss(
        median=prior.mean + weight * (raw - prior.mean),
        sd=sd,
        se=float(np.sqrt(weight * se**2)),
        raw_median=raw,
        n_green=n_green,
        n_other=len(other),
        other_median=other_median,
        weight=float(weight),
        source="encolhido",
    )
