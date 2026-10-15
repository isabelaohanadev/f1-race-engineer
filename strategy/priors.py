"""Constantes que saem dos DADOS, calculadas SEM a corrida que está sendo avaliada.

Quatro números do modelo vêm de olhar as corridas: a variação entre pilotos/stints (`tau`, quando a
corrida ainda não tem dados para estimá-la), o piso da diferença entre compostos, e a perda no pit
(mediana, variação entre corridas e dispersão entre paradas). Calculá-los com TODAS as corridas e
depois testar o modelo numa delas é vazamento: a corrida de teste já ajudou a escolher a constante.

`leave_one_out` (a regra do backtest) recalcula cada um sem a corrida avaliada. `DEFAULT_PRIORS` são
os valores com as 21 corridas de 2024 TODAS dentro: servem a quem usa o modelo ao vivo, sem nada a
avaliar, e NUNCA a um backtest.

O CIRCUITO. A perda no pit é muito diferente entre pistas (a mediana por corrida vai de 18,8 s em
Spa
a 29,8 s em Singapura), então a melhor reserva é a mesma pista em edições anteriores:
`circuit_history`
devolve só edições ANTERIORES (nunca posteriores) do mesmo circuito. Com o banco só de 2024 não há
edição anterior de nada, então hoje o caminho do circuito não tem o que devolver e a reserva é a
mediana das demais corridas.

DUAS REGRAS DE EXCLUSÃO, para o backtest escolher:
* `earlier_only=False` (padrão): todas as outras corridas, inclusive as POSTERIORES do calendário.
  Para as constantes gerais é o leave-one-race-out pedido, e não vaza a corrida avaliada; mas usa
  informação que, ao vivo, ainda não existia.
* `earlier_only=True`: só as corridas ANTERIORES no calendário: o que de fato se saberia ao vivo.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from strategy.pitloss import PitLossPrior

MIN_RACES = 3  # com menos corridas de referência, a constante não é estimada: vale o padrão
_MAD_TO_SD = 1.4826
PIT_LOSS_BETWEEN_FLOOR = 0.5  # s
OFFSET_SD_FLOOR = 0.1  # s
MIN_OFFSETS = 6  # com menos diferenças de referência, vale o padrão


@dataclass(frozen=True, slots=True)
class Priors:
    tau: float  # s por volta de idade: variação entre pilotos/stints
    between_compounds: float  # s por volta de idade: piso da diferença entre compostos
    pit_loss: PitLossPrior
    # O quanto a diferença de ritmo entre compostos (medida dentro do piloto) varia ENTRE corridas,
    # em s: o viés por corrida que o erro-padrão de cada uma não vê. Entra na incerteza do undercut.
    offset_sd: float
    source: str  # de onde vêm, para ninguém usar as erradas sem saber


# Medidas nas 21 corridas secas de 2024 com todas dentro (docs/perda-no-pit.md).
DEFAULT_PRIORS = Priors(
    tau=0.022,
    between_compounds=0.017,
    pit_loss=PitLossPrior(mean=22.5, between_sd=2.1, stop_sd=1.4),
    offset_sd=0.46,
    source="padrão: 21 corridas de 2024, INCLUSIVE a avaliada (não use em backtest)",
)


@dataclass(frozen=True, slots=True)
class RaceStats:
    """O que uma corrida, com a corrida inteira, diz sobre cada constante."""

    label: str
    circuit: str | None
    year: int
    round: int
    tau: float | None  # None se a corrida não tinha dados para estimar
    between_compounds: float | None  # None se menos de 2 compostos com dados
    pit_loss: float | None  # mediana das paradas verdes (None se não houve)
    pit_loss_sd: float | None  # dispersão entre paradas
    n_green_stops: int
    # As perdas de cada parada (verdes e as demais), só para o relatório comparar uma com a outra.
    green_losses: tuple[float, ...] = ()
    other_losses: tuple[float, ...] = ()
    # As diferenças de ritmo entre compostos (do mais rápido ao mais lento) medidas dentro do
    # piloto.
    offsets: tuple[float, ...] = ()


def _robust_sd(values: Sequence[float], floor: float) -> float:
    if len(values) < 2:
        return floor
    centre = median(values)
    return max(_MAD_TO_SD * median(abs(v - centre) for v in values), floor)


def _median_or(values: Sequence[float | None], default: float) -> float:
    known = [v for v in values if v is not None]
    return float(median(known)) if len(known) >= MIN_RACES else default


def circuit_history(stats: Sequence[RaceStats], label: str) -> list[RaceStats]:
    """Edições ANTERIORES (e só elas) do mesmo circuito da corrida `label`, com perda no pit medida.
    Uma edição posterior nunca entra: ao vivo ela ainda não aconteceu."""
    target = next((s for s in stats if s.label == label), None)
    if target is None or target.circuit is None:
        return []
    return [
        s
        for s in stats
        if s.label != label
        and s.circuit == target.circuit
        and (s.year, s.round) < (target.year, target.round)
        and s.pit_loss is not None
    ]


def leave_one_out(
    stats: Sequence[RaceStats], exclude: str, *, earlier_only: bool = False
) -> Priors:
    """As constantes sem a corrida `exclude` (e, com `earlier_only`, só com as anteriores a ela)."""
    target = next((s for s in stats if s.label == exclude), None)
    if earlier_only and target is None:
        raise ValueError(
            f"corrida {exclude!r} não está nas estatísticas: earlier_only não se aplica"
        )
    others = [s for s in stats if s.label != exclude]
    if earlier_only and target is not None:
        others = [s for s in others if (s.year, s.round) < (target.year, target.round)]

    default = DEFAULT_PRIORS
    medians = [s.pit_loss for s in others if s.pit_loss is not None]
    stop_sds = [s.pit_loss_sd for s in others if s.pit_loss_sd is not None]
    history = circuit_history(stats, exclude)
    if history:
        centre = float(median(h.pit_loss for h in history if h.pit_loss is not None))
    else:
        centre = float(median(medians)) if len(medians) >= MIN_RACES else default.pit_loss.mean
    prior = PitLossPrior(
        mean=centre,
        between_sd=(
            _robust_sd(medians, PIT_LOSS_BETWEEN_FLOOR)
            if len(medians) >= MIN_RACES
            else default.pit_loss.between_sd
        ),
        stop_sd=_median_or(stop_sds, default.pit_loss.stop_sd),
    )
    pooled_offsets = [x for s in others for x in s.offsets]
    offset_sd = (
        _robust_sd(pooled_offsets, OFFSET_SD_FLOOR)
        if len(pooled_offsets) >= MIN_OFFSETS
        else default.offset_sd
    )
    n = len(others)
    where = "só anteriores" if earlier_only else "todas as outras"
    circuit = f", centro do circuito ({len(history)} edições anteriores)" if history else ""
    return Priors(
        tau=_median_or([s.tau for s in others], default.tau),
        between_compounds=_median_or(
            [s.between_compounds for s in others], default.between_compounds
        ),
        pit_loss=prior,
        offset_sd=offset_sd,
        source=f"sem {exclude}: {n} corridas ({where}){circuit}",
    )
