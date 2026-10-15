"""O que cada corrida diz sobre as constantes, e as constantes SEM ela: `python -m
strategy.race_stats`.

Para cada corrida do backtest principal: a variação entre pilotos/stints (`tau`), a diferença típica
entre compostos e a perda no pit das paradas em bandeira verde, todas com a corrida inteira. O
relatório mostra, ao lado, as constantes que o backtest usará para ELA (`leave_one_out`): as das
outras 20, nunca a dela.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from statistics import median

import numpy as np

from db.session import get_session_factory
from strategy.fuel import FuelVariant
from strategy.laps import LapRow, load_race_laps
from strategy.pace import MIN_PAIRS, fit_pace
from strategy.pitloss import MIN_STOPS, find_pit_stops
from strategy.population import RaceRef, split_population
from strategy.priors import (
    DEFAULT_PRIORS,
    PIT_LOSS_BETWEEN_FLOOR,
    RaceStats,
    circuit_history,
    leave_one_out,
)
from strategy.tables import render_markdown, render_text

_MAD_TO_SD = 1.4826


MIN_UNITS_FOR_COMPOUND = 3  # um composto com menos unidades não entra na diferença entre compostos


def _between_compounds(model) -> float | None:
    """O desvio entre as médias dos compostos DESTA corrida, só com os que têm 3+ unidades: com 1 ou
    2 a média do composto é pouco mais que o dado de um piloto. É a mesma definição que originou o
    piso do último recurso (`BETWEEN_COMPOUNDS_FLOOR`)."""
    means = [p.mean for p in model.compounds.values() if p.n_units >= MIN_UNITS_FOR_COMPOUND]
    return float(np.std(means, ddof=1)) if len(means) >= 2 else None


_FASTER_TO_SLOWER = (("SOFT", "MEDIUM"), ("MEDIUM", "HARD"), ("SOFT", "HARD"))


def _pair_offsets(pace) -> tuple[float, ...]:
    """As diferenças de ritmo entre compostos desta corrida, do mais rápido ao mais lento, só as
    medidas DENTRO do piloto em pelo menos MIN_PAIRS pilotos (as das médias não entram)."""
    found = (pace.offset(a, b) for a, b in _FASTER_TO_SLOWER)
    return tuple(
        o.delta for o in found if o is not None and o.source == "pares" and o.n_pairs >= MIN_PAIRS
    )


def compute_race_stats(ref: RaceRef, rows: Sequence[LapRow]) -> RaceStats:
    """As constantes que esta corrida, vista inteira, daria. Usa o `k` da variante encolhimento."""
    pace = fit_pace(
        rows, upto_lap=ref.total_laps, total_laps=ref.total_laps, variant=FuelVariant.ENCOLHIMENTO
    )
    model = pace.degradation
    stops = find_pit_stops(rows, kappa=model.fuel.kappa, upto_lap=ref.total_laps)
    green = np.array([s.loss for s in stops if s.green])
    other = tuple(s.loss for s in stops if not s.green)
    enough = len(green) >= MIN_STOPS
    return RaceStats(
        label=ref.label,
        circuit=ref.circuit,
        year=ref.year,
        round=ref.round,
        tau=model.tau if model.tau_source == "estimado" else None,
        between_compounds=_between_compounds(model),
        pit_loss=float(np.median(green)) if enough else None,
        pit_loss_sd=(
            max(_MAD_TO_SD * float(np.median(np.abs(green - np.median(green)))), 0.5)
            if enough
            else None
        ),
        n_green_stops=len(green),
        green_losses=tuple(float(x) for x in green),
        other_losses=other,
        offsets=_pair_offsets(pace),
    )


def load_race_stats(session, year: int | None = None) -> list[RaceStats]:
    main, _ = split_population(session, year)
    return [compute_race_stats(ref, load_race_laps(session, ref.race_id)) for ref in main]


# ---------------------------------------------------------------- relatório


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def stats_table(stats: Sequence[RaceStats]) -> list[list[str]]:
    rows = [
        [
            "corrida",
            "circuito",
            "paradas verdes",
            "perda no pit (mediana, s)",
            "dispersão entre paradas (s)",
            "τ",
            "diferença entre compostos",
            "reserva da perda SEM esta corrida (s)",
            "edições anteriores do circuito",
        ]
    ]
    for s in stats:
        loo = leave_one_out(stats, s.label)
        rows.append(
            [
                s.label,
                s.circuit or "-",
                str(s.n_green_stops),
                _fmt(s.pit_loss, 1),
                _fmt(s.pit_loss_sd, 1),
                _fmt(s.tau),
                _fmt(s.between_compounds),
                _fmt(loo.pit_loss.mean, 1),
                str(len(circuit_history(stats, s.label))),
            ]
        )
    return rows


def _summary(stats: Sequence[RaceStats]) -> list[str]:
    if not stats:
        return ["- Nenhuma corrida."]
    losses = [s.pit_loss for s in stats if s.pit_loss is not None]
    sds = [s.pit_loss_sd for s in stats if s.pit_loss_sd is not None]
    taus = [s.tau for s in stats if s.tau is not None]
    betweens = [s.between_compounds for s in stats if s.between_compounds is not None]
    centre = median(losses)
    between_sd = max(_MAD_TO_SD * median(abs(v - centre) for v in losses), PIT_LOSS_BETWEEN_FLOOR)
    few = [s.label for s in stats if s.n_green_stops < 10]
    green_all = [x for s in stats for x in s.green_losses]
    other_all = [x for s in stats for x in s.other_losses]
    offsets = [x for st in stats for x in st.offsets]
    offset_line = (
        f"- Diferença de ritmo entre compostos medida dentro do piloto ({len(offsets)} "
        "diferenças, uma "
        f"por par de compostos e corrida): mediana {median(offsets):+.2f} s, desvio ENTRE corridas "
        f"{_MAD_TO_SD * median(abs(v - median(offsets)) for v in offsets):.2f} s, bem maior que o "
        "erro-padrão de cada uma (0,05 a 0,17 s): é o viés por corrida que o erro-padrão não vê."
        if len(offsets) >= 2
        else "- Sem diferenças de ritmo entre compostos medidas dentro do piloto."
    )
    other_line = (
        f"- Paradas FORA da bandeira verde (safety car, VSC, vermelha, amarela): {len(other_all)}, "
        f"mediana {median(other_all):.1f} s contra {median(green_all):.1f} s das verdes. "
        "Não saíram "
        "mais baratas; ver a nota abaixo sobre por que isso não responde à pergunta."
        if other_all and green_all
        else "- Nenhuma parada fora da bandeira verde medida."
    )
    d = DEFAULT_PRIORS
    return [
        f"- {len(stats)} corridas; {sum(s.n_green_stops for s in stats)} paradas em bandeira "
        "verde.",
        f"- Perda no pit (mediana por corrida): mediana {centre:.1f} s, de {min(losses):.1f} a "
        f"{max(losses):.1f}; a variação ENTRE corridas (desvio robusto) é {between_sd:.1f} s e a "
        f"dispersão entre paradas dentro de uma corrida, em mediana, {median(sds):.1f} s.",
        other_line,
        offset_line,
        f"- Com menos de 10 paradas verdes: {', '.join(few) if few else 'nenhuma corrida'}. "
        "É nelas "
        "que o prior pesa.",
        f"- τ mediano {median(taus):.3f}; diferença entre compostos mediana "
        f"{median(betweens):.3f}.",
        f"- Valores de `DEFAULT_PRIORS` (todas as corridas dentro): perda {d.pit_loss.mean:.1f} s, "
        f"variação entre corridas {d.pit_loss.between_sd:.1f} s, dispersão entre paradas "
        f"{d.pit_loss.stop_sd:.1f} s, τ {d.tau}, piso entre compostos {d.between_compounds}, "
        f"espalhamento entre corridas da diferença de compostos {d.offset_sd}.",
    ]


MARKDOWN_NOTES = """\
## Como ler

- **Paradas verdes**: a volta de entrada e a de saída têm status exatamente `1`. A perda é
  `(entrada + saída) - 2 x ritmo normal`, com o ritmo normal vindo das voltas limpas antes e depois.
- **Reserva SEM esta corrida**: a mediana das perdas das **outras** corridas (leave-one-race-out).
  Com `earlier_only` só entram as anteriores no calendário, o que se saberia ao vivo.
- **Edições anteriores do circuito**: com o banco só de 2024 não há nenhuma, então o caminho "mesma
  pista em edições anteriores" não tem o que devolver e vale a mediana das outras corridas.
- **Paradas fora da bandeira verde** (safety car, VSC, vermelha, amarela): ficam fora da estimativa.
  Medidas contra o ritmo normal do próprio piloto elas **não** saem mais baratas (ver o resumo), mas
  a amostra é pequena e enviesada, porque só entram paradas com voltas verdes ao redor para servir
  de referência; e "custa menos" sob safety car é uma afirmação **relativa ao pelotão** (todos andam
  devagar), que esta medida não captura.
"""


def build_report(
    stats: Sequence[RaceStats], *, markdown: bool, generated_on: date | None = None
) -> str:
    table = stats_table(stats)
    summary = "\n".join(_summary(stats))
    if not markdown:
        return f"{summary}\n\n{render_text(table)}\n"
    day = (generated_on or date.today()).isoformat()
    return (
        "# Perda no pit e constantes por corrida\n\n"
        "Gerado por `python -m strategy.race_stats --markdown docs/perda-no-pit.md` em "
        f"{day}. As regras estão em `strategy/pitloss.py` e `strategy/priors.py`.\n\n"
        f"## Resumo\n\n{summary}\n\n{MARKDOWN_NOTES}\n## Por corrida\n\n{render_markdown(table)}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.race_stats",
        description="Mostra a perda no pit e as constantes de cada corrida, e as de reserva sem "
        "ela.",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="grava o relatório neste arquivo .md"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        stats = load_race_stats(session, args.year)
    if not stats:
        print(
            "Nenhuma corrida seca no banco. Ingira com: python -m ingestion --year 2024 --round 10"
        )
        return 1
    print(build_report(stats, markdown=False))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_report(stats, markdown=True), encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
