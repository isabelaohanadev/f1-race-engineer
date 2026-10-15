"""DIAGNÓSTICO PÓS-RESULTADO do Monte Carlo: `python -m strategy.diagnostics`.

ISTO NÃO É O BACKTEST. O backtest (`strategy.backtest`, protocolo congelado) já foi rodado e
publicado como saiu. Este módulo existe depois de ver esse resultado, para responder à pergunta
que ele deixou: por que o Monte Carlo hesita tanto no meio da corrida e erra mais que simplesmente
apostar no líder atual? Nada aqui muda um valor do protocolo nem o que foi medido: cada ablação
roda o MESMO simulador, com os MESMOS números aleatórios (as variantes não mudam a quantidade
sorteada), desligando UM componente de incerteza, e compara com a execução congelada. A diferença
é o que aquele componente custa em hesitação.

AS ABLAÇÕES, cada uma desliga:
* a incerteza do modelo (erro do nível e da inclinação de cada piloto, da diferença entre
  compostos e da perda no pit): cada piloto vira "exatamente o que o modelo estimou";
* o ruído de cada volta;
* o Safety Car e os abandonos;
* a folga sorteada na decisão de parar;
* tudo isso junto: o que sobra é a projeção QUASE determinística do modelo (ritmo, desgaste,
  paradas e tráfego; só o composto seguinte, sorteado, e o desempate de relógios iguais continuam
  aleatórios), e o quanto ela acerta o vencedor diz se o problema está nela ou na incerteza.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np

from strategy.backtest import TEMPO, RaceBundle, RaceOutcome, load_bundles, load_outcomes
from strategy.forecast_eval import (
    Interval,
    LapScore,
    cluster_bootstrap,
    current_leader,
    per_race_mean,
    score_forecast,
)
from strategy.fuel import FuelVariant
from strategy.montecarlo import SimInput, draw_all, prepare, simulate
from strategy.protocol import FROZEN, Protocol, frozen_sim_config
from strategy.results import on_track_order
from strategy.tables import render_markdown

Transform = Callable[[SimInput], SimInput]


def without_parameter_uncertainty(inp: SimInput) -> SimInput:
    """Cada piloto vira exatamente o que o modelo estimou: sem erro de nível, de inclinação, de
    diferença de composto nem de perda no pit."""
    return replace(
        inp,
        drivers=tuple(
            replace(
                d,
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


def without_lap_noise(inp: SimInput) -> SimInput:
    return replace(inp, drivers=tuple(replace(d, sigma=0.0) for d in inp.drivers))


def chain(*transforms: Transform) -> Transform:
    def apply(inp: SimInput) -> SimInput:
        for transform in transforms:
            inp = transform(inp)
        return inp

    return apply


@dataclass(frozen=True, slots=True)
class Ablation:
    name: str
    transform: Transform | None = None
    overrides: dict = field(default_factory=dict)  # campos de `SimConfig`


ALL_OFF = {"sc_rate_multiplier": 0.0, "retire_rate_multiplier": 0.0, "pit_decision_sd": 0.0}

ABLATIONS = (
    Ablation("sem incerteza do modelo", without_parameter_uncertainty),
    Ablation("sem ruído de volta", without_lap_noise),
    Ablation(
        "sem SC e sem abandono", None, {"sc_rate_multiplier": 0.0, "retire_rate_multiplier": 0.0}
    ),
    Ablation("sem folga na decisão de parar", None, {"pit_decision_sd": 0.0}),
    Ablation(
        "tudo desligado (quase determinístico)",
        chain(without_parameter_uncertainty, without_lap_noise),
        ALL_OFF,
    ),
)


def evaluate_ablations(
    bundle: RaceBundle,
    protocol: Protocol = FROZEN,
    *,
    ablations: Sequence[Ablation] = ABLATIONS,
    n_sims: int | None = None,
    every: int | None = None,
) -> dict[str, list[LapScore]]:
    """As notas de cada ablação na corrida, nas mesmas voltas do protocolo, com o estado por
    instante e a variante principal de `k`."""
    ref, rows = bundle.ref, bundle.rows
    config = frozen_sim_config(protocol, n_sims=n_sims or protocol.n_sims)
    order = on_track_order(rows)
    out: dict[str, list[LapScore]] = {a.name: [] for a in ablations}
    for lap in range(protocol.first_lap, ref.total_laps, every or protocol.every):
        try:
            base = prepare(
                rows,
                year=ref.year,
                round=ref.round,
                upto_lap=lap,
                total_laps=ref.total_laps,
                sim_priors=bundle.sim_priors,
                priors=bundle.priors,
                variant=FuelVariant(protocol.headline_variant),
                snapshot=TEMPO,
            )
        except ValueError:
            continue
        if len(base.drivers) < 3 or len(order) < 3:
            continue
        draws = draw_all(base, config)  # os mesmos números do protocolo
        leader = current_leader(rows, lap)
        for ablation in ablations:
            inp = ablation.transform(base) if ablation.transform else base
            forecast = simulate(inp, replace(config, **ablation.overrides), draws=draws)
            out[ablation.name].append(
                score_forecast(forecast, order, race=ref.label, current_leader=leader)
            )
    return out


# ---------------------------------------------------------------- o favorito


@dataclass(frozen=True, slots=True)
class FavoriteRow:
    low: float
    high: float
    n: int  # previsões (corrida, volta)
    n_races: int
    brier: Interval | None
    favorite_p: float | None  # a probabilidade que o Monte Carlo deu ao favorito dele
    favorite_won: float | None  # a frequência com que ele de fato ganhou
    leader_won: float | None  # a frequência com que o líder atual ganhou


def favorite_of(score: LapScore) -> tuple[float, bool] | None:
    """(probabilidade, ganhou) do favorito do Monte Carlo (o de maior probabilidade de vitória)."""
    if not score.win_pairs:
        return None
    return max(score.win_pairs, key=lambda pair: pair[0])


def favorite_rows(
    scores: Sequence[LapScore], edges: Sequence[float] = (0.0, 0.4, 0.8, 1.0001)
) -> list[FavoriteRow]:
    """Por faixa de andamento: o Brier (média por corrida, com bootstrap sobre corridas) e como o
    favorito do Monte Carlo se compara com o que de fato aconteceu e com o líder atual."""
    rows = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        chosen = [s for s in scores if lo <= s.progress < hi and favorite_of(s)]
        favorites = [favorite_of(s) for s in chosen]
        rows.append(
            FavoriteRow(
                float(lo),
                float(min(hi, 1.0)),
                len(chosen),
                len({s.race for s in chosen}),
                cluster_bootstrap(per_race_mean(scores, "brier_win", low=lo, high=min(hi, 1.0001))),
                float(np.mean([p for p, _ in favorites])) if favorites else None,
                float(np.mean([won for _, won in favorites])) if favorites else None,
                float(np.mean([s.leader_brier == 0 for s in chosen])) if chosen else None,
            )
        )
    return rows


def favorite_table(rows_by_name: dict[str, list[FavoriteRow]]) -> list[list[str]]:
    table = [
        [
            "configuração",
            "andamento",
            "previsões",
            "Brier da vitória",
            "p do favorito",
            "favorito ganhou",
            "líder atual ganhou",
        ]
    ]
    for name, rows in rows_by_name.items():
        for r in rows:
            table.append(
                [
                    name,
                    f"{r.low:.0%} a {r.high:.0%}",
                    str(r.n),
                    _fmt(r.brier),
                    _num(r.favorite_p),
                    _num(r.favorite_won),
                    _num(r.leader_won),
                ]
            )
    return table


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _fmt(interval: Interval | None) -> str:
    if interval is None:
        return "-"
    if interval.high - interval.low < 0.0005:
        return f"{interval.mean:.3f}"
    return f"{interval.mean:.3f} [{interval.low:.3f}, {interval.high:.3f}]"


NOTES = """\
## O que é isto

**Depois de ver o resultado do backtest, e fora dele.** O backtest congelado (`docs/backtest.md`)
mostrou que o Monte Carlo não supera "o líder atual vence" e é pior nas últimas faixas. Esta análise
pergunta **onde** está a hesitação. Nenhum valor do protocolo mudou e nada do que foi medido foi
refeito: cada linha abaixo é o mesmo simulador, nos mesmos números aleatórios, com UM componente
desligado.

- "p do favorito" é a probabilidade que o Monte Carlo dá ao piloto que ele considera mais provável
  de vencer; "favorito ganhou" é a frequência com que ele ganhou. Se a primeira é menor que a
  segunda, o Monte Carlo é **sub-confiante** (hesita mais do que deveria).
- "líder atual ganhou" é a frequência com que quem lidera no instante da previsão vence. Se o
  favorito do Monte Carlo ganha menos que o líder atual, a escolha dele é pior que a posição na
  pista.
- As ablações desligam componentes que **existem na realidade**: o resultado mostra quanto cada um
  custa em hesitação, não que deva ser desligado.
"""


CALIBRATION_TOLERANCE = 0.05  # até 5 pontos de diferença entre o dado e o ocorrido é "calibrado"


def calibration_word(given: float, happened: float) -> str:
    """Ganhou mais do que o Monte Carlo lhe deu: hesitou (sub-confiante); menos: exagerou."""
    gap = happened - given
    if gap > CALIBRATION_TOLERANCE:
        return "sub-confiante"
    if gap < -CALIBRATION_TOLERANCE:
        return "super-confiante"
    return "calibrado"


def reading_lines(
    frozen: dict[str, list[FavoriteRow]], ablated: dict[str, list[FavoriteRow]]
) -> list[str]:
    """A leitura do resultado, gerada dos mesmos números (a faixa do meio da corrida, 40% a 80%)."""

    def middle(rows: list[FavoriteRow]) -> FavoriteRow | None:
        row = next((r for r in rows if r.low == 0.4 and r.n), None)
        has_all = row and None not in (row.favorite_p, row.favorite_won, row.leader_won)
        return row if has_all else None

    lines = []
    base = middle(next(iter(frozen.values())))
    if base:
        word = calibration_word(base.favorite_p, base.favorite_won)  # type: ignore[arg-type]
        lines.append(
            f"- No meio da corrida o Monte Carlo congelado dá ao favorito {base.favorite_p:.2f} e "
            f"o favorito ganha {base.favorite_won:.2f}: {word}. O líder atual ganha "
            f"{base.leader_won:.2f}."
        )
    for name, rows in ablated.items():
        row = middle(rows)
        if row is None:
            continue
        word = calibration_word(row.favorite_p, row.favorite_won)  # type: ignore[arg-type]
        versus = (
            "ganha menos que o líder atual"
            if row.favorite_won < row.leader_won  # type: ignore[operator]
            else "ganha tanto quanto ou mais que o líder atual"
        )
        lines.append(
            f"- **{name}:** p do favorito {row.favorite_p:.2f}, ganhou {row.favorite_won:.2f} "
            f"({word}); o favorito {versus} ({row.leader_won:.2f})."
        )
    return lines


def build_report(
    frozen: dict[str, list[FavoriteRow]],
    ablated: dict[str, list[FavoriteRow]],
    *,
    n_races: int,
    generated_on: date | None = None,
) -> str:
    day = (generated_on or date.today()).isoformat()
    table = favorite_table({**frozen, **ablated})
    return (
        "# Diagnóstico pós-resultado do Monte Carlo\n\n"
        f"Gerado por `python -m strategy.diagnostics` em {day} sobre {n_races} corridas secas, "
        "**depois** do backtest. Não é parte do backtest.\n\n"
        f"{NOTES}\n## Resultado\n\n{render_markdown(table)}\n\n"
        "## Leitura (faixa de 40% a 80% da corrida)\n\n"
        + "\n".join(reading_lines(frozen, ablated))
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(
        prog="python -m strategy.diagnostics",
        description="Diagnóstico pós-resultado: desliga um componente de incerteza do Monte Carlo "
        "por vez e compara a hesitação com a execução congelada.",
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="os resultados do backtest (.pkl)"
    )
    parser.add_argument("--markdown", type=Path, default=None, help="grava o relatório (.md)")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    from db.session import get_session_factory

    outcomes: list[RaceOutcome] = [o for o in load_outcomes(args.cache) if o.main]
    with get_session_factory()() as session:
        bundles = [b for b in load_bundles(session) if b.main]
    frozen_scores = [
        s for o in outcomes for s in o.forecast.get((FROZEN.headline_variant, TEMPO, 1.0), [])
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_ablations, bundles))
    frozen = {"congelado (o backtest)": favorite_rows(frozen_scores)}
    ablated = {}
    for ablation in ABLATIONS:
        scores = [s for result in results for s in result[ablation.name]]
        ablated[ablation.name] = favorite_rows(scores)
    report = build_report(frozen, ablated, n_races=len(bundles))
    print(report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    from strategy.diagnostics import main as _main

    sys.exit(_main())
