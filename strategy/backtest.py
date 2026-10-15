"""O backtest da Fase 3: `python -m strategy.backtest`.

Roda o protocolo congelado (`strategy/protocol.py`, `docs/backtest-protocolo.md`) em cada corrida:
leave-one-race-out nas 21 corridas secas, com a britânica, o Canadá e o Brasil à parte (as
constantes
delas vêm das 21 secas), e guarda o que o relatório (`strategy/backtest_report.py`) precisa.

POR CORRIDA (`evaluate_bundle`, uma função pura e serializável, para rodar uma corrida por
processo):
* o erro do tempo de volta (`strategy.laptime_eval`), nas três variantes de `k` e no estado antigo;
* o Brier do Monte Carlo (`strategy.forecast_eval`), nas três variantes, com a taxa de SC em 0,5x,
1x e
  2x na variante principal e o estado antigo como diagnóstico;
* o undercut nas paradas reais (`strategy.undercut_eval`), ao vivo e no oráculo, e ao vivo no estado
  antigo como diagnóstico.

O ESTADO. Tudo o que está marcado "tempo" vê só o que se sabia no instante do evento (o líder
completa
a volta N, para o Monte Carlo e o tempo de volta; o atacante completa a volta N, para o undercut):
`strategy/snapshot.py`. O "volta" é o desenho antigo e só serve para medir o vazamento.

SEM A CORRIDA AVALIADA. As constantes (`Priors` do ritmo e da perda no pit, `SimPriors` do SC, dos
abandonos e das trocas) de cada corrida seca saem das outras 20.
"""

import argparse
import logging
import os
import pickle
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from db.session import get_session_factory
from strategy.forecast_eval import LapScore, evaluate_race_configs
from strategy.fuel import FuelVariant
from strategy.history import (
    SimPriors,
    leave_one_out_history,
    load_race_histories,
)
from strategy.laps import LapRow, load_race_laps
from strategy.laptime_eval import LapTimeRecord, lap_time_records
from strategy.population import RaceRef, split_population
from strategy.priors import Priors, leave_one_out
from strategy.protocol import FROZEN, Protocol, frozen_sim_config, frozen_undercut_config
from strategy.race_stats import load_race_stats
from strategy.undercut_eval import EventResult, evaluate_race_modes

LIVE, ORACLE = "ao vivo", "oráculo"
TEMPO, VOLTA = "tempo", "volta"


@dataclass(frozen=True, slots=True)
class RaceBundle:
    """Uma corrida com tudo o que o avaliador precisa, pronta para ir a um processo."""

    ref: RaceRef
    rows: list[LapRow]
    priors: Priors
    sim_priors: SimPriors
    main: bool  # as 21 secas (leave-one-out) ou as de fora (constantes das 21)


@dataclass(slots=True)
class RaceOutcome:
    label: str
    kind: str
    main: bool
    total_laps: int
    laptime: dict[tuple[str, str], list[LapTimeRecord]] = field(
        default_factory=dict
    )  # (variante, estado)
    forecast: dict[tuple[str, str, float], list[LapScore]] = field(default_factory=dict)
    undercut: dict[tuple[str, str], list[EventResult]] = field(
        default_factory=dict
    )  # (modo, estado)
    skipped: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def evaluate_bundle(
    bundle: RaceBundle,
    protocol: Protocol = FROZEN,
    *,
    n_sims: int | None = None,
    every: int | None = None,
) -> RaceOutcome:
    """Roda o protocolo numa corrida. `n_sims` e `every` só existem para as execuções de verificação
    (menos simulações, menos voltas): o backtest de verdade usa os valores do protocolo."""
    started = time.perf_counter()
    ref, rows = bundle.ref, bundle.rows
    headline = FuelVariant(protocol.headline_variant)
    variants = [FuelVariant(v) for v in protocol.variants]
    step = every or protocol.every
    laps = list(range(protocol.first_lap, ref.total_laps, step))
    sim_config = frozen_sim_config(protocol, n_sims=n_sims or protocol.n_sims)
    undercut_config = frozen_undercut_config(protocol)
    out = RaceOutcome(ref.label, ref.kind.value, bundle.main, ref.total_laps)

    # ---- o erro do tempo de volta
    lap_runs = (
        [(v, TEMPO) for v in variants] + [(headline, VOLTA)] if bundle.main else [(headline, TEMPO)]
    )
    for variant, snapshot in lap_runs:
        out.laptime[(variant.value, snapshot)] = lap_time_records(
            rows,
            race=ref.label,
            total_laps=ref.total_laps,
            laps=laps,
            horizons=protocol.horizons,
            priors=bundle.priors,
            variant=variant,
            snapshot=snapshot,
            coverage=protocol.interval_coverage,
            max_silence_laps=protocol.max_silence_laps,
        )

    # ---- as probabilidades de vitória e pódio
    if bundle.main:
        configs: list[tuple[FuelVariant, str, Sequence[float]]] = [
            (v, TEMPO, (1.0,)) for v in variants if v is not headline
        ]
        configs.append((headline, TEMPO, protocol.sc_multipliers))
        configs.append((headline, VOLTA, (1.0,)))
    else:
        configs = [(headline, TEMPO, (1.0,))]
    out.forecast = evaluate_race_configs(
        rows,
        ref,
        laps=laps,
        priors=bundle.priors,
        sim_priors=bundle.sim_priors,
        configs=configs,
        config=sim_config,
    )

    # ---- o undercut nas paradas reais
    skipped: Counter[str] = Counter()
    undercut_runs = (
        [(TEMPO, (False, True)), (VOLTA, (False,))] if bundle.main else [(TEMPO, (False,))]
    )
    for snapshot, modes in undercut_runs:
        by_mode, why = evaluate_race_modes(
            rows,
            ref.total_laps,
            race=ref.label,
            priors=bundle.priors,
            config=undercut_config,
            variant=headline,
            snapshot=snapshot,
            modes=modes,
        )
        for oracle, results in by_mode.items():
            out.undercut[(ORACLE if oracle else LIVE, snapshot)] = results
        if snapshot == TEMPO:
            skipped.update(why)
    out.skipped = dict(skipped)
    out.seconds = time.perf_counter() - started
    return out


def load_bundles(session, rounds: Sequence[int] | None = None) -> list[RaceBundle]:
    """As corridas do backtest: as secas (cada uma com as constantes SEM ela) e as de fora
    (britânica,
    Canadá, Brasil: com as constantes das 21 secas)."""
    main, apart = split_population(session)
    stats = load_race_stats(session)
    histories = load_race_histories(session)
    bundles = []
    for ref in main:
        if rounds is not None and ref.round not in rounds:
            continue
        bundles.append(
            RaceBundle(
                ref,
                load_race_laps(session, ref.race_id),
                leave_one_out(stats, ref.label),
                leave_one_out_history(histories, ref.label),
                True,
            )
        )
    everything, history_everything = leave_one_out(stats, ""), leave_one_out_history(histories, "")
    for ref in apart:
        if rounds is not None and ref.round not in rounds:
            continue
        bundles.append(
            RaceBundle(
                ref,
                load_race_laps(session, ref.race_id),
                everything,
                history_everything,
                False,
            )
        )
    return bundles


def run_backtest(
    bundles: Sequence[RaceBundle],
    protocol: Protocol = FROZEN,
    *,
    workers: int | None = None,
    n_sims: int | None = None,
    every: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[RaceOutcome]:
    """Todas as corridas, uma por processo (as maiores primeiro). A ordem do resultado é a das
    corridas dadas."""
    say = progress or (lambda message: None)
    order = sorted(range(len(bundles)), key=lambda i: -bundles[i].ref.total_laps)
    results: dict[int, RaceOutcome] = {}
    n_workers = max(1, min(workers or (os.cpu_count() or 2) - 1, len(bundles)))
    if n_workers == 1:
        for i in order:
            results[i] = evaluate_bundle(bundles[i], protocol, n_sims=n_sims, every=every)
            say(f"{bundles[i].ref.label}: {results[i].seconds:.0f} s")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(evaluate_bundle, bundles[i], protocol, n_sims=n_sims, every=every): i
                for i in order
            }
            for done in as_completed(futures):
                i = futures[done]
                results[i] = (
                    done.result()
                )  # um erro numa corrida derruba o backtest: nada é omitido
                say(f"{bundles[i].ref.label}: {results[i].seconds:.0f} s")
    return [results[i] for i in range(len(bundles))]


def save_outcomes(outcomes: Sequence[RaceOutcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(outcomes), handle)


def load_outcomes(path: Path) -> list[RaceOutcome]:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 (um arquivo que este módulo mesmo gravou)


def main(argv: list[str] | None = None) -> int:
    from strategy.backtest_report import build_report

    parser = argparse.ArgumentParser(
        prog="python -m strategy.backtest",
        description="Roda o protocolo congelado do backtest e escreve o relatório.",
    )
    parser.add_argument("--markdown", type=Path, default=None, help="grava o relatório (.md)")
    parser.add_argument("--cache", type=Path, default=None, help="guarda/lê os resultados (.pkl)")
    parser.add_argument("--reuse", action="store_true", help="lê o cache em vez de rodar de novo")
    parser.add_argument("--round", type=int, action="append", default=None, help="só esta etapa")
    parser.add_argument("--workers", type=int, default=None, help="processos (padrão: CPUs - 1)")
    parser.add_argument(
        "--quick", action="store_true", help="verificação: 300 simulações, 1 volta em 20"
    )
    parser.add_argument(
        "--adjustment",
        action="append",
        default=[],
        help="um ajuste pós-resultado, em uma frase (repetível): entra no relatório",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.reuse:
        if args.cache is None:
            parser.error("--reuse precisa de --cache")
        outcomes = load_outcomes(args.cache)
    else:
        with get_session_factory()() as session:
            bundles = load_bundles(session, args.round)
        if not bundles:
            print(
                "Nenhuma corrida no banco. Ingira com: python -m ingestion --year 2024 --round 10"
            )
            return 1
        started = time.perf_counter()
        outcomes = run_backtest(
            bundles,
            workers=args.workers,
            n_sims=300 if args.quick else None,
            every=20 if args.quick else None,
            progress=lambda m: print(m, file=sys.stderr, flush=True),
        )
        print(f"{len(outcomes)} corridas em {time.perf_counter() - started:.0f} s", file=sys.stderr)
        if args.cache is not None:
            save_outcomes(outcomes, args.cache)
    report = build_report(outcomes, adjustments=tuple(args.adjustment), quick=args.quick)
    print(report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
