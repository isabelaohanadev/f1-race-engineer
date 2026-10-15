"""Avaliação do Monte Carlo contra a ordem NA PISTA: `python -m strategy.forecast_eval`.

A UNIDADE DA AMOSTRA É A CORRIDA, não a volta. As voltas de uma mesma corrida são fortemente
correlacionadas: se o modelo erra o ritmo de um piloto na volta 20, erra na 21, na 22... Tratar as
~1.200 voltas das 21 corridas secas como 1.200 observações independentes faria o intervalo de
confiança parecer estreito demais. O tamanho EFETIVO da amostra da calibração é o número de corridas
(21), e este módulo reporta o Brier por volta com um intervalo de confiança por BOOTSTRAP SOBRE
CORRIDAS: ressorteia corridas inteiras, com reposição, e refaz a média. O relatório diz isso.

O QUE SE MEDE, por (corrida, volta N):
* Brier da vitória: soma sobre os pilotos de (p_i - y_i)^2, com y_i = 1 para quem ganhou NA PISTA
  (de 0, perfeito, a 2); o do pódio idem com y_i = 1 para os três primeiros. O vencedor que o
  simulador nem tinha (sem ritmo) conta como previsto com probabilidade 0.
* a probabilidade que o modelo deu a quem de fato ganhou, e se o favorito ganhou.
* DOIS COMPARADORES simples, para o modelo ter o que bater: "o líder atual vence" (p = 1 para quem
  lidera na volta N) e "todos iguais" (p = 1/n). Um Monte Carlo que não bate "o líder atual vence"
  não está acrescentando nada ao que já se vê na tela.

SEM VAZAMENTO. Cada previsão usa `prepare` (que corta o futuro), as constantes sem a corrida
avaliada
(`leave_one_out` do ritmo e `leave_one_out_history` do SC, dos abandonos e das trocas de composto),
e a
verdade só aparece depois, na comparação.

VARIANTES. Para comparar o `k` físico, o estimado e o encolhimento, `paired_bootstrap` ressorteia
corridas e olha a DIFERENÇA do Brier por corrida: as variantes usam os mesmos números aleatórios
(montecarlo.py), então essa diferença é sinal, e o bootstrap mede se ele passa do ruído entre
corridas.
"""

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from db.session import get_session_factory
from strategy.fuel import FuelVariant
from strategy.history import (
    RaceHistory,
    SimPriors,
    leave_one_out_history,
    load_race_histories,
)
from strategy.laps import LapRow, load_race_laps
from strategy.montecarlo import PODIUM_SIZE, RaceForecast, SimConfig, prepare, simulate
from strategy.population import RaceRef, split_population
from strategy.priors import DEFAULT_PRIORS, Priors, RaceStats, leave_one_out
from strategy.race_stats import load_race_stats
from strategy.results import on_track_order
from strategy.tables import render_markdown, render_text

DEFAULT_BINS = 10
DEFAULT_BOOTSTRAP = 2000
MIN_DRIVERS = 3  # abaixo disso não há pódio para prever


# ---------------------------------------------------------------- a nota de uma previsão


@dataclass(frozen=True, slots=True)
class LapScore:
    race: str
    lap: int
    total_laps: int
    brier_win: float
    brier_podium: float
    p_winner: float  # a probabilidade que o modelo deu a quem ganhou na pista
    favorite_won: bool
    leader_brier: float  # "o líder atual vence": p = 1 para ele
    uniform_brier: float  # "todos iguais": p = 1/n

    @property
    def progress(self) -> float:
        """Quanto da corrida já tinha passado (0 a 1)."""
        return self.lap / self.total_laps


def score_forecast(
    forecast: RaceForecast, order: Sequence[str], *, race: str, current_leader: str | None
) -> LapScore:
    """Nota uma previsão contra a ordem de chegada na pista (`order[0]` ganhou)."""
    winner = order[0]
    podium = set(order[:PODIUM_SIZE])
    wins = {d.driver: d.win for d in forecast.drivers}
    podiums = {d.driver: d.podium for d in forecast.drivers}
    brier_win = sum((p - (driver == winner)) ** 2 for driver, p in wins.items())
    brier_podium = sum((q - (driver in podium)) ** 2 for driver, q in podiums.items())
    # quem ganhou ou subiu ao pódio e o simulador nem tinha: previsto com probabilidade 0
    brier_win += 0.0 if winner in wins else 1.0
    brier_podium += float(sum(1 for driver in podium if driver not in podiums))
    n = len(wins)
    favorite = max(wins, key=wins.get) if wins else None  # type: ignore[arg-type]
    leader = {driver: float(driver == current_leader) for driver in wins}
    leader_brier = sum((p - (driver == winner)) ** 2 for driver, p in leader.items())
    leader_brier += 0.0 if winner in wins else 1.0
    uniform_brier = sum((1.0 / n - (driver == winner)) ** 2 for driver in wins) if n else 1.0
    uniform_brier += 0.0 if winner in wins else 1.0
    return LapScore(
        race=race,
        lap=forecast.lap,
        total_laps=forecast.total_laps,
        brier_win=brier_win,
        brier_podium=brier_podium,
        p_winner=wins.get(winner, 0.0),
        favorite_won=favorite == winner,
        leader_brier=leader_brier,
        uniform_brier=uniform_brier,
    )


def current_leader(rows: Sequence[LapRow], lap: int) -> str | None:
    """Quem lidera na volta `lap`: o menor relógio entre os que a completaram."""
    clocks = {
        r.driver: r.session_time_ms
        for r in rows
        if r.lap_number == lap and r.session_time_ms is not None
    }
    return min(clocks, key=clocks.get) if clocks else None  # type: ignore[arg-type]


def evaluate_race(
    rows: Sequence[LapRow],
    ref: RaceRef,
    *,
    laps: Sequence[int],
    priors: Priors = DEFAULT_PRIORS,
    sim_priors: SimPriors,
    config: SimConfig | None = None,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
) -> list[LapScore]:
    """As notas de cada volta de `laps` da corrida, com as constantes dadas (as SEM a corrida)."""
    order = on_track_order(rows)
    if len(order) < MIN_DRIVERS:
        return []
    scores = []
    for lap in laps:
        if not 1 <= lap < ref.total_laps:
            continue
        inp = prepare(
            rows,
            year=ref.year,
            round=ref.round,
            upto_lap=lap,
            total_laps=ref.total_laps,
            sim_priors=sim_priors,
            priors=priors,
            variant=variant,
        )
        if len(inp.drivers) < MIN_DRIVERS:
            continue
        forecast = simulate(inp, config)
        scores.append(
            score_forecast(
                forecast, order, race=ref.label, current_leader=current_leader(rows, lap)
            )
        )
    return scores


# ---------------------------------------------------------------- bootstrap sobre corridas


@dataclass(frozen=True, slots=True)
class Interval:
    mean: float
    low: float
    high: float
    n_races: int


def cluster_bootstrap(
    per_race: Mapping[str, float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval | None:
    """A média de um valor por corrida e o intervalo de `1 - alpha` por bootstrap: ressorteia
    CORRIDAS inteiras, com reposição, e refaz a média (percentis). Com uma só corrida não há
    intervalo, e a função devolve o ponto com low = high = média."""
    values = np.asarray(list(per_race.values()), dtype=float)
    if values.size == 0:
        return None
    mean = float(values.mean())
    if values.size == 1:
        return Interval(mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(mean, float(low), float(high), int(values.size))


def per_race_mean(
    scores: Sequence[LapScore], attribute: str, *, low: float = 0.0, high: float = 1.0
) -> dict[str, float]:
    """A média de `attribute` por corrida nas voltas com progresso em [low, high). A volta é o que
    se promedia DENTRO da corrida; a corrida é a unidade que o bootstrap ressorteia."""
    by_race: dict[str, list[float]] = {}
    last = high >= 1.0
    for s in scores:
        if low <= s.progress < high or (last and s.progress == 1.0):
            by_race.setdefault(s.race, []).append(float(getattr(s, attribute)))
    return {race: float(np.mean(v)) for race, v in by_race.items()}


def brier_by_progress(
    scores: Sequence[LapScore],
    attribute: str = "brier_win",
    *,
    bins: int = DEFAULT_BINS,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> list[tuple[float, float, Interval | None]]:
    """O Brier por faixa de andamento da corrida (0-10%, 10-20%...), cada uma com o seu intervalo de
    bootstrap sobre corridas."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    return [
        (
            float(lo),
            float(hi),
            cluster_bootstrap(
                per_race_mean(scores, attribute, low=float(lo), high=float(hi)),
                n_boot=n_boot,
                seed=seed,
            ),
        )
        for lo, hi in zip(edges, edges[1:], strict=False)
    ]


def paired_bootstrap(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval | None:
    """O intervalo da diferença `a - b` (por exemplo, um `k` contra o outro), corrida a corrida:
    só as corridas presentes nos dois entram, e o bootstrap ressorteia essas corridas."""
    shared = sorted(set(a) & set(b))
    return cluster_bootstrap(
        {race: a[race] - b[race] for race in shared}, n_boot=n_boot, alpha=alpha, seed=seed
    )


# ---------------------------------------------------------------- a população


def evaluate_population(
    session,
    *,
    rounds: Sequence[int] | None = None,
    every: int = 5,
    n_sims: int = 5000,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    earlier_only: bool = False,
) -> dict[str, list[LapScore]]:
    """As corridas secas do backtest principal, cada uma com as constantes SEM ela (as do ritmo e as
    do SC, dos abandonos e das trocas). `every`: uma volta em cada `every`, a partir da 5."""
    main, _ = split_population(session)
    stats: list[RaceStats] = load_race_stats(session)
    histories: list[RaceHistory] = load_race_histories(session)
    config = SimConfig(n_sims=n_sims)
    out: dict[str, list[LapScore]] = {}
    for ref in main:
        if rounds is not None and ref.round not in rounds:
            continue
        rows = load_race_laps(session, ref.race_id)
        laps = list(range(5, ref.total_laps, every))
        out[ref.label] = evaluate_race(
            rows,
            ref,
            laps=laps,
            priors=leave_one_out(stats, ref.label, earlier_only=earlier_only),
            sim_priors=leave_one_out_history(histories, ref.label, earlier_only=earlier_only),
            config=config,
            variant=variant,
        )
    return out


# ---------------------------------------------------------------- o relatório


def _fmt(interval: Interval | None, digits: int = 3) -> str:
    if interval is None:
        return "-"
    return f"{interval.mean:.{digits}f} [{interval.low:.{digits}f}, {interval.high:.{digits}f}]"


def progress_table(scores: Sequence[LapScore], *, bins: int = DEFAULT_BINS) -> list[list[str]]:
    """O Brier da vitória por faixa de andamento, do modelo e dos dois comparadores."""
    rows = [["andamento", "corridas", "Monte Carlo", "o líder vence", "todos iguais", "pódio (MC)"]]
    model = brier_by_progress(scores, "brier_win", bins=bins)
    leader = brier_by_progress(scores, "leader_brier", bins=bins)
    uniform = brier_by_progress(scores, "uniform_brier", bins=bins)
    podium = brier_by_progress(scores, "brier_podium", bins=bins)
    for (lo, hi, m), (_, _, ld), (_, _, un), (_, _, pd) in zip(
        model, leader, uniform, podium, strict=True
    ):
        rows.append(
            [
                f"{lo:.0%} a {hi:.0%}",
                str(m.n_races if m else 0),
                _fmt(m),
                _fmt(ld),
                _fmt(un),
                _fmt(pd),
            ]
        )
    return rows


def overall_table(scores: Sequence[LapScore]) -> list[list[str]]:
    rows = [["medida", "média por corrida [IC 95% por bootstrap sobre corridas]"]]
    for label, attribute in (
        ("Brier da vitória", "brier_win"),
        ("Brier do pódio", "brier_podium"),
        ("probabilidade dada a quem ganhou", "p_winner"),
        ("o favorito ganhou (fração)", "favorite_won"),
        ("Brier: o líder atual vence", "leader_brier"),
        ("Brier: todos iguais", "uniform_brier"),
    ):
        rows.append([label, _fmt(cluster_bootstrap(per_race_mean(scores, attribute)))])
    model, leader = per_race_mean(scores, "brier_win"), per_race_mean(scores, "leader_brier")
    rows.append(
        [
            "Brier MC menos Brier do líder (negativo = o MC é melhor)",
            _fmt(paired_bootstrap(model, leader)),
        ]
    )
    return rows


NOTES = """\
## O que este relatório mede, e o que não mede

- **A unidade é a corrida, não a volta.** As voltas de uma mesma corrida são correlacionadas (um
erro
  de ritmo se repete volta após volta), então o tamanho efetivo da amostra é o **número de
  corridas**
  (21), não o de voltas (~1.200). Os intervalos são de **bootstrap sobre corridas**: ressorteiam
  corridas inteiras com reposição. Dentro de cada corrida as voltas são promediadas primeiro.
- **A verdade é a ordem na pista** (`docs/verdade-do-backtest.md`), não o resultado oficial.
- **Sem vazamento:** cada previsão usa só as voltas até N e as constantes sem a corrida avaliada.
- **Dois comparadores:** "o líder atual vence" e "todos iguais". Um Monte Carlo que não bate o
  primeiro não acrescenta nada ao que já se vê na tela.
- **O que ficou de fora do simulador** (e portanto pesa nestes números): VSC, bandeira vermelha e
  abandono causando SC; as regras de parada e de ultrapassagem são heurísticas.
"""


def build_report(
    by_race: Mapping[str, Sequence[LapScore]],
    *,
    markdown: bool = False,
    generated_on: date | None = None,
    bins: int = DEFAULT_BINS,
) -> str:
    scores = [s for race in by_race.values() for s in race]
    n_races = sum(1 for race in by_race.values() if race)
    header = (
        f"{n_races} corridas, {len(scores)} previsões (volta a volta). "
        f"Unidade do intervalo: a corrida ({n_races}), não a volta."
    )
    overall, progress = overall_table(scores), progress_table(scores, bins=bins)
    if not markdown:
        return (
            f"{header}\n\n{render_text(overall)}\n\n"
            f"Por andamento da corrida:\n{render_text(progress)}\n"
        )
    day = (generated_on or date.today()).isoformat()
    return (
        "# Avaliação do Monte Carlo contra a ordem na pista\n\n"
        f"Gerado por `python -m strategy.forecast_eval --markdown ...` em {day}.\n\n{NOTES}\n"
        f"## Resumo\n\n{header}\n\n{render_markdown(overall)}\n"
        f"\n## Brier da vitória por andamento da corrida\n\n{render_markdown(progress)}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.forecast_eval",
        description="Avalia as probabilidades de vitória e pódio do Monte Carlo contra a ordem na "
        "pista, com bootstrap sobre corridas.",
    )
    parser.add_argument("--round", type=int, action="append", default=None, help="só esta etapa")
    parser.add_argument("--every", type=int, default=5, help="uma volta em cada N (padrão 5)")
    parser.add_argument("--sims", type=int, default=5000, help="simulações por volta")
    parser.add_argument(
        "--variant", choices=[v.value for v in FuelVariant], default=FuelVariant.ENCOLHIMENTO.value
    )
    parser.add_argument(
        "--earlier-only", action="store_true", help="constantes só de corridas anteriores"
    )
    parser.add_argument("--markdown", type=Path, default=None, help="grava o relatório (.md)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    with get_session_factory()() as session:
        by_race = evaluate_population(
            session,
            rounds=args.round,
            every=args.every,
            n_sims=args.sims,
            variant=FuelVariant(args.variant),
            earlier_only=args.earlier_only,
        )
    if not by_race:
        print(
            "Nenhuma corrida seca avaliada. Ingira com: python -m ingestion --year 2024 --round 10"
        )
        return 1
    print(build_report(by_race))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_report(by_race, markdown=True), encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
