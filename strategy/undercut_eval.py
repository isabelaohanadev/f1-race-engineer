"""Avaliação do alerta de undercut nas paradas REAIS: `python -m strategy.undercut_eval`.

O PROTOCOLO. Cada parada de um piloto (o ATACANTE) na volta `s` é um evento: decide-se na volta
N = s - 1, o rival é o carro à frente dele (pelo relógio) e o resultado real é quem está à frente ao
fim da volta `s_r + 2`, em que `s_r` é a volta em que o RIVAL de fato parou (a mesma volta N+K+2 da
margem prevista, com K = s_r - N). A margem REALIZADA é a diferença dos relógios nessa volta; o
"sucesso" do undercut é a margem realizada ser positiva (o atacante terminou à frente).

Dois modos:
* `oráculo` (padrão): usa o horizonte K e os compostos que de fato aconteceram. Mede a FÍSICA do
  modelo (ritmo, degradação, perda no pit, tráfego) sem o que ninguém sabia na volta N (quando o
  rival pararia, para qual composto).
* `ao vivo`: o que o sistema faria na volta N, com o horizonte esperado e o composto mais provável.
  É o que decide o alerta de verdade e erra mais, e a diferença entre os dois modos é o custo de não
  saber o que o rival fará.

O VIÉS QUE NÃO DÁ PARA TIRAR (e que este relatório não finge tirar). Só se observam as paradas que
ACONTECERAM, e as equipes tendem a parar quando o undercut funciona:
1. a taxa de sucesso entre as paradas observadas é MAIOR que a de todas as situações em que um
   atacante estava atrás de um rival (as em que não funcionaria, ninguém parou, e não aparecem);
2. por isso a PRECISÃO medida aqui (dos alertas, quantos deram certo) é otimista, e a REVOCAÇÃO (dos
   undercuts que dariam certo, quantos o modelo alertou) não é mensurável: o denominador não existe;
3. a calibração é CONDICIONAL a "a equipe decidiu parar": a curva de confiabilidade diz se, nas
   paradas feitas, a probabilidade prevista bate com a frequência, e não se vale para uma situação
   qualquer;
4. o resultado real também depende do que os dois lados fizeram depois (o rival respondeu? parou
   cedo?). Eventos com parada extra na janela são descartados (`parada extra na janela`).
O que sobra é o que dá para dizer com honestidade: nas paradas feitas, o modelo ordena melhor que o
acaso, está bem calibrado ou não, e erra a margem em quantos segundos.

SEM VAZAMENTO. Cada evento é avaliado com `build_context` na volta N (que corta o futuro) e com as
constantes (`Priors`) calculadas SEM a corrida avaliada (`leave_one_out`). O fator de inflação do
desvio é calibrado em outras corridas (`loo_inflation`) e aplicado depois, a margem e o desvio cru
ficam guardados em cada resultado, então refazer a probabilidade com outro fator não reavalia nada.

O QUE ENTRA NA MÉDIA: eventos `primários` = sem regra dura de não alertar e SEM pista não verde
entre
N-1 e a volta final (uma parada sob SC não é comparável: custa menos). O relatório mostra os outros
em linhas separadas e quantos foram descartados, e por quê.
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np

from db.session import get_session_factory
from strategy.fuel import FuelVariant
from strategy.laps import LapRow, load_race_laps
from strategy.population import split_population
from strategy.priors import DEFAULT_PRIORS, RaceStats, leave_one_out
from strategy.race_stats import load_race_stats
from strategy.snapshot import car_states, driver_crossing_ms
from strategy.tables import render_markdown, render_text
from strategy.undercut import (
    GREEN,
    LapContext,
    UndercutAssessment,
    UndercutConfig,
    assess_undercut,
    build_context,
    car_ahead,
    normal_cdf,
)

MIN_EVAL_LAP = 8  # antes disso o modelo quase não tem dados; o alerta nem seria avaliado
CONTENTION_GAP_S = 5.0  # "em disputa": o atacante está a menos que isto do rival
DEFAULT_BINS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)  # os cortes de 0,5 e 0,6 são os do alerta


# ---------------------------------------------------------------- os eventos


@dataclass(frozen=True, slots=True)
class UndercutEvent:
    race: str
    attacker: str
    rival: str
    lap: int  # N: a volta em que se decide (o atacante entra na N+1)
    rival_stop: int  # a volta de entrada REAL do rival
    new_compound_a: str | None  # os compostos que de fato foram montados
    new_compound_r: str | None
    gap_s: float  # atacante atrás do rival na volta N

    @property
    def horizon(self) -> int:
        return self.rival_stop - self.lap

    @property
    def end_lap(self) -> int:
        return self.rival_stop + 2


def _pit_in_laps(rows: Sequence[LapRow]) -> dict[str, list[int]]:
    stops: dict[str, list[int]] = defaultdict(list)
    for lap in rows:
        if lap.pit_in:
            stops[lap.driver].append(lap.lap_number)
    return {driver: sorted(laps) for driver, laps in stops.items()}


def find_events(
    rows: Sequence[LapRow],
    total_laps: int,
    *,
    race: str = "",
    config: UndercutConfig | None = None,
) -> tuple[list[UndercutEvent], Counter[str]]:
    """Cada parada real de um piloto que dá um undercut observável, e por que as outras ficam de
    fora.

    Olha a corrida INTEIRA (o resultado real é do futuro de N) e só para montar o gabarito: a
    avaliação do evento usa `build_context`, que corta tudo depois de N."""
    cfg = config or UndercutConfig()
    stops = _pit_in_laps(rows)
    by_key = {(r.driver, r.lap_number): r for r in rows}
    events: list[UndercutEvent] = []
    skipped: Counter[str] = Counter()
    for attacker in sorted(stops):
        for s in stops[attacker]:
            n = s - 1
            if n < MIN_EVAL_LAP:
                skipped["volta inicial"] += 1
                continue
            states = car_states(rows, n)
            me = states.get(attacker)
            if me is None:
                skipped["sem relógio do atacante"] += 1
                continue
            rival = car_ahead(states, attacker)
            if rival is None:
                skipped["sem rival à frente"] += 1
                continue
            if states[rival].stint != me.stint:
                skipped["rival em outro stint"] += 1
                continue
            later = [lap for lap in stops.get(rival, []) if lap > n]
            if not later:
                skipped["rival não parou"] += 1
                continue
            s_r = later[0]
            horizon = s_r - n
            if s_r == s:
                skipped["paradas na mesma volta"] += 1
                continue
            if not cfg.k_min <= horizon <= cfg.k_max:
                skipped["horizonte fora do intervalo"] += 1
                continue
            end = s_r + 2
            if end > total_laps:
                skipped["corrida acaba antes"] += 1
                continue
            extra = any(s < lap <= end for lap in stops[attacker]) or any(
                s_r < lap <= end for lap in stops.get(rival, [])
            )
            if extra:
                skipped["parada extra na janela"] += 1
                continue
            end_a, end_r = by_key.get((attacker, end)), by_key.get((rival, end))
            if (
                end_a is None
                or end_r is None
                or None
                in (
                    end_a.session_time_ms,
                    end_r.session_time_ms,
                )
            ):
                skipped["sem relógio no fim"] += 1
                continue
            out_a, out_r = by_key.get((attacker, s + 1)), by_key.get((rival, s_r + 1))
            events.append(
                UndercutEvent(
                    race=race,
                    attacker=attacker,
                    rival=rival,
                    lap=n,
                    rival_stop=s_r,
                    new_compound_a=out_a.compound if out_a else None,
                    new_compound_r=out_r.compound if out_r else None,
                    gap_s=me.clock_s - states[rival].clock_s,
                )
            )
    return events, skipped


def realized_margin(rows: Sequence[LapRow], event: UndercutEvent) -> float | None:
    """O relógio do rival menos o do atacante ao fim da volta final: positivo = o atacante à
    frente."""
    clocks = {
        r.driver: r.session_time_ms
        for r in rows
        if r.lap_number == event.end_lap and r.driver in (event.attacker, event.rival)
    }
    a, r = clocks.get(event.attacker), clocks.get(event.rival)
    return None if a is None or r is None else (r - a) / 1000


def is_neutralized(rows: Sequence[LapRow], event: UndercutEvent) -> bool:
    """Alguma volta dos dois, de N-1 até a final, fora da bandeira verde (SC, VSC, vermelha,
    amarela):
    a parada nessa janela não é comparável com a em bandeira verde."""
    return any(
        r.track_status != GREEN
        for r in rows
        if r.driver in (event.attacker, event.rival)
        and event.lap - 1 <= r.lap_number <= event.end_lap
    )


# ---------------------------------------------------------------- o resultado de um evento


@dataclass(frozen=True, slots=True)
class EventResult:
    event: UndercutEvent
    assessment: UndercutAssessment
    realized: float  # s
    neutralized: bool
    inflation_used: float  # o fator embutido em `assessment.margin_sd`

    @property
    def success(self) -> bool:
        return self.realized > 0

    @property
    def error(self) -> float | None:
        """Margem prevista menos a realizada (s)."""
        m = self.assessment.margin
        return None if m is None else m - self.realized

    @property
    def raw_sd(self) -> float | None:
        """O desvio da margem SEM o fator de inflação."""
        sd = self.assessment.margin_sd
        return None if sd is None else sd / self.inflation_used

    @property
    def primary(self) -> bool:
        return self.assessment.eligible and not self.neutralized

    def probability(self, factor: float | None = None) -> float | None:
        """P(margem > 0) com outro fator de inflação, sem reavaliar nada."""
        m, sd = self.assessment.margin, self.raw_sd
        if m is None or sd is None or sd <= 0:
            return self.assessment.probability
        return normal_cdf(m / (sd * (self.inflation_used if factor is None else factor)))


def evaluate_race(
    rows: Sequence[LapRow],
    total_laps: int,
    *,
    race: str,
    priors=DEFAULT_PRIORS,
    config: UndercutConfig | None = None,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    oracle: bool = True,
    snapshot: str = "tempo",
) -> tuple[list[EventResult], Counter[str]]:
    """Avalia cada evento da corrida. `priors` deve ser o `leave_one_out` dela (nunca o padrão).

    `snapshot="tempo"` (o padrão): o modelo vê só o que se sabia no INSTANTE em que o atacante
    completa
    a volta N (`strategy.snapshot`). `"volta"` é o estado antigo, por contagem de voltas, que dá aos
    de
    trás informação de depois desse instante (diagnóstico do vazamento)."""
    cfg = config or UndercutConfig()
    events, skipped = find_events(rows, total_laps, race=race, config=cfg)
    contexts: dict[tuple[int, int | None], LapContext] = {}  # um ajuste por (volta, instante)
    results: list[EventResult] = []
    for event in events:
        realized = realized_margin(rows, event)
        if realized is None:
            skipped["sem relógio no fim"] += 1
            continue
        as_of = driver_crossing_ms(rows, event.attacker, event.lap) if snapshot == "tempo" else None
        if (event.lap, as_of) not in contexts:
            contexts[(event.lap, as_of)] = build_context(
                rows,
                upto_lap=event.lap,
                total_laps=total_laps,
                variant=variant,
                priors=priors,
                config=cfg,
                as_of_ms=as_of,
            )
        ctx = contexts[(event.lap, as_of)]
        kwargs = (
            {
                "new_compound": event.new_compound_a,
                "rival_new_compound": event.new_compound_r,
                "horizon_k": event.horizon,
            }
            if oracle
            else {}
        )
        assessment = assess_undercut(ctx, event.attacker, event.rival, **kwargs)
        results.append(
            EventResult(event, assessment, realized, is_neutralized(rows, event), cfg.sd_inflation)
        )
    return results, skipped


# ---------------------------------------------------------------- as métricas


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    """Erro quadrático médio da probabilidade (0 = perfeito; 0,25 = sempre dizer 50%)."""
    if not probabilities:
        return None
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True, slots=True)
class Bin:
    low: float
    high: float
    n: int
    mean_predicted: float | None
    observed_rate: float | None


def reliability_curve(
    probabilities: Sequence[float], outcomes: Sequence[bool], edges: Sequence[float] = DEFAULT_BINS
) -> list[Bin]:
    """Probabilidade prevista média x frequência observada, por faixa (a última inclui 1,0)."""
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    bins = []
    for i, (low, high) in enumerate(zip(edges, edges[1:], strict=False)):
        last = i == len(edges) - 2
        mask = (p >= low) & ((p <= high) if last else (p < high))
        n = int(mask.sum())
        bins.append(
            Bin(
                low,
                high,
                n,
                float(p[mask].mean()) if n else None,
                float(y[mask].mean()) if n else None,
            )
        )
    return bins


def precision_at(
    probabilities: Sequence[float], outcomes: Sequence[bool], threshold: float
) -> tuple[int, float | None]:
    """(alertas, precisão): dos eventos com probabilidade >= limiar, a fração que deu certo."""
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=bool)
    mask = p >= threshold
    n = int(mask.sum())
    return n, (float(y[mask].mean()) if n else None)


@dataclass(frozen=True, slots=True)
class InflationFit:
    factor: float  # o que se usa (nunca abaixo do piso)
    raw_factor: float | None  # o que os dados pedem
    n: int
    coverage_before: float | None  # cobertura do intervalo com o fator que estava em uso


def calibrate_inflation(
    results: Sequence[EventResult], *, coverage: float = 0.90, floor: float = 1.0
) -> InflationFit:
    """O fator que faz o intervalo central de `coverage` da margem cobrir a margem REALIZADA nessa
    fração dos eventos. `raw_factor = quantil(|erro| / desvio_cru) / z`; o piso evita "desinflar"
    por uma amostra pequena (o fator é de inflação). Só eventos primários entram."""
    z = NormalDist().inv_cdf((1 + coverage) / 2)
    ratios = [
        abs(r.error) / r.raw_sd
        for r in results
        if r.primary and r.error is not None and r.raw_sd and r.raw_sd > 0
    ]
    if not ratios:
        return InflationFit(floor, None, 0, None)
    raw = float(np.quantile(ratios, coverage, method="higher")) / z
    used = [r.inflation_used for r in results if r.primary]
    before = (
        float(np.mean([x <= z * u for x, u in zip(ratios, used, strict=False)])) if used else None
    )
    return InflationFit(max(raw, floor), raw, len(ratios), before)


def loo_inflation(
    by_race: dict[str, list[EventResult]], *, coverage: float = 0.90, floor: float = 1.0
) -> dict[str, InflationFit]:
    """O fator de cada corrida calibrado SÓ nas outras: calibrar no mesmo conjunto em que depois se
    mede o Brier deixaria o fator escolher a resposta."""
    return {
        label: calibrate_inflation(
            [r for other, rs in by_race.items() if other != label for r in rs],
            coverage=coverage,
            floor=floor,
        )
        for label in by_race
    }


@dataclass(frozen=True, slots=True)
class Summary:
    n_events: int
    n_primary: int
    base_rate: float | None  # a fração de sucesso entre os primários
    mean_predicted: float | None
    brier: float | None
    brier_baseline: float | None  # prever sempre a taxa-base (o piso a bater)
    bins: list[Bin]
    alerts: int
    precision: float | None
    threshold: float
    margin_bias: float | None  # prevista - realizada, em média
    margin_mae: float | None
    coverage_90: float | None  # fração em que a margem realizada cai a 1,645 desvios da prevista
    excluded: dict[str, int] = field(default_factory=dict)


def summarize(
    results: Sequence[EventResult],
    *,
    threshold: float = 0.6,
    factors: dict[str, float] | None = None,
    max_gap_s: float | None = None,
) -> Summary:
    """As métricas dos eventos primários. `factors` (por corrida) refaz a probabilidade com o fator
    de inflação calibrado nas OUTRAS corridas; `max_gap_s` fica só com os atacantes em disputa."""

    def keep(r: EventResult) -> bool:
        return max_gap_s is None or r.event.gap_s <= max_gap_s

    pool = [r for r in results if keep(r)]
    primary = [r for r in pool if r.primary]
    excluded = Counter()
    for r in pool:
        if r.neutralized:
            excluded["pista não verde na janela"] += 1
        elif not r.assessment.eligible:
            for reason in r.assessment.reasons or ("sem probabilidade",):
                excluded[reason] += 1
    probs, wins = [], []
    for r in primary:
        factor = None if factors is None else factors.get(r.event.race)
        p = r.probability(factor)
        if p is not None:
            probs.append(p)
            wins.append(r.success)
    rate = float(np.mean(wins)) if wins else None
    errors = [r.error for r in primary if r.error is not None]
    z90 = NormalDist().inv_cdf(0.95)

    def covered(r: EventResult) -> bool:
        """A margem realizada cai dentro de 1,645 desvios da prevista (com o fator desta
        corrida)."""
        factor = r.inflation_used if factors is None else factors.get(r.event.race, 1.0)
        return abs(r.error) <= z90 * r.raw_sd * factor

    scored = [r for r in primary if r.error is not None and r.raw_sd and r.raw_sd > 0]
    coverage = float(np.mean([covered(r) for r in scored])) if scored else None
    n_alerts, precision = precision_at(probs, wins, threshold)
    return Summary(
        n_events=len(pool),
        n_primary=len(primary),
        base_rate=rate,
        mean_predicted=float(np.mean(probs)) if probs else None,
        brier=brier_score(probs, wins),
        brier_baseline=brier_score([rate] * len(wins), wins) if rate is not None else None,
        bins=reliability_curve(probs, wins),
        alerts=n_alerts,
        precision=precision,
        threshold=threshold,
        margin_bias=float(np.mean(errors)) if errors else None,
        margin_mae=float(np.mean(np.abs(errors))) if errors else None,
        coverage_90=coverage,
        excluded=dict(excluded),
    )


# ---------------------------------------------------------------- a população


def evaluate_population(
    session,
    *,
    rounds: Sequence[int] | None = None,
    oracle: bool = True,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    earlier_only: bool = False,
    config: UndercutConfig | None = None,
) -> tuple[dict[str, list[EventResult]], Counter[str]]:
    """As corridas secas do backtest principal, cada uma com as constantes SEM ela. `rounds` limita
    às etapas pedidas (as constantes continuam vindo das outras corridas, todas)."""
    main, _ = split_population(session)
    stats: list[RaceStats] = load_race_stats(session)
    by_race: dict[str, list[EventResult]] = {}
    skipped: Counter[str] = Counter()
    for ref in main:
        if rounds is not None and ref.round not in rounds:
            continue
        rows = load_race_laps(session, ref.race_id)
        priors = leave_one_out(stats, ref.label, earlier_only=earlier_only)
        results, why = evaluate_race(
            rows,
            ref.total_laps,
            race=ref.label,
            priors=priors,
            config=config,
            variant=variant,
            oracle=oracle,
        )
        by_race[ref.label] = results
        skipped.update(why)
    return by_race, skipped


# ---------------------------------------------------------------- o relatório


def _f(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summary_table(label_to_summary: dict[str, Summary]) -> list[list[str]]:
    header = [
        "conjunto", "eventos", "primários", "taxa de sucesso", "prob. média", "Brier",
        "Brier (taxa-base)", "alertas", "precisão", "viés da margem (s)", "erro abs. (s)",
        "cobertura 90%",
    ]  # fmt: skip
    rows = [header]
    for label, s in label_to_summary.items():
        rows.append(
            [
                label,
                str(s.n_events),
                str(s.n_primary),
                _f(s.base_rate),
                _f(s.mean_predicted),
                _f(s.brier, 3),
                _f(s.brier_baseline, 3),
                str(s.alerts),
                _f(s.precision),
                _f(s.margin_bias),
                _f(s.margin_mae),
                _f(s.coverage_90),
            ]
        )
    return rows


def reliability_table(summary: Summary) -> list[list[str]]:
    rows = [["faixa de probabilidade", "eventos", "prevista (média)", "observada"]]
    for b in summary.bins:
        rows.append(
            [f"{b.low:.1f} a {b.high:.1f}", str(b.n), _f(b.mean_predicted), _f(b.observed_rate)]
        )
    return rows


PROTOCOL_NOTES = """\
## O que este relatório mede, e o que não mede

- **Só paradas que aconteceram, e as equipes tendem a parar quando funciona.** A taxa de sucesso
  daqui é mais alta que a de uma situação qualquer de atacante atrás de rival; a **precisão é
  otimista**
  e a **revocação não é mensurável** (as situações em que não funcionaria, e ninguém parou, não
  aparecem). A calibração é condicional a "a equipe decidiu parar".
- **Resultado real** = quem está à frente ao fim da volta `s_r + 2` (`s_r` = a parada real do
rival), pelos
  relógios da sessão. Eventos com parada extra de qualquer um na janela são descartados, e os com
  pista não verde entre N-1 e a volta final ficam em linha à parte (parada sob SC não é comparável).
- **Sem vazamento:** cada evento usa só as voltas até N e as constantes sem a corrida avaliada. O
fator
  de inflação é calibrado nas OUTRAS corridas.
- **Cada evento é uma decisão isolada** (um limiar, sem a histerese de voltas seguidas): a histerese
  evita piscar numa sequência de voltas, e aqui não há sequência.
"""


def build_report(
    by_race: dict[str, list[EventResult]],
    skipped: Counter[str],
    *,
    oracle: bool,
    threshold: float = 0.6,
    markdown: bool = False,
    generated_on: date | None = None,
) -> str:
    flat = [r for rs in by_race.values() for r in rs]
    fits = loo_inflation(by_race)
    factors = {label: fit.factor for label, fit in fits.items()}
    overall = calibrate_inflation(flat)
    sets = {
        "todos (fator em uso)": summarize(flat, threshold=threshold),
        "todos (fator calibrado nas outras corridas)": summarize(
            flat, threshold=threshold, factors=factors
        ),
        f"em disputa (gap <= {CONTENTION_GAP_S:.0f} s, fator calibrado)": summarize(
            flat, threshold=threshold, factors=factors, max_gap_s=CONTENTION_GAP_S
        ),
    }
    mode = (
        "oráculo (horizonte e compostos reais)"
        if oracle
        else "ao vivo (horizonte e composto previstos)"
    )
    lines = [
        f"Modo: **{mode}**. Corridas: {len(by_race)}. Limiar: {threshold}.",
        "",
        "Eventos descartados na busca: "
        + (", ".join(f"{k}: {v}" for k, v in skipped.most_common()) or "nenhum")
        + ".",
        "",
        f"Fator de inflação: pedido pelos dados {_f(overall.raw_factor)} (n={overall.n}), usado "
        f"{_f(overall.factor)}; cobertura do intervalo de 90% com o fator em uso: "
        f"{_f(overall.coverage_before)}.",
    ]
    excluded = sets["todos (fator em uso)"].excluded
    if excluded:
        lines += [
            "",
            "Fora dos primários: " + ", ".join(f"{k}: {v}" for k, v in excluded.items()) + ".",
        ]
    table = summary_table(sets)
    curve = reliability_table(sets["todos (fator calibrado nas outras corridas)"])
    per_race = [["corrida", "eventos", "primários", "taxa de sucesso", "Brier"]]
    for label, rs in by_race.items():
        s = summarize(rs, threshold=threshold, factors={label: factors[label]})
        per_race.append([label, str(s.n_events), str(s.n_primary), _f(s.base_rate), _f(s.brier, 3)])
    summary = "\n".join(lines)
    if not markdown:
        return (
            f"{summary}\n\n{render_text(table)}\n\nCalibração (fator calibrado):\n"
            f"{render_text(curve)}\n\nPor corrida:\n{render_text(per_race)}\n"
        )
    day = (generated_on or date.today()).isoformat()
    return (
        "# Avaliação do alerta de undercut nas paradas reais\n\n"
        f"Gerado por `python -m strategy.undercut_eval --markdown ...` em {day}. O protocolo está "
        "em `strategy/undercut_eval.py`.\n\n"
        f"{PROTOCOL_NOTES}\n## Resumo\n\n{summary}\n\n{render_markdown(table)}\n"
        "\n## Calibração (probabilidade prevista x frequência observada)\n\n"
        f"{render_markdown(curve)}\n"
        f"\n## Por corrida\n\n{render_markdown(per_race)}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.undercut_eval",
        description="Avalia o alerta de undercut nas paradas reais das corridas secas, com as "
        "constantes sem a corrida avaliada.",
    )
    parser.add_argument(
        "--round", type=int, action="append", default=None, help="só esta etapa (repetível)"
    )
    parser.add_argument("--live", action="store_true", help="horizonte e composto previstos")
    parser.add_argument(
        "--variant", choices=[v.value for v in FuelVariant], default=FuelVariant.ENCOLHIMENTO.value
    )
    parser.add_argument(
        "--earlier-only", action="store_true", help="constantes só de corridas anteriores"
    )
    parser.add_argument("--threshold", type=float, default=UndercutConfig().threshold)
    parser.add_argument("--markdown", type=Path, default=None, help="grava o relatório (.md)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        by_race, skipped = evaluate_population(
            session,
            rounds=args.round,
            oracle=not args.live,
            variant=FuelVariant(args.variant),
            earlier_only=args.earlier_only,
            config=replace(UndercutConfig(), threshold=args.threshold),
        )
    if not by_race:
        print(
            "Nenhuma corrida seca avaliada. Ingira com: python -m ingestion --year 2024 --round 10"
        )
        return 1
    print(build_report(by_race, skipped, oracle=not args.live, threshold=args.threshold))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            build_report(
                by_race, skipped, oracle=not args.live, threshold=args.threshold, markdown=True
            ),
            encoding="utf-8",
        )
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
