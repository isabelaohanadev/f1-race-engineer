"""Avaliação OFFLINE do alerta de undercut na forma em que ele roda ao vivo:
`python -m strategy.undercut_live`. Relatório: `docs/undercut-ao-vivo.md`.

POR QUE EXISTE. O backtest (`docs/backtest.md`) mediu o alerta no instante em que o ATACANTE
completa a volta N (`strategy.snapshot.state_at_driver_lap`). O consumidor de estratégia o calcula
no instante em que o LÍDER completa a volta (um contexto só por volta, `build_context(as_of_ms=)`:
~0,15 s; um por atacante seriam ~3 s por volta). São duas formas diferentes: ao vivo, quem está
atrás
do líder aparece com a volta anterior, e o líder nunca é o "carro à frente" de ninguém (ele já
completou a volta N e os outros estão na N-1, e `car_ahead` só compara carros na mesma volta). Os
números do backtest NÃO valem para a forma ao vivo. Este módulo mede a forma ao vivo.

PROTOCOLO (escrito e commitado ANTES de rodar na população; limiar e fator de inflação não mudam
depois de ver o resultado: qualquer mudança entra como "ajuste pós-resultado" no relatório).

1. Os eventos são os do backtest (`undercut_eval.find_events`, a mesma definição de evento, de
   resultado real, de evento primário e de "pista não verde"), nas 21 corridas secas, com as
   constantes SEM a corrida avaliada e a configuração congelada (`strategy/protocol.py`: limiar 0,6,
   fator de inflação 1,3, K de 3 a 10).
2. Em cada corrida, para cada volta N do líder de `FIRST_LAP` até a penúltima (o que o consumidor
   calcula), o contexto e os alertas de TODOS os pilotos vêm de `live.leader_lap_assessments`, a
   MESMA função que o consumidor chama, com o instante do líder (`crossing_ms`).
3. O alerta de um evento (o atacante A, que de fato parou na volta s; decisão na volta N = s - 1) é
   o resultado ao vivo EM VIGOR quando A cruza a linha da volta N: o da maior volta do líder cujo
   instante é menor ou igual ao do cruzamento de A (o que estava na tela quando a decisão foi
   tomada). Nele A está, em geral, uma volta mais velho que no backtest.
   SENSIBILIDADE declarada de antemão: o resultado SEGUINTE (a decisão tomada mais tarde na volta,
   depois de o líder completar a volta s), só se ele sai antes de A entrar nos boxes.
4. O par precisa ser o mesmo: o rival que o alerta ao vivo escolhe (o carro à frente de A no estado
   daquele instante) tem de ser o rival do evento. Senão o evento fica FORA, contado por motivo
   ("rival é o líder", "par diferente", "sem rival à frente no estado ao vivo", ...). É uma seleção
   a
   mais além do viés de só observar paradas que aconteceram, e está dita no relatório.
5. As métricas são as do backtest (`undercut_eval.summarize`: Brier contra a taxa-base, calibração
   com o n de cada faixa, precisão no limiar com Wilson, viés e erro da margem, cobertura), mais a
   comparação, nos MESMOS eventos, com a forma antiga (o cache do backtest). A precisão com a
   histerese (`AlertTracker` passando por todas as voltas do líder, como o consumidor) é secundária.
6. O fator de inflação que os dados pedem sai como diagnóstico e NÃO é aplicado.
"""

import argparse
import logging
import pickle
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from db.session import get_session_factory
from strategy.backtest import LIVE, TEMPO, RaceBundle, RaceOutcome, load_bundles, load_outcomes
from strategy.backtest_report import git_head, wilson
from strategy.forecast_eval import Interval, paired_bootstrap, verdict
from strategy.fuel import FuelVariant
from strategy.laps import LapRow
from strategy.live import FIRST_LAP, leader_lap_assessments
from strategy.priors import Priors
from strategy.protocol import FROZEN, frozen_undercut_config
from strategy.snapshot import crossing_ms, driver_crossing_ms
from strategy.tables import render_markdown
from strategy.undercut import (
    AlertTracker,
    Pair,
    UndercutAssessment,
    UndercutConfig,
)
from strategy.undercut_eval import (
    EventResult,
    Summary,
    UndercutEvent,
    brier_score,
    calibrate_inflation,
    find_events,
    is_neutralized,
    precision_at,
    realized_margin,
    summarize,
)

PRIMARY = "cruzamento"  # o resultado em vigor quando o atacante cruza a linha da volta N
LATE = "tarde"  # o resultado seguinte, antes de o atacante entrar nos boxes
DECISIONS = (PRIMARY, LATE)

NO_RESULT = "sem resultado ao vivo nesse instante"
NO_ATTACKER = "atacante fora do estado do líder"
RIVAL_IS_LEADER = "rival é o líder (já completou a volta; o atacante ainda não)"
NO_RIVAL = "sem rival à frente no estado ao vivo"
DIFFERENT_PAIR = "par diferente do evento"
NO_CLOCK = "sem relógio do atacante"


# ---------------------------------------------------------------- os resultados ao vivo de uma
# corrida


@dataclass(frozen=True, slots=True)
class LeaderLap:
    """O que o consumidor entregaria quando o líder completou a volta `lap`."""

    lap: int
    t_ms: int
    leader: str | None
    assessments: dict[str, UndercutAssessment]  # por atacante
    laps_done: dict[str, int]  # a volta em que cada carro estava no estado desse instante
    alerts: frozenset[Pair]  # os pares ligados pelo AlertTracker (com histerese), como ao vivo


def leader_laps(
    rows: Sequence[LapRow],
    total_laps: int,
    *,
    priors: Priors,
    config: UndercutConfig | None = None,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    first_lap: int = FIRST_LAP,
) -> dict[int, LeaderLap]:
    """Cada volta do líder que o consumidor calcula (`first_lap` até a penúltima), em ordem, com o
    `AlertTracker` passando por todas elas (a histerese depende da sequência)."""
    config = config or frozen_undercut_config()
    tracker = AlertTracker(config)
    out: dict[int, LeaderLap] = {}
    for lap in range(first_lap, total_laps):
        t = crossing_ms(rows, lap)
        if t is None:
            continue
        context, assessments = leader_lap_assessments(
            rows,
            lap=lap,
            total_laps=total_laps,
            t_ms=t,
            priors=priors,
            config=config,
            variant=variant,
        )
        alerts = tracker.update_all(assessments)
        leader = min(
            (r for r in rows if r.lap_number == lap and r.session_time_ms is not None),
            key=lambda r: r.session_time_ms,  # type: ignore[arg-type, return-value]
            default=None,
        )
        out[lap] = LeaderLap(
            lap,
            t,
            leader.driver if leader else None,
            {a.attacker: a for a in assessments},
            {driver: state.laps_done for driver, state in context.states.items()},
            frozenset(alerts),
        )
    return out


# ---------------------------------------------------------------- do evento ao resultado ao vivo


def result_in_effect(crossings: dict[int, int], t_ms: int) -> int | None:
    """A maior volta do líder cujo instante é menor ou igual a `t_ms`: o resultado que estava na
    tela nesse instante (None se ainda não havia nenhum)."""
    laps = [lap for lap, t in crossings.items() if t <= t_ms]
    return max(laps, default=None)


def next_result(crossings: dict[int, int], t_ms: int, deadline_ms: int) -> int | None:
    """O primeiro resultado DEPOIS de `t_ms` que sai até `deadline_ms` (a entrada do atacante nos
    boxes): a decisão tomada mais tarde na volta."""
    laps = [lap for lap, t in crossings.items() if t_ms < t <= deadline_ms]
    return min(laps, default=None)


@dataclass(frozen=True, slots=True)
class LiveEvent:
    """Um evento do backtest visto pela forma ao vivo."""

    event: UndercutEvent
    decision: str
    result: EventResult | None  # None = o evento fica fora (ver `reason`)
    reason: str | None
    leader_lap: int | None = None
    staleness: int | None = None  # voltas que o atacante estava atrás da volta da decisão
    tracker_alert: bool = False  # o AlertTracker tinha este par ligado nesse resultado


def map_event(
    event: UndercutEvent,
    rows: Sequence[LapRow],
    laps: dict[int, LeaderLap],
    crossings: dict[int, int],
    *,
    decision: str,
    inflation: float,
) -> LiveEvent:
    def out(reason: str, lap: int | None = None) -> LiveEvent:
        return LiveEvent(event, decision, None, reason, lap)

    t_dec = driver_crossing_ms(rows, event.attacker, event.lap)
    if t_dec is None:
        return out(NO_CLOCK)
    if decision == PRIMARY:
        lap = result_in_effect(crossings, t_dec)
    else:
        deadline = driver_crossing_ms(rows, event.attacker, event.lap + 1)
        lap = None if deadline is None else next_result(crossings, t_dec, deadline)
    if lap is None:
        return out(NO_RESULT)
    live = laps[lap]
    assessment = live.assessments.get(event.attacker)
    if assessment is None:
        return out(NO_ATTACKER, lap)
    if assessment.rival != event.rival:
        if event.rival == live.leader:
            return out(RIVAL_IS_LEADER, lap)
        return out(NO_RIVAL if assessment.rival is None else DIFFERENT_PAIR, lap)
    realized = realized_margin(rows, event)
    if realized is None:
        return out(NO_CLOCK, lap)
    result = EventResult(event, assessment, realized, is_neutralized(rows, event), inflation)
    return LiveEvent(
        event,
        decision,
        result,
        None,
        lap,
        event.lap - live.laps_done.get(event.attacker, event.lap),
        (event.attacker, event.rival) in live.alerts,
    )


@dataclass(slots=True)
class RaceLive:
    race: str
    events: dict[str, list[LiveEvent]] = field(default_factory=dict)  # por decisão
    skipped: dict[str, int] = field(default_factory=dict)  # os descartes da busca de eventos
    n_leader_laps: int = 0


def evaluate_race_live(
    rows: Sequence[LapRow],
    total_laps: int,
    *,
    race: str,
    priors: Priors,
    config: UndercutConfig | None = None,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
) -> RaceLive:
    config = config or frozen_undercut_config()
    events, skipped = find_events(rows, total_laps, race=race, config=config)
    laps = leader_laps(rows, total_laps, priors=priors, config=config, variant=variant)
    crossings = {lap: live.t_ms for lap, live in laps.items()}
    return RaceLive(
        race,
        {
            decision: [
                map_event(
                    e, rows, laps, crossings, decision=decision, inflation=config.sd_inflation
                )
                for e in events
            ]
            for decision in DECISIONS
        },
        dict(skipped),
        len(laps),
    )


def evaluate_bundle(bundle: RaceBundle) -> RaceLive:
    """Uma corrida por processo (o `RaceBundle` é o do backtest, com as constantes sem ela)."""
    return evaluate_race_live(
        bundle.rows,
        bundle.ref.total_laps,
        race=bundle.ref.label,
        priors=bundle.priors,
        config=frozen_undercut_config(FROZEN),
        variant=FuelVariant(FROZEN.headline_variant),
    )


# ---------------------------------------------------------------- o relatório


def _f(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _precision(probs: Sequence[float], wins: Sequence[bool], threshold: float) -> str:
    n, precision = precision_at(probs, wins, threshold)
    if n == 0 or precision is None:
        return "0 alertas"
    hits = round(precision * n)
    low, high = wilson(hits, n) or (0.0, 1.0)
    return f"{hits} de {n} ({precision:.0%}, Wilson {low:.0%} a {high:.0%})"


def primary_results(events: Sequence[LiveEvent]) -> list[EventResult]:
    return [e.result for e in events if e.result is not None]


def old_form(outcomes: Sequence[RaceOutcome]) -> dict[tuple[str, str, int], EventResult]:
    """Os resultados da forma antiga (instante do atacante), por (corrida, atacante, volta)."""
    return {
        (r.event.race, r.event.attacker, r.event.lap): r
        for o in outcomes
        if o.main
        for r in o.undercut.get((LIVE, TEMPO), [])
    }


def scored(results: Sequence[EventResult], threshold: float) -> tuple[list[float], list[bool]]:
    probs, wins = [], []
    for r in results:
        if r.primary and (p := r.probability()) is not None:
            probs.append(p)
            wins.append(r.success)
    return probs, wins


def section_mapping(by_race: Sequence[RaceLive]) -> str:
    primary = [e for race in by_race for e in race.events[PRIMARY]]
    late = [e for race in by_race for e in race.events[LATE]]
    total = len(primary)

    def fates(events: Sequence[LiveEvent]) -> list[list[str]]:
        why = Counter(e.reason for e in events if e.reason)
        mapped = sum(1 for e in events if e.result is not None)
        rows = [["destino do evento", "eventos"], ["no par do evento (avaliado)", str(mapped)]]
        rows += [[reason, str(n)] for reason, n in why.most_common()]
        return rows

    stale = Counter(e.staleness for e in primary if e.result is not None)
    stale_late = Counter(e.staleness for e in late if e.result is not None)
    return (
        f"Eventos do backtest (a mesma busca e a mesma definição): **{total}** nas "
        f"{len(by_race)} corridas secas. Cada um virou uma decisão ao vivo de duas formas.\n\n"
        "**Primária** (o resultado em vigor quando o atacante cruza a linha da volta N):\n\n"
        f"{render_markdown(fates(primary))}\n\n"
        "**Sensibilidade** (o resultado seguinte, antes de o atacante entrar nos boxes):\n\n"
        f"{render_markdown(fates(late))}\n\n"
        "Quantas voltas o atacante estava atrás da volta da decisão no estado ao vivo (primária: "
        + ", ".join(f"{k} volta(s): {v}" for k, v in sorted(stale.items()))
        + "; sensibilidade: "
        + ", ".join(f"{k} volta(s): {v}" for k, v in sorted(stale_late.items()))
        + ")."
    )


def summary_row(label: str, results: Sequence[EventResult], threshold: float) -> list[str]:
    s: Summary = summarize(results, threshold=threshold)
    probs, wins = scored(results, threshold)
    return [
        label,
        str(len(results)),
        str(s.n_primary),
        _f(s.base_rate),
        _f(s.mean_predicted),
        _f(s.brier, 3),
        _f(s.brier_baseline, 3),
        _precision(probs, wins, threshold),
        _f(s.margin_bias),
        _f(s.margin_mae),
        _f(s.coverage_90),
    ]


SUMMARY_HEADER = [
    "conjunto", "eventos", "primários", "taxa de sucesso", "prob. média", "Brier",
    "Brier (taxa-base)", "alertas no limiar: acertos (Wilson)", "viés da margem (s)",
    "erro abs. (s)", "cobertura 90%",
]  # fmt: skip


def reliability_rows(results: Sequence[EventResult], threshold: float) -> list[list[str]]:
    s = summarize(results, threshold=threshold)
    rows = [["faixa de probabilidade", "eventos (n)", "prevista (média)", "observada"]]
    for b in s.bins:
        rows.append(
            [f"{b.low:.1f} a {b.high:.1f}", str(b.n), _f(b.mean_predicted), _f(b.observed_rate)]
        )
    return rows


def common_pairs(
    by_race: Sequence[RaceLive], old: dict[tuple[str, str, int], EventResult]
) -> list[tuple[EventResult, EventResult]]:
    """(forma antiga, forma ao vivo) dos eventos que são primários nas DUAS: a mesma parada, o mesmo
    rival e o mesmo resultado real."""
    pairs = []
    for race in by_race:
        for e in race.events[PRIMARY]:
            if e.result is None or not e.result.primary:
                continue
            o = old.get((e.event.race, e.event.attacker, e.event.lap))
            if o is not None and o.primary:
                pairs.append((o, e.result))
    return pairs


def brier_difference(pairs: Sequence[tuple[EventResult, EventResult]]) -> Interval | None:
    """O Brier da forma ao vivo menos o da antiga, nos mesmos eventos, com o intervalo de 95% por
    bootstrap sobre as CORRIDAS (os eventos da mesma corrida não são independentes). Negativo = a
    forma ao vivo erra menos."""
    live: dict[str, list[float]] = {}
    antiga: dict[str, list[float]] = {}
    for o, n in pairs:
        po, pn = o.probability(), n.probability()
        if po is None or pn is None:
            continue
        antiga.setdefault(o.event.race, []).append((po - o.success) ** 2)
        live.setdefault(o.event.race, []).append((pn - n.success) ** 2)
    return paired_bootstrap(
        {race: float(np.mean(v)) for race, v in live.items()},
        {race: float(np.mean(v)) for race, v in antiga.items()},
    )


def _interval(interval: Interval | None) -> str:
    if interval is None:
        return "sem dados"
    return (
        f"{interval.mean:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}] "
        f"({interval.n_races} corridas)"
    )


def section_comparison(
    by_race: Sequence[RaceLive], old: dict[tuple[str, str, int], EventResult], threshold: float
) -> str:
    """Nos MESMOS eventos (primários nas duas formas): forma antiga contra forma ao vivo."""
    pairs = common_pairs(by_race, old)
    if not pairs:
        return "Nenhum evento primário nas duas formas: sem comparação."
    olds, lives = [p[0] for p in pairs], [p[1] for p in pairs]
    po, wo = scored(olds, threshold)
    pl, wl = scored(lives, threshold)
    both = sum(1 for a, b in zip(po, pl, strict=True) if a >= threshold and b >= threshold)
    only_old = sum(1 for a, b in zip(po, pl, strict=True) if a >= threshold > b)
    only_live = sum(1 for a, b in zip(po, pl, strict=True) if b >= threshold > a)
    gap = float(np.mean([abs(a - b) for a, b in zip(po, pl, strict=True)]))
    difference = brier_difference(pairs)
    table = [
        ["forma", "eventos", "Brier", "alertas no limiar: acertos (Wilson)"],
        [
            "antiga (instante do atacante)",
            str(len(olds)),
            _f(brier_score(po, wo), 3),
            _precision(po, wo, threshold),
        ],
        [
            "ao vivo (instante do líder)",
            str(len(lives)),
            _f(brier_score(pl, wl), 3),
            _precision(pl, wl, threshold),
        ],
    ]
    return (
        f"Os **{len(pairs)}** eventos que são primários nas duas formas (a mesma parada, o mesmo "
        "rival, o mesmo resultado real):\n\n"
        f"{render_markdown(table)}\n\n"
        f"Alertas no limiar: nas duas formas **{both}**, só na antiga **{only_old}**, só na ao "
        f"vivo **{only_live}**. Diferença absoluta média entre as duas probabilidades: "
        f"{gap:.3f}.\n\n"
        f"Brier da forma ao vivo menos o da antiga, com bootstrap sobre as corridas: "
        f"**{_interval(difference)}**. Pela regra do backtest: **{_verdict(difference)}**."
    )


def _verdict(difference: Interval | None) -> str:
    """`forecast_eval.verdict` lê "supera / é pior / não supera"; aqui a pergunta é se a forma ao
    vivo erra menos que a antiga."""
    return {
        "supera": "a forma ao vivo erra menos que a antiga",
        "é pior": "a forma ao vivo erra mais que a antiga",
        "não supera": "não se distingue da antiga nesta amostra",
        "sem dados": "sem dados",
    }[verdict(difference)]


def section_hysteresis(by_race: Sequence[RaceLive], threshold: float) -> str:
    events = [e for race in by_race for e in race.events[PRIMARY]]
    hits = [e for e in events if e.result is not None and e.result.primary and e.tracker_alert]
    n = len(hits)
    wins = sum(1 for e in hits if e.result is not None and e.result.success)
    low, high = wilson(wins, n) or (0.0, 1.0)
    return (
        "O que o consumidor de fato publicaria é o alerta do `AlertTracker` (limiar "
        f"{threshold} para ligar, 0,5 para desligar, passando por todas as voltas do líder). Nos "
        f"eventos primários da decisão primária, o par estava **ligado** em **{n}**; deram certo "
        f"**{wins}**" + (f" ({wins / n:.0%}, Wilson {low:.0%} a {high:.0%})." if n else ".")
    )


def section_per_race(by_race: Sequence[RaceLive], threshold: float) -> str:
    rows = [
        [
            "corrida",
            "voltas do líder avaliadas",
            "eventos",
            "avaliados",
            "primários",
            "taxa de sucesso",
        ]
    ]
    for race in by_race:
        events = race.events[PRIMARY]
        results = primary_results(events)
        s = summarize(results, threshold=threshold)
        rows.append(
            [
                race.race,
                str(race.n_leader_laps),
                str(len(events)),
                str(len(results)),
                str(s.n_primary),
                _f(s.base_rate),
            ]
        )
    return render_markdown(rows)


def section_summary(
    by_race: Sequence[RaceLive], old: dict[tuple[str, str, int], EventResult], threshold: float
) -> str:
    """O resumo, gerado dos mesmos números das seções (a redação não pode divergir deles)."""
    primary = [e for race in by_race for e in race.events[PRIMARY]]
    main = primary_results(primary)
    summary = summarize(main, threshold=threshold)
    probs, wins = scored(main, threshold)
    why = Counter(e.reason for e in primary if e.reason)
    lines = [
        f"- **{len(main)} de {len(primary)} eventos** viraram uma decisão ao vivo no par do "
        "evento; "
        + ", ".join(f'{n} ficaram fora por "{reason}"' for reason, n in why.most_common())
        + ". **Nenhum undercut contra o líder é avaliado nem alertado ao vivo.**",
        f"- Dos **{summary.n_primary}** eventos primários, os alertas no limiar acertaram "
        f"{_precision(probs, wins, threshold)}; Brier {_f(summary.brier, 3)} contra "
        f"{_f(summary.brier_baseline, 3)} "
        "de quem prevê sempre a taxa-base. **A precisão é otimista** (só as paradas que "
        "aconteceram) e a amostra é pequena.",
        f"- A margem: viés de {_f(summary.margin_bias)} s, erro absoluto de "
        f"{_f(summary.margin_mae)} s e o intervalo de 90% cobre {_f(summary.coverage_90)} (o "
        f"congelado é 1,3; os dados "
        f"pedem {_f(calibrate_inflation(main).raw_factor)}: **não aplicado**).",
    ]
    pairs = common_pairs(by_race, old)
    if pairs:
        difference = brier_difference(pairs)
        lines.append(
            f"- **Contra a forma antiga, nos mesmos {len(pairs)} eventos:** Brier da ao vivo "
            f"menos o da antiga {_interval(difference)}: {_verdict(difference)}."
        )
    return "\n".join(lines)


CAVEATS = """\
- **Só paradas que aconteceram, e as equipes tendem a parar quando funciona.** A taxa de sucesso
  daqui é mais alta que a de uma situação qualquer de atacante atrás de rival: a **precisão é
  otimista**, a **revocação não é mensurável** e a calibração é **condicional** a "a equipe decidiu
  parar".
- **Uma seleção a mais.** Só entram os eventos em que o par ao vivo é o do evento. Os que ficam fora
  (principalmente o líder como rival) não são sorteados: o líder, por exemplo, nunca é o "carro à
  frente" de ninguém na forma ao vivo, então **nenhum undercut contra o líder é avaliado nem
  alertado**.
- **Amostra pequena e dependente.** São dezenas de eventos em 21 corridas, e os eventos da mesma
  corrida não são independentes (o intervalo de Wilson os trata como se fossem). Faixas com poucos
  eventos não dizem nada sobre calibração: o `n` de cada uma está na tabela.
- **O resultado real depende do que os dois lados fizeram depois.** Eventos com parada extra na
  janela são descartados; os de pista não verde ficam fora dos primários.
- **Nada foi ajustado.** O limiar (0,6 liga, 0,5 desliga) e o fator de inflação (1,3) são os
  congelados antes do backtest. O fator que os dados pedem sai só como diagnóstico.
"""


def build_report(
    by_race: Sequence[RaceLive],
    old: dict[tuple[str, str, int], EventResult],
    *,
    threshold: float | None = None,
    adjustments: Sequence[str] = (),
    notes: Sequence[str] = (),
    generated_on: date | None = None,
) -> str:
    threshold = FROZEN.alert_threshold if threshold is None else threshold
    day = (generated_on or date.today()).isoformat()
    primary = [e for race in by_race for e in race.events[PRIMARY]]
    late = [e for race in by_race for e in race.events[LATE]]
    main, sens = primary_results(primary), primary_results(late)
    fit = calibrate_inflation(main)
    table = [
        SUMMARY_HEADER,
        summary_row("ao vivo, decisão no cruzamento (primária)", main, threshold),
        summary_row("ao vivo, decisão mais tarde na volta (sensibilidade)", sens, threshold),
    ]
    adjusted = (
        "\n".join(f"- {a}" for a in adjustments)
        if adjustments
        else "Nenhum. O limiar e o fator de inflação são os congelados antes do backtest."
    )
    return (
        "# O alerta de undercut na forma ao vivo\n\n"
        f"Gerado por `python -m strategy.undercut_live` em {day} sobre o commit `{git_head()}`. "
        "O protocolo está no docstring de `strategy/undercut_live.py` e foi commitado **antes** de "
        "rodar na população.\n\n"
        "## O que é isto\n\n"
        "O backtest (`docs/backtest.md`) mediu o alerta de undercut no instante em que o "
        "**atacante** completa a volta: a **forma antiga**. O consumidor de estratégia o calcula "
        "no instante em que o **líder** completa a volta, com um contexto só por volta: a **forma "
        "ao vivo**. São formas diferentes, e **os números do backtest não valem para a ao vivo**. "
        "Este relatório mede a ao vivo, offline, nas mesmas paradas reais.\n\n"
        "Duas diferenças estruturais, conhecidas antes de medir: (1) quem está atrás do líder "
        "aparece com a volta anterior; (2) o líder nunca é o carro à frente de ninguém "
        "(`car_ahead` só compara carros na mesma volta, e no instante em que o líder completa a "
        "volta N os outros estão na N-1).\n\n"
        "## Resumo\n\n"
        f"{section_summary(by_race, old, threshold)}\n\n"
        "## Como cada parada real virou uma decisão ao vivo\n\n"
        f"{section_mapping(by_race)}\n\n"
        "## Resultado\n\n"
        f"{render_markdown(table)}\n\n"
        "Calibração da decisão primária, com o `n` de cada faixa (eventos, não corridas):\n\n"
        f"{render_markdown(reliability_rows(main, threshold))}\n\n"
        f"Fator de inflação que os dados pedem para 90% de cobertura da margem: "
        f"{_f(fit.raw_factor)} (n={fit.n}); em uso, o congelado "
        f"({_f(FROZEN.undercut_sd_inflation)}). "
        "**Diagnóstico, não aplicado.**\n\n"
        "## Com a histerese do consumidor\n\n"
        f"{section_hysteresis(by_race, threshold)}\n\n"
        "## Contra a forma antiga, nos mesmos eventos\n\n"
        f"{section_comparison(by_race, old, threshold)}\n\n"
        "## Por corrida\n\n"
        f"{section_per_race(by_race, threshold)}\n\n"
        "## Ressalvas\n\n"
        f"{CAVEATS}\n"
        "## Ajustes pós-resultado\n\n"
        f"{adjusted}\n"
        + (
            "\n**Análises e acréscimos feitos depois do resultado, que NÃO alteram o protocolo nem "
            "o que foi medido:**\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
            if notes
            else ""
        )
    )


# ---------------------------------------------------------------- a linha de comando


def run(bundles: Sequence[RaceBundle], workers: int | None = None) -> list[RaceLive]:
    import os

    n = max(1, min(workers or (os.cpu_count() or 2) - 1, len(bundles)))
    if n == 1:
        return [evaluate_bundle(b) for b in bundles]
    with ProcessPoolExecutor(max_workers=n) as pool:
        return list(pool.map(evaluate_bundle, bundles))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.undercut_live",
        description="Avalia, offline, o alerta de undercut na forma ao vivo (instante do líder).",
    )
    parser.add_argument(
        "--backtest-cache", type=Path, required=True, help="o cache do backtest (.pkl)"
    )
    parser.add_argument("--cache", type=Path, default=None, help="guarda/lê os resultados (.pkl)")
    parser.add_argument("--reuse", action="store_true", help="lê o cache em vez de rodar de novo")
    parser.add_argument("--markdown", type=Path, default=None, help="grava o relatório (.md)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--adjustment", action="append", default=[], help="um ajuste pós-resultado")
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="análise ou acréscimo feito depois do resultado, sem alterar o protocolo (repetível)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.reuse:
        if args.cache is None:
            parser.error("--reuse precisa de --cache")
        with args.cache.open("rb") as handle:
            by_race = pickle.load(handle)  # noqa: S301 (um arquivo que este módulo mesmo gravou)
    else:
        with get_session_factory()() as session:
            bundles = [b for b in load_bundles(session) if b.main]
        by_race = run(bundles, args.workers)
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with args.cache.open("wb") as handle:
                pickle.dump(by_race, handle)
    report = build_report(
        by_race,
        old_form(load_outcomes(args.backtest_cache)),
        adjustments=tuple(args.adjustment),
        notes=tuple(args.note),
    )
    print(report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    from strategy.undercut_live import main as _main

    sys.exit(_main())
