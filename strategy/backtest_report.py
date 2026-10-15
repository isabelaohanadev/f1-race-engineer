"""O relatório do backtest (`docs/backtest.md`), gerado dos resultados de `strategy.backtest`.

O TEXTO SEGUE OS NÚMEROS. As afirmações (supera, não supera, é pior, melhor, sem diferença
detectável)
saem de regras fixadas no protocolo congelado e aplicadas aos intervalos: ninguém as escolhe depois
de
ver o resultado. Quando o Monte Carlo não supera "o líder atual vence" numa faixa de andamento, o
relatório diz isso, faixa a faixa, mesmo que o resultado seja ruim.
"""

import dataclasses
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date

from strategy.backtest import LIVE, ORACLE, TEMPO, VOLTA, RaceOutcome
from strategy.forecast_eval import (
    CalibrationBin,
    Interval,
    LapScore,
    ProgressRow,
    brier_by_progress,
    calibration_curve,
    cluster_bootstrap,
    paired_bootstrap,
    per_race_mean,
    progress_rows,
    verdict,
)
from strategy.laptime_eval import LapTimeRecord, per_race
from strategy.protocol import FREEZE_BASE_COMMIT, FROZEN, Protocol
from strategy.tables import render_markdown
from strategy.undercut_eval import (
    EventResult,
    calibrate_inflation,
    loo_inflation,
    reliability_table,
    summarize,
    summary_table,
)

FREEZE_COMMIT = "3be1004"  # o commit do documento do protocolo, anterior a qualquer execução


# ---------------------------------------------------------------- formatação


def fmt(interval: Interval | None, digits: int = 3) -> str:
    if interval is None:
        return "-"
    if interval.high - interval.low < 5 * 10 ** -(digits + 1):  # um ponto, não um intervalo
        return f"{interval.mean:.{digits}f}"
    return f"{interval.mean:.{digits}f} [{interval.low:.{digits}f}, {interval.high:.{digits}f}]"


def num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def pct(interval: Interval | None) -> str:
    if interval is None:
        return "-"
    if interval.high - interval.low < 0.005:
        return f"{interval.mean:.0%}"
    return f"{interval.mean:.0%} [{interval.low:.0%}, {interval.high:.0%}]"


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """O intervalo de Wilson de uma proporção (para as poucas dezenas de alertas)."""
    if n == 0:
        return None
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def _split(outcomes: Sequence[RaceOutcome]) -> tuple[list[RaceOutcome], list[RaceOutcome]]:
    return [o for o in outcomes if o.main], [o for o in outcomes if not o.main]


def _scores(outcomes: Sequence[RaceOutcome], variant: str, snapshot: str, mult: float = 1.0):
    return [s for o in outcomes for s in o.forecast.get((variant, snapshot, mult), [])]


def _records(outcomes: Sequence[RaceOutcome], variant: str, snapshot: str) -> list[LapTimeRecord]:
    return [r for o in outcomes for r in o.laptime.get((variant, snapshot), [])]


def _events(
    outcomes: Sequence[RaceOutcome], mode: str, snapshot: str
) -> dict[str, list[EventResult]]:
    return {o.label: o.undercut.get((mode, snapshot), []) for o in outcomes}


def pooled_mae(records: Sequence[LapTimeRecord]) -> dict[str, float]:
    """O erro absoluto por corrida, juntando os horizontes (cada volta pesa pelos pilotos dela)."""
    total: dict[str, float] = defaultdict(float)
    weight: dict[str, int] = defaultdict(int)
    for r in records:
        total[r.race] += r.mae * r.n
        weight[r.race] += r.n
    return {race: total[race] / weight[race] for race in total if weight[race]}


def overall_brier(scores: Sequence[LapScore], attribute: str = "brier_win") -> dict[str, float]:
    return per_race_mean(scores, attribute)


# ---------------------------------------------------------------- as seções


def section_frozen(protocol: Protocol) -> str:
    rows = [["parâmetro", "valor"]]
    for field in dataclasses.fields(protocol):
        value = getattr(protocol, field.name)
        rows.append([f"`{field.name}`", str(value)])
    return (
        f"Congelado no commit `{FREEZE_COMMIT}` (`docs/backtest-protocolo.md`), sobre o código do "
        f"commit "
        f"`{FREEZE_BASE_COMMIT}`, **antes** de qualquer execução na população. Valores de "
        f"`strategy/protocol.py`:\n\n{render_markdown(rows)}\n"
    )


def section_laptime(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    rows = [
        [
            "variante de k",
            "horizonte",
            "viés (s)",
            "erro absoluto (s)",
            "persistência: erro abs. (s)",
            "modelo menos persistência",
            "cobertura do intervalo de 90%",
        ]
    ]
    beats: dict[str, list[int]] = defaultdict(list)
    for variant in protocol.variants:
        records = _records(main, variant, TEMPO)
        for h in protocol.horizons:
            mae = per_race(records, h, "mae")
            baseline = per_race(records, h, "baseline_mae")
            difference = paired_bootstrap(mae, baseline, seed=protocol.bootstrap_seed)
            if verdict(difference) == "supera":
                beats[variant].append(h)
            rows.append(
                [
                    variant,
                    str(h),
                    fmt(
                        cluster_bootstrap(
                            per_race(records, h, "bias"), seed=protocol.bootstrap_seed
                        )
                    ),
                    fmt(cluster_bootstrap(mae, seed=protocol.bootstrap_seed)),
                    fmt(cluster_bootstrap(baseline, seed=protocol.bootstrap_seed)),
                    fmt(difference),
                    pct(
                        cluster_bootstrap(
                            per_race(records, h, "coverage"), seed=protocol.bootstrap_seed
                        )
                    ),
                ]
            )
    headline = protocol.headline_variant
    won = beats.get(headline, [])
    text = (
        f"Voltas limpas, de seco e do mesmo stint, no instante em que o líder completa a volta N, "
        f"para os "
        f"horizontes de {protocol.horizons[0]} a {protocol.horizons[-1]} voltas à frente de cada "
        f"piloto. "
        f"Cada célula é a média por corrida com o intervalo de 95% por bootstrap sobre corridas. "
        f'"Modelo menos persistência" é pareado por corrida (negativo = o modelo erra menos).\n\n'
        f"{render_markdown(rows)}\n"
    )
    if len(won) == len(protocol.horizons):
        text += (
            f"\n**Na variante principal ({headline}) o modelo erra menos que a persistência nos "
            "5 horizontes** (intervalo inteiro abaixo de zero).\n"
        )
    elif won:
        text += (
            f"\n**Na variante principal ({headline}) o modelo só supera a persistência nos "
            f"horizontes "
            f"{', '.join(map(str, won))}**; nos demais o intervalo da diferença contém o zero ou "
            f"fica acima.\n"
        )
    else:
        text += (
            f"\n**Na variante principal ({headline}) o modelo NÃO supera a persistência em nenhum "
            f"horizonte** "
            f"(o intervalo da diferença contém o zero ou fica acima).\n"
        )
    return text


def verdict_summary(rows: Sequence[ProgressRow], comparator: str, getter) -> str:
    """As faixas por veredito, com todas as letras."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.n_races:
            groups[getter(row)].append(f"{row.low:.0%}-{row.high:.0%}")
    total = sum(len(v) for v in groups.values())
    parts = []
    for label, headline in (
        ("supera", "SUPERA"),
        ("não supera", "NÃO SUPERA"),
        ("é pior", "É PIOR"),
    ):
        if groups.get(label):
            parts.append(
                f"{headline} em {len(groups[label])} de {total} faixas ({', '.join(groups[label])})"
            )
    if not parts:
        return f"Sem dados para comparar com {comparator}."
    return (
        f"Contra **{comparator}**, o Monte Carlo " + "; ".join(parts) + ". "
        "SUPERA = o intervalo de 95% da diferença pareada de Brier fica inteiro abaixo de zero; "
        "NÃO SUPERA = o intervalo contém o zero (não se distingue do comparador nesta amostra); "
        "É PIOR = fica inteiro acima."
    )


def section_brier(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    scores = _scores(main, protocol.headline_variant, TEMPO)
    rows = progress_rows(scores, n_boot=protocol.bootstrap_resamples)
    table = [
        [
            "andamento",
            "corridas",
            "Monte Carlo",
            "o líder vence",
            "todos iguais",
            "MC − líder",
            "contra o líder",
            "MC − iguais",
            "contra todos iguais",
            "pódio (MC)",
        ]
    ]
    for r in rows:
        table.append(
            [
                f"{r.low:.0%} a {r.high:.0%}",
                str(r.n_races),
                fmt(r.model),
                fmt(r.leader),
                fmt(r.uniform),
                fmt(r.versus_leader),
                r.verdict_leader,
                fmt(r.versus_uniform),
                r.verdict_uniform,
                fmt(r.podium),
            ]
        )
    model, leader = overall_brier(scores), overall_brier(scores, "leader_brier")
    uniform = overall_brier(scores, "uniform_brier")
    overall = [
        ["", "Brier da vitória por corrida (todas as voltas avaliadas)"],
        ["Monte Carlo", fmt(cluster_bootstrap(model))],
        ["o líder atual vence", fmt(cluster_bootstrap(leader))],
        ["todos iguais", fmt(cluster_bootstrap(uniform))],
        ["Monte Carlo menos o líder (pareado)", fmt(paired_bootstrap(model, leader))],
        ["Monte Carlo menos todos iguais (pareado)", fmt(paired_bootstrap(model, uniform))],
    ]
    favorite = overall_brier(scores, "favorite_won")
    return (
        "O Brier da vitória é a soma, sobre os pilotos, de (p - y)^2: 0 é perfeito e 2, o pior. "
        "Brier por "
        "faixa de andamento da corrida; a unidade do intervalo é a **corrida**, não a volta.\n\n"
        f"{render_markdown(table)}\n\n"
        f"{verdict_summary(rows, 'o líder atual vence', lambda r: r.verdict_leader)}\n\n"
        f"{verdict_summary(rows, 'todos iguais', lambda r: r.verdict_uniform)}\n\n"
        f"{render_markdown(overall)}\n\n"
        f"O favorito do Monte Carlo ganhou em {fmt(cluster_bootstrap(favorite), 2)} das previsões "
        f"(média por corrida).\n"
    )


def calibration_rows(bins: Sequence[CalibrationBin]) -> list[list[str]]:
    rows = [
        [
            "faixa de probabilidade",
            "pares (piloto, volta)",
            "corridas",
            "prevista (média)",
            "observada [IC 95%]",
        ]
    ]
    for b in bins:
        rows.append(
            [
                f"{b.low:.2f} a {b.high:.2f}",
                str(b.n),
                str(b.n_races),
                num(b.mean_predicted),
                pct(b.observed),
            ]
        )
    return rows


def section_calibration(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    scores = _scores(main, protocol.headline_variant, TEMPO)
    win = calibration_curve(scores, "win", n_boot=protocol.bootstrap_resamples)
    podium = calibration_curve(scores, "podium", n_boot=protocol.bootstrap_resamples)
    return (
        "Probabilidade prevista contra a frequência observada, **com o n de cada faixa**. Os "
        "pares de uma "
        "mesma corrida e volta não são independentes (as probabilidades somam 1) nem as voltas "
        "entre si; "
        'por isso a coluna "corridas" é o tamanho efetivo e o intervalo da frequência ressorteia '
        "corridas. Uma faixa com poucas corridas não diz nada sobre calibração, qualquer que seja "
        "a "
        "frequência observada.\n\n"
        f"**Vitória**\n\n{render_markdown(calibration_rows(win))}\n\n"
        f"**Pódio**\n\n{render_markdown(calibration_rows(podium))}\n"
    )


def section_variants(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    headline = protocol.headline_variant
    base_brier = overall_brier(_scores(main, headline, TEMPO))
    base_mae = pooled_mae(_records(main, headline, TEMPO))
    rows = [
        [
            "variante",
            "Brier da vitória",
            "erro de tempo de volta (s)",
            "Brier − principal",
            "erro − principal",
            "conclusão",
        ]
    ]
    lines = []
    for variant in protocol.variants:
        brier = overall_brier(_scores(main, variant, TEMPO))
        mae = pooled_mae(_records(main, variant, TEMPO))
        if variant == headline:
            rows.append(
                [
                    f"{variant} (principal)",
                    fmt(cluster_bootstrap(brier)),
                    fmt(cluster_bootstrap(mae)),
                    "-",
                    "-",
                    "-",
                ]
            )
            continue
        d_brier = paired_bootstrap(brier, base_brier, seed=protocol.bootstrap_seed)
        d_mae = paired_bootstrap(mae, base_mae, seed=protocol.bootstrap_seed)
        better = d_brier is not None and d_mae is not None and d_brier.high < 0 and d_mae.high < 0
        worse = d_brier is not None and d_mae is not None and d_brier.low > 0 and d_mae.low > 0
        conclusion = (
            "melhor que o principal"
            if better
            else "pior que o principal"
            if worse
            else "sem diferença detectável"
        )
        rows.append(
            [
                variant,
                fmt(cluster_bootstrap(brier)),
                fmt(cluster_bootstrap(mae)),
                fmt(d_brier),
                fmt(d_mae),
                conclusion,
            ]
        )
        lines.append(f"- `{variant}`: {conclusion}.")
    return (
        "A regra do protocolo: uma variante só é declarada melhor se o intervalo de 95% da "
        "diferença "
        "pareada por corrida exclui o zero, **no mesmo sentido**, no erro de tempo de volta **e** "
        "no "
        'Brier da vitória; do contrário, "sem diferença detectável" e o principal permanece. '
        "Diferença "
        "negativa = a variante erra menos que o principal.\n\n"
        f"{render_markdown(rows)}\n\n" + "\n".join(lines) + "\n"
    )


def section_sensitivity(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    headline = protocol.headline_variant
    by_mult = {m: _scores(main, headline, TEMPO, m) for m in protocol.sc_multipliers}
    bins = 5
    per_bin = {
        m: brier_by_progress(
            by_mult[m], "brier_win", bins=bins, n_boot=protocol.bootstrap_resamples
        )
        for m in by_mult
    }
    rows = [["andamento"] + [f"taxa de SC {m:g}x" for m in protocol.sc_multipliers]]
    for i in range(bins):
        lo, hi, _ = per_bin[protocol.sc_multipliers[0]][i]
        rows.append(
            [f"{lo:.0%} a {hi:.0%}"] + [fmt(per_bin[m][i][2]) for m in protocol.sc_multipliers]
        )
    overall = {m: overall_brier(s) for m, s in by_mult.items()}
    rows.append(
        ["todas as voltas"] + [fmt(cluster_bootstrap(overall[m])) for m in protocol.sc_multipliers]
    )
    base = 1.0 if 1.0 in overall else protocol.sc_multipliers[0]
    diffs = []
    for m in protocol.sc_multipliers:
        if m != base:
            diffs.append(
                f"- taxa {m:g}x menos 1x (Brier, pareado): "
                f"{fmt(paired_bootstrap(overall[m], overall[base]))}"
            )
    return (
        "A mesma simulação (os **mesmos números aleatórios**) com a taxa de SC multiplicada. O "
        "Brier por "
        "faixa de andamento (5 faixas) e o geral:\n\n"
        f"{render_markdown(rows)}\n\n" + "\n".join(diffs) + "\n\n"
        "Se os intervalos se sobrepõem e a diferença pareada contém o zero, o resultado é "
        "**insensível** à "
        "taxa de SC nesta amostra (com 8 períodos de SC em 21 corridas, é o que se espera: a "
        "maioria das "
        "corridas não tem SC depois da volta 10).\n"
    )


def undercut_block(
    outcomes: Sequence[RaceOutcome], mode: str, snapshot: str, protocol: Protocol
) -> str:
    by_race = _events(outcomes, mode, snapshot)
    flat = [r for results in by_race.values() for r in results]
    summary = summarize(flat, threshold=protocol.alert_threshold)
    hits = round((summary.precision or 0.0) * summary.alerts) if summary.alerts else 0
    interval = wilson(hits, summary.alerts)
    precision = (
        f"{summary.alerts} alertas, {hits} deram certo "
        + (
            f"(precisão {hits / summary.alerts:.0%}, intervalo de Wilson {interval[0]:.0%} a "
            f"{interval[1]:.0%})"
            if interval
            else ""
        )
        if summary.alerts
        else "nenhum alerta no limiar"
    )
    fit = calibrate_inflation(flat)
    factors = {label: f.factor for label, f in loo_inflation(by_race).items()}
    calibrated = summarize(flat, threshold=protocol.alert_threshold, factors=factors)
    table = summary_table(
        {
            "fator congelado (1,3)": summary,
            "diagnóstico: fator calibrado nas outras corridas": calibrated,
        }
    )
    return (
        f"Eventos: {summary.n_events}; primários (sem regra dura de não alertar e sem pista não "
        f"verde): "
        f"{summary.n_primary}. Alertas no limiar de {protocol.alert_threshold}: {precision}.\n\n"
        f"{render_markdown(table)}\n\n"
        f"Calibração (fator congelado), com o n de cada faixa (eventos, não corridas):\n\n"
        f"{render_markdown(reliability_table(summary))}\n\n"
        f"O fator de inflação que os dados pedem para 90% de cobertura da margem: "
        f"{num(fit.raw_factor, 2)} (n={fit.n}); a cobertura do intervalo com o fator congelado: "
        f"{num(fit.coverage_before, 2)}.\n"
    )


def section_undercut(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    skipped: Counter[str] = Counter()
    for o in main:
        skipped.update(o.skipped)
    why = ", ".join(f"{k}: {v}" for k, v in skipped.most_common()) or "nenhum"
    return (
        "**A ressalva do viés, antes dos números.** Só se observam as paradas que "
        "**aconteceram**, e as "
        "equipes tendem a parar quando o undercut funciona. Por isso (1) a taxa de sucesso entre "
        "as "
        "paradas observadas é maior que a de uma situação qualquer de atacante atrás de rival, "
        "(2) a "
        "**precisão é otimista** e a **revocação não é mensurável** (o denominador, as situações "
        "em que "
        'funcionaria e ninguém parou, não existe) e (3) a calibração é **condicional** a "a '
        "equipe decidiu "
        'parar". O resultado real também depende do que os dois lados fizeram depois.\n\n'
        f"Eventos descartados na busca (21 corridas): {why}.\n\n"
        f"### Ao vivo (horizonte e composto previstos; o resultado principal)\n\n"
        f"{undercut_block(main, LIVE, TEMPO, protocol)}\n"
        f"### Oráculo (horizonte e compostos reais; mede a física do modelo)\n\n"
        f"{undercut_block(main, ORACLE, TEMPO, protocol)}\n"
    )


def section_leakage(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    headline = protocol.headline_variant
    brier_t = overall_brier(_scores(main, headline, TEMPO))
    brier_v = overall_brier(_scores(main, headline, VOLTA))
    mae_t = pooled_mae(_records(main, headline, TEMPO))
    mae_v = pooled_mae(_records(main, headline, VOLTA))
    live_t = summarize(
        [r for v in _events(main, LIVE, TEMPO).values() for r in v],
        threshold=protocol.alert_threshold,
    )
    live_v = summarize(
        [r for v in _events(main, LIVE, VOLTA).values() for r in v],
        threshold=protocol.alert_threshold,
    )
    rows = [
        [
            "medida",
            "estado no instante t (o resultado)",
            "estado por contagem de voltas (diagnóstico)",
            "diferença pareada (contagem − instante)",
        ],
        [
            "Brier da vitória",
            fmt(cluster_bootstrap(brier_t)),
            fmt(cluster_bootstrap(brier_v)),
            fmt(paired_bootstrap(brier_v, brier_t)),
        ],
        [
            "erro do tempo de volta (s)",
            fmt(cluster_bootstrap(mae_t)),
            fmt(cluster_bootstrap(mae_v)),
            fmt(paired_bootstrap(mae_v, mae_t)),
        ],
        ["undercut: Brier (eventos primários)", num(live_t.brier), num(live_v.brier), "-"],
        [
            "undercut: alertas / precisão",
            f"{live_t.alerts} / {num(live_t.precision, 2)}",
            f"{live_v.alerts} / {num(live_v.precision, 2)}",
            "-",
        ],
    ]
    return (
        "O estado antigo dava a cada piloto a volta N DELE, o que dá ao retardatário informação "
        "de depois do "
        "instante em que o líder completa a volta. Aqui as duas versões, sobre as mesmas corridas "
        "e voltas, "
        "mostram o quanto esse vazamento flatearia o backtest. **Diferença negativa = o estado "
        "antigo é "
        "mais otimista.** O resultado do relatório é o da coluna do instante t.\n\n"
        f"{render_markdown(rows)}\n"
    )


def section_apart(apart: Sequence[RaceOutcome], protocol: Protocol) -> str:
    if not apart:
        return "Nenhuma corrida de fora nesta execução.\n"
    headline = protocol.headline_variant
    rows = [
        [
            "corrida",
            "tipo",
            "previsões",
            "Brier MC",
            "líder vence",
            "todos iguais",
            "tempo de volta h=1 (s)",
            "persistência h=1 (s)",
            "eventos de undercut",
        ]
    ]
    for o in apart:
        scores = o.forecast.get((headline, TEMPO, 1.0), [])
        records = o.laptime.get((headline, TEMPO), [])
        mae = per_race(records, 1, "mae").get(o.label)
        base = per_race(records, 1, "baseline_mae").get(o.label)
        events = o.undercut.get((LIVE, TEMPO), [])

        def avg(attr, scores=scores):
            values = [getattr(s, attr) for s in scores]
            return sum(values) / len(values) if values else None

        rows.append(
            [
                o.label,
                o.kind,
                str(len(scores)),
                num(avg("brier_win")),
                num(avg("leader_brier")),
                num(avg("uniform_brier")),
                num(mae),
                num(base),
                f"{len(events)} ({sum(r.primary for r in events)} primários)",
            ]
        )
    return (
        "A corrida britânica (pista secando), o Canadá e o Brasil (molhadas) **não entram nas "
        "médias das "
        "secas**. O modelo só conhece compostos de seco: pilotos de intermediário ou de chuva "
        "ficam sem "
        "modelo e, se vencerem, contam como previsão com probabilidade 0. **Uma corrida cada: sem "
        "intervalo de confiança, só os pontos**, que descrevem o que acontece fora do domínio do "
        "modelo, "
        "não o desempenho dele.\n\n"
        f"{render_markdown(rows)}\n"
    )


def section_summary(main: Sequence[RaceOutcome], protocol: Protocol) -> str:
    """O resumo, gerado dos mesmos números das seções: a redação segue as regras congeladas."""
    headline = protocol.headline_variant
    lines = []

    # ---- tempo de volta contra a persistência
    records = _records(main, headline, TEMPO)
    by_horizon = {
        h: paired_bootstrap(per_race(records, h, "mae"), per_race(records, h, "baseline_mae"))
        for h in protocol.horizons
    }
    verdicts = Counter(verdict(d) for d in by_horizon.values())
    one = by_horizon[protocol.horizons[0]]
    lines.append(
        "**Tempo de volta.** Contra a persistência (a mediana das 3 últimas voltas limpas do "
        "piloto, "
        f"sem modelo): o modelo supera em {verdicts.get('supera', 0)} dos {len(by_horizon)} "
        f"horizontes, "
        f"não se distingue em {verdicts.get('não supera', 0)} e é pior em "
        f"{verdicts.get('é pior', 0)}. "
        f"No horizonte de {protocol.horizons[0]} volta a diferença de erro absoluto (modelo menos "
        f"persistência) é {fmt(one)} s."
    )

    # ---- Monte Carlo contra o líder
    scores = _scores(main, headline, TEMPO)
    rows = progress_rows(scores, n_boot=protocol.bootstrap_resamples)
    model, leader = overall_brier(scores), overall_brier(scores, "leader_brier")
    uniform = overall_brier(scores, "uniform_brier")
    overall = paired_bootstrap(model, leader, seed=protocol.bootstrap_seed)
    groups: dict[str, int] = Counter(r.verdict_leader for r in rows if r.n_races)
    lines.append(
        '**Probabilidade de vitória.** Contra "o líder atual vence", em '
        f"{sum(groups.values())} faixas de andamento, o Monte Carlo supera em "
        f"{groups.get('supera', 0)}, "
        f"não se distingue em {groups.get('não supera', 0)} e é pior em {groups.get('é pior', 0)}. "
        f"No geral o Brier é {fmt(cluster_bootstrap(model))} contra "
        f"{fmt(cluster_bootstrap(leader))} "
        f'(diferença pareada {fmt(overall)}: {verdict(overall)}). Contra "todos iguais" '
        f"({fmt(cluster_bootstrap(uniform))}) o veredito é "
        f'"{verdict(paired_bootstrap(model, uniform))}".'
    )

    # ---- as variantes de k e a taxa de SC
    base_brier = overall_brier(_scores(main, headline, TEMPO))
    base_mae = pooled_mae(_records(main, headline, TEMPO))
    distinct = []
    for variant in protocol.variants:
        if variant == headline:
            continue
        d_brier = paired_bootstrap(overall_brier(_scores(main, variant, TEMPO)), base_brier)
        d_mae = paired_bootstrap(pooled_mae(_records(main, variant, TEMPO)), base_mae)
        if (
            d_brier
            and d_mae
            and ((d_brier.high < 0 and d_mae.high < 0) or (d_brier.low > 0 and d_mae.low > 0))
        ):
            distinct.append(variant)
    lines.append(
        "**Variantes de k.** "
        + (
            f"Distinguíveis do principal pela regra: {', '.join(distinct)}."
            if distinct
            else "Nenhuma variante se distingue do encolhimento (sem diferença detectável)."
        )
    )
    by_mult = {m: overall_brier(_scores(main, headline, TEMPO, m)) for m in protocol.sc_multipliers}
    reading = {
        "supera": "Brier menor",
        "é pior": "Brier maior",
        "não supera": "sem diferença detectável",
    }
    moves = [
        f"taxa {m:g}x: {reading[verdict(paired_bootstrap(by_mult[m], by_mult[1.0]))]}"
        for m in protocol.sc_multipliers
        if m != 1.0 and 1.0 in by_mult
    ]
    lines.append(
        "**Taxa de SC** (Brier contra a taxa de 1x, pareado por corrida): " + "; ".join(moves) + "."
    )

    # ---- o undercut ao vivo
    flat = [r for v in _events(main, LIVE, TEMPO).values() for r in v]
    live = summarize(flat, threshold=protocol.alert_threshold)
    hits = round((live.precision or 0.0) * live.alerts) if live.alerts else 0
    interval = wilson(hits, live.alerts)
    lines.append(
        f"**Undercut ao vivo** ({live.n_primary} eventos primários de {live.n_events}). "
        + (
            f"{live.alerts} alertas no limiar de {protocol.alert_threshold}, {hits} deram certo "
            f"(precisão {hits / live.alerts:.0%}, Wilson {interval[0]:.0%} a {interval[1]:.0%}); "
            if interval
            else "Nenhum alerta no limiar; "
        )
        + f"Brier {num(live.brier)} contra {num(live.brier_baseline)} de quem prevê sempre a "
        "taxa-base; "
        f"cobertura do intervalo de 90% da margem: {num(live.coverage_90, 2)}. **Precisão "
        f"otimista** "
        "(só as paradas que aconteceram) e revocação não mensurável."
    )

    # ---- o vazamento
    brier_v = overall_brier(_scores(main, headline, VOLTA))
    leak = paired_bootstrap(brier_v, base_brier)
    if leak is not None and leak.high < 0:
        reading_leak = "o estado antigo é mais otimista"
    elif leak is not None and leak.low > 0:
        reading_leak = "o estado antigo é mais pessimista"
    else:
        reading_leak = "sem diferença detectável"
    lines.append(
        "**Vazamento do estado por contagem de voltas.** O Brier da vitória com o estado antigo "
        f"menos o do estado no instante t: {fmt(leak)} ({reading_leak})."
    )
    return "\n\n".join(lines) + "\n"


LIMITATIONS = """\
- **O SC depois da volta 10 vem de 3 eventos** (8 períodos de SC em 7 das 21 corridas secas, 5 deles
  nas voltas 1 a 7). As paradas sob SC são 15 e a razão de custo (0,58) tem desvio de ~0,4.
- **As regras de parada, de ultrapassagem e o desgaste sob SC são heurísticas**, não medidas:
  equipes decidem por rivais, tráfego e estratégia que o ganho de tempo não vê.
- **VSC (8 períodos) e bandeira vermelha (2) ficam fora do simulador**, e abandono não causa SC.
- **Vinte e uma corridas** é uma amostra pequena: os intervalos de bootstrap são largos, e o poder
  para distinguir variantes ou comparadores é baixo.
- **Uma só temporada** (2024) e um só conjunto de regras: nada garante que o resultado vale para
  2025.
- **O viés do undercut** (só as paradas que aconteceram): a precisão é otimista e a revocação não é
  mensurável.
- **As constantes (perda no pit, SC, abandonos, trocas)** saem das outras 20 corridas, não só das
  anteriores no calendário, o que ao vivo não se saberia.
- **O estado por instante** exclui quem não cruza a linha há mais de duas voltas típicas
  (heurística).
"""


def build_report(
    outcomes: Sequence[RaceOutcome],
    protocol: Protocol = FROZEN,
    *,
    adjustments: Sequence[str] = (),
    notes: Sequence[str] = (),
    generated_on: date | None = None,
    quick: bool = False,
) -> str:
    main, apart = _split(outcomes)
    day = (generated_on or date.today()).isoformat()
    seconds = sum(o.seconds for o in outcomes)
    banner = (
        "> **VERIFICAÇÃO RÁPIDA: menos simulações e menos voltas. Isto NÃO é o backtest.**\n\n"
        if quick
        else ""
    )
    if adjustments:
        adjusted = "\n".join(f"- {a}" for a in adjustments)
    else:
        adjusted = "Nenhum. Nenhum valor do protocolo foi alterado depois de ver o resultado."
    if notes:
        adjusted += (
            "\n\n**Análises e correções feitas depois do resultado, que NÃO alteram o protocolo "
            "nem o "
            "que foi medido:**\n\n" + "\n".join(f"- {n}" for n in notes)
        )
    return (
        "# Backtest da Fase 3: previsão do tempo de volta, Monte Carlo e alerta de undercut\n\n"
        f"{banner}"
        f"Gerado por `python -m strategy.backtest` em {day} sobre o commit `{git_head()}` "
        f"({len(main)} corridas secas com leave-one-race-out e {len(apart)} fora delas; "
        f"{seconds:.0f} s de CPU). O protocolo foi **congelado antes** de ver o resultado "
        "(`docs/backtest-protocolo.md`).\n\n"
        "## Resumo\n\n"
        f"{section_summary(main, protocol)}\n"
        "## Valores congelados\n\n"
        f"{section_frozen(protocol)}\n"
        "## 1. Erro de previsão do tempo de volta\n\n"
        f"{section_laptime(main, protocol)}\n"
        "## 2. Probabilidade de vitória: Brier por andamento da corrida\n\n"
        f"{section_brier(main, protocol)}\n"
        "## 3. Curva de calibração\n\n"
        f"{section_calibration(main, protocol)}\n"
        "## 4. As variantes de k\n\n"
        f"{section_variants(main, protocol)}\n"
        "## 5. Sensibilidade à taxa de Safety Car\n\n"
        f"{section_sensitivity(main, protocol)}\n"
        "## 6. O alerta de undercut nas paradas reais\n\n"
        f"{section_undercut(main, protocol)}\n"
        "## 7. Diagnóstico do vazamento: o estado por instante contra o por contagem de voltas\n\n"
        f"{section_leakage(main, protocol)}\n"
        "## 8. A britânica, o Canadá e o Brasil, à parte\n\n"
        f"{section_apart(apart, protocol)}\n"
        "## Limitações\n\n"
        f"{LIMITATIONS}\n"
        "## Ajustes pós-resultado\n\n"
        f"{adjusted}\n"
    )
