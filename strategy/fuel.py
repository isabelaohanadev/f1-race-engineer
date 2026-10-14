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
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np

from db.session import get_session_factory
from strategy.cleaning import clean_laps
from strategy.laps import DRY_COMPOUNDS, LapRow, list_races, load_race_laps, race_label
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
# Corrida "seca" para validar o modelo: a pista não pode secar durante os stints.
MIN_DRY_RACE_SHARE = 0.99
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
    full: FuelChoice
    progress: dict[float, FuelChoice]
    sensitivity: list[SensitivityPoint]
    chosen: dict[str, CompoundSlope]  # a degradação com o k que o critério escolheu


def dry_race_share(rows: Sequence[LapRow]) -> float:
    known = [lap for lap in rows if lap.compound is not None]
    return sum(lap.compound in DRY_COMPOUNDS for lap in known) / len(known) if known else 0.0


def analyze_race(label: str, rows: Sequence[LapRow], total_laps: int) -> RaceFuelReport:
    laps = dry_clean_laps(rows)
    physical = physical_coefficient(total_laps)
    progress = {
        fraction: choose_fuel_coefficient(
            estimate_fuel_coefficient(laps_until(laps, round(fraction * total_laps))), physical
        )
        for fraction in PROGRESS_FRACTIONS
    }
    full = choose_fuel_coefficient(estimate_fuel_coefficient(laps), physical)
    return RaceFuelReport(
        label=label,
        total_laps=total_laps,
        n_laps=len(laps),
        physical=physical,
        full=full,
        progress=progress,
        sensitivity=sensitivity(laps, physical),
        chosen={s.compound: s for s in compound_slopes(laps, full.kappa)},
    )


def load_reports(session, year: int | None = None) -> tuple[list[RaceFuelReport], list[str]]:
    """Relatório das corridas secas e os rótulos das que ficaram de fora (molhada ou mista)."""
    reports, skipped = [], []
    for race in list_races(session, year):
        rows = load_race_laps(session, race.id)
        label = race_label(race.year, race.round, race.name)
        total = race.total_laps or max((lap.lap_number for lap in rows), default=0)
        if not rows or not total or dry_race_share(rows) < MIN_DRY_RACE_SHARE:
            skipped.append(label)
            continue
        reports.append(analyze_race(label, rows, total))
    return reports, skipped


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
    header = ["corrida", *[f"até {pct(f)} da corrida" for f in PROGRESS_FRACTIONS]]
    rows = [header]
    for r in reports:
        cells = []
        for f in PROGRESS_FRACTIONS:
            c = r.progress[f]
            cells.append(
                f"{c.source} ({_num(c.ratio, 2)}x)" if c.ratio is not None else "físico (-)"
            )
        rows.append([r.label, *cells])
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
        f"- {len(reports)} corridas secas analisadas (ficaram de fora, por pista molhada ou mista: "
        f"{', '.join(skipped) if skipped else 'nenhuma'}).",
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
        lines.append(
            f"- Com dados só até {pct(fraction)} da corrida: estimado em {n} de {len(reports)}."
        )
    return lines


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
            f"DECISÃO COM DADOS ATÉ CADA FRAÇÃO DA CORRIDA\n"
            f"{render_text(progress_table(reports))}\n\n"
            f"SENSIBILIDADE DA DEGRADAÇÃO (s por volta de idade) AO COEFICIENTE\n"
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
        "## Decisão com dados só até cada fração da corrida (sem o futuro)\n\n"
        f"{render_markdown(progress_table(reports))}\n\n"
        "## Sensibilidade: degradação (s por volta de idade) para `k` de 0 a 2x o físico\n\n"
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        reports, skipped = load_reports(session, args.year)
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
