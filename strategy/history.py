"""O que as outras corridas dizem sobre Safety Car, abandonos e trocas de composto, SEM a corrida
avaliada: `python -m strategy.history --markdown docs/safety-car.md`.

O Monte Carlo (strategy/montecarlo.py) precisa de coisas que o ritmo dos pilotos não diz: se vai
haver Safety Car, quando, por quanto tempo e o que ele faz com a corrida; quem abandona; para qual
composto cada carro troca. Tudo isso vem de OUTRAS corridas, e aqui está o porquê de cada escolha.

SC COMO EVENTO DA CORRIDA, não como taxa por volta. Uma taxa por volta (SC por volta de corrida)
mistura três coisas que se comportam diferente: a chance de haver SC na corrida, em que ponto dela e
por quantas voltas. Modelo: o número de SCs de uma corrida é Poisson(lambda) e cada um começa numa
fração da corrida sorteada das frações históricas e dura o que as durações históricas dizem. É um
processo de Poisson marcado, e dele sai a conta que importa na volta N: dado que `k` SCs já
aconteceram, os que faltam são Poisson(lambda * (1 - F(N))), com F a distribuição das frações de
início. Com 21 corridas a incerteza de `lambda` é grande, então ela entra na simulação: cada
simulação sorteia lambda de Gamma(eventos + 0,5, corridas) (a posteriori de Jeffreys), em vez de
fixá-lo num ponto. A incerteza da taxa vira incerteza da probabilidade de vitória.

O QUE O SC FAZ, MEDIDO e não suposto:
* a volta sob SC leva `lap_ratio` vezes a volta verde do mesmo piloto (todas as voltas no meio do
  SC, fora de box), mediana por corrida;
* os carros fecham: o espaçamento entre carros consecutivos do pelotão (`gap`), em segundos;
* a parada custa menos. A perda contra o PRÓPRIO ritmo (a do pitloss.py) não serve aqui: no SC todo
  o campo anda devagar, então o piloto que para perde mais, não menos, contra o ritmo verde dele.
  O que importa é a perda CONTRA O CAMPO: o tempo de (entrada + saída) do piloto menos o dos
  carros que não pararam nas mesmas voltas. Em bandeira verde ela dá 22,6 s (n=530), igual à do
  pitloss.py, o que valida o método. Sob SC dá 13,2 s (n=15), uma razão de ~0,58, com muito ruído.
O desgaste do pneu sob SC NÃO está aqui: não se mede o desgaste num tempo de volta sob SC, e o fator
(`SC_WEAR`) é uma heurística declarada no simulador.

SÓ O SC (código "4"). VSC e bandeira vermelha ficam de fora do simulador, e o relatório conta
quantos foram: 8 períodos de VSC nas corridas secas de 2024 e 2 vermelhas (as duas na volta 1).

POUCOS DADOS. 8 períodos de SC em 7 das 21 corridas secas, e 5 deles nas voltas 1 a 7 (incidentes de
largada): o SC DEPOIS da volta 10 vem de 3 eventos. O relatório e o simulador carregam essas
contagens; ninguém deve ler a probabilidade de SC como mais precisa do que ela é.

ABANDONO. Um piloto-corrida abandona com probabilidade `r` (global, sem a corrida avaliada), em
fração da corrida sorteada das frações históricas, o mesmo desenho do SC. `r ~ Beta(k + 0,5,
n - k + 0,5)`. Só o status `Retired` conta (desqualificação não é abandono: o carro correu).

TROCA DE COMPOSTO. Para qual composto cada carro troca vem da contagem histórica (composto antes ->
depois de cada parada), suavizada.
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import median

from db.session import get_session_factory
from strategy.laps import DRY_COMPOUNDS, EntryRow, LapRow, load_race_entries, load_race_laps
from strategy.population import RaceRef, split_population
from strategy.tables import render_markdown, render_text

SC_CODE = "4"
VSC_CODE = "6"
RED_CODE = "5"
MAJORITY = 0.5  # fração dos carros com o código na volta para a volta valer como "sob SC"
MIN_SC_LAPS = 10  # voltas interiores de SC para a medição da corrida valer
MIN_FIELD = 8  # carros que não pararam, para servir de referência a uma parada
MIN_STOPS = 8  # paradas (verdes ou sob SC) no conjunto para a razão de custo valer
MAD_TO_SD = 1.4826
STARTER_EXCLUDED = frozenset({"Did not start"})

# O que valem as constantes quando as outras corridas não bastam. Vêm destas 21 corridas (todas
# dentro): servem ao uso ao vivo, e NUNCA a um backtest.
DEFAULT_LAP_RATIO = 1.46
DEFAULT_GAP_S = 1.75
DEFAULT_COST_RATIO = (0.58, 0.40)  # (razão central, desvio)
DEFAULT_SOURCE = "padrão: 21 corridas de 2024, INCLUSIVE a avaliada (não use em backtest)"


# ---------------------------------------------------------------- por corrida


@dataclass(frozen=True, slots=True)
class RaceHistory:
    """O que UMA corrida, vista inteira, diz sobre o SC, os abandonos e as trocas de composto."""

    label: str
    year: int
    round: int
    total_laps: int
    sc_periods: tuple[tuple[int, int], ...]  # (volta inicial, volta final), inclusive
    vsc_periods: tuple[tuple[int, int], ...]
    red_periods: tuple[tuple[int, int], ...]
    starters: int  # pilotos que largaram (com status conhecido)
    retirement_laps: tuple[int, ...]  # a última volta de quem abandonou
    transitions: tuple[tuple[str, str], ...]  # (composto antes, depois) de cada parada
    sc_lap_ratio: float | None  # mediana da volta sob SC sobre a verde (None sem voltas bastantes)
    sc_gap: float | None  # mediana do espaçamento entre carros consecutivos sob SC
    green_stop_losses: tuple[float, ...]  # perda contra o campo, paradas em bandeira verde
    sc_stop_losses: tuple[float, ...]  # a mesma, paradas sob SC


def lap_flags(rows: Sequence[LapRow], code: str) -> dict[int, bool]:
    """Por volta da corrida: o código de status aparece em pelo menos metade dos carros?"""
    seen: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        seen[row.lap_number].append(code in (row.track_status or ""))
    return {lap: sum(v) >= MAJORITY * len(v) for lap, v in seen.items()}


def runs(flags: dict[int, bool]) -> tuple[tuple[int, int], ...]:
    """Os intervalos contíguos de voltas com a flag ligada."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    previous = 0
    for lap in sorted(flags):
        if flags[lap] and start is None:
            start = lap
        elif not flags[lap] and start is not None:
            out.append((start, previous))
            start = None
        previous = lap
    if start is not None:
        out.append((start, previous))
    return tuple(out)


def _interior(flags: dict[int, bool]) -> set[int]:
    """Voltas de SC com SC também na anterior e na seguinte: sem a largada nem a relargada."""
    return {n for n, on in flags.items() if on and flags.get(n - 1) and flags.get(n + 1)}


def _green_medians(rows: Sequence[LapRow]) -> dict[str, float]:
    times: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if (
            r.lap_time_ms is not None
            and r.track_status == "1"
            and not r.pit_in
            and not r.pit_out
            and r.lap_number > 1
        ):
            times[r.driver].append(r.lap_time_ms / 1000)
    return {d: float(median(v)) for d, v in times.items() if v}


def measure_lap_ratio(rows: Sequence[LapRow], sc: dict[int, bool]) -> float | None:
    """A volta sob SC sobre a volta verde do mesmo piloto, mediana (só o miolo do SC, fora de
    box)."""
    green = _green_medians(rows)
    interior = _interior(sc)
    ratios = [
        r.lap_time_ms / 1000 / green[r.driver]
        for r in rows
        if r.lap_number in interior
        and r.lap_time_ms is not None
        and not r.pit_in
        and not r.pit_out
        and r.driver in green
    ]
    return float(median(ratios)) if len(ratios) >= MIN_SC_LAPS else None


def measure_gap(rows: Sequence[LapRow], sc: dict[int, bool]) -> float | None:
    """O espaçamento mediano entre carros consecutivos do pelotão sob SC. Distâncias maiores que
    meia
    volta são entre carros de voltas diferentes (retardatários), e não contam."""
    clocks: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r.session_time_ms is not None and r.lap_number in _interior(sc):
            clocks[r.lap_number].append((r.session_time_ms / 1000, (r.lap_time_ms or 0) / 1000))
    gaps: list[float] = []
    for entries in clocks.values():
        entries.sort()
        half = float(median(t for _, t in entries)) / 2 if entries else 0.0
        gaps += [
            b[0] - a[0] for a, b in zip(entries, entries[1:], strict=False) if b[0] - a[0] < half
        ]
    return float(median(gaps)) if len(gaps) >= MIN_SC_LAPS else None


def measure_field_stop_losses(
    rows: Sequence[LapRow], sc: dict[int, bool]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(paradas verdes, paradas sob SC): a perda de cada parada CONTRA O CAMPO.

    Para a parada com entrada na volta `n`: o tempo do piloto da volta n-1 até a n+1 (a de entrada e
    a de saída) menos a mediana, nas mesmas voltas, dos carros que não pararam por perto. Compara
    carros nas mesmas voltas e nas mesmas condições de pista, então vale tanto em bandeira verde
    como sob SC, onde a perda contra o ritmo próprio engana (todo o campo anda devagar)."""
    clock = {(r.driver, r.lap_number): r.session_time_ms / 1000 for r in rows if r.session_time_ms}
    pit_in = {(r.driver, r.lap_number) for r in rows if r.pit_in}
    pit_out = {(r.driver, r.lap_number) for r in rows if r.pit_out}
    status = {(r.driver, r.lap_number): r.track_status for r in rows}
    drivers = {r.driver for r in rows}
    green: list[float] = []
    under_sc: list[float] = []
    for driver, n in sorted(pit_in):
        if n < 3 or (driver, n - 1) not in clock or (driver, n + 1) not in clock:
            continue
        window = (n - 1, n, n + 1)
        in_sc = sc.get(n, False) and sc.get(n + 1, False)
        in_green = all(status.get((driver, lap)) == "1" for lap in window)
        if not (in_sc or in_green):
            continue  # a pista mudou no meio da parada: não é de nenhum dos dois
        reference = [
            clock[(e, n + 1)] - clock[(e, n - 1)]
            for e in drivers - {driver}
            if (e, n - 1) in clock
            and (e, n + 1) in clock
            and not any((e, lap) in pit_in for lap in window)
            and (e, n) not in pit_out
        ]
        if len(reference) < MIN_FIELD:
            continue
        loss = clock[(driver, n + 1)] - clock[(driver, n - 1)] - float(median(reference))
        (under_sc if in_sc else green).append(loss)
    return tuple(green), tuple(under_sc)


def retirements(entries: Sequence[EntryRow], rows: Sequence[LapRow]) -> tuple[int, tuple[int, ...]]:
    """(largaram, última volta de quem abandonou). Só o status `Retired` conta; quem não largou ou
    não tem status sai da conta (desconhecido não é "terminou")."""
    last: dict[str, int] = {}
    for r in rows:
        last[r.driver] = max(last.get(r.driver, 0), r.lap_number)
    started = [e for e in entries if e.status is not None and e.status not in STARTER_EXCLUDED]
    laps = tuple(sorted(last.get(e.driver, 0) for e in started if e.status == "Retired"))
    return len(started), laps


def compound_transitions(rows: Sequence[LapRow]) -> tuple[tuple[str, str], ...]:
    """Cada parada como (composto antes, composto depois), só entre compostos de pista seca."""
    stints: dict[str, dict[int, str]] = defaultdict(dict)
    for r in rows:
        if r.stint is not None and r.compound is not None:
            stints[r.driver].setdefault(r.stint, r.compound)
    out = []
    for per_stint in stints.values():
        order = [per_stint[k] for k in sorted(per_stint)]
        out += [
            (a, b)
            for a, b in zip(order, order[1:], strict=False)
            if a in DRY_COMPOUNDS and b in DRY_COMPOUNDS
        ]
    return tuple(sorted(out))


def compute_race_history(
    ref: RaceRef, rows: Sequence[LapRow], entries: Sequence[EntryRow]
) -> RaceHistory:
    sc = lap_flags(rows, SC_CODE)
    green, under_sc = measure_field_stop_losses(rows, sc)
    starters, retired = retirements(entries, rows)
    return RaceHistory(
        label=ref.label,
        year=ref.year,
        round=ref.round,
        total_laps=ref.total_laps,
        sc_periods=runs(sc),
        vsc_periods=runs(lap_flags(rows, VSC_CODE)),
        red_periods=runs(lap_flags(rows, RED_CODE)),
        starters=starters,
        retirement_laps=retired,
        transitions=compound_transitions(rows),
        sc_lap_ratio=measure_lap_ratio(rows, sc),
        sc_gap=measure_gap(rows, sc),
        green_stop_losses=green,
        sc_stop_losses=under_sc,
    )


def load_race_histories(session, year: int | None = None) -> list[RaceHistory]:
    main, _ = split_population(session, year)
    return [
        compute_race_history(
            ref, load_race_laps(session, ref.race_id), load_race_entries(session, ref.race_id)
        )
        for ref in main
    ]


# ---------------------------------------------------------------- as constantes sem a corrida


@dataclass(frozen=True, slots=True)
class SafetyCarPrior:
    """Tudo o que o simulador precisa saber do SC, das OUTRAS corridas."""

    events: int  # períodos de SC em todas as corridas de referência
    races: int
    start_fractions: tuple[float, ...]  # a fração da corrida em que cada um começou (volta/total)
    durations: tuple[int, ...]  # voltas de cada um
    lap_ratios: tuple[float, ...]  # a medida de cada corrida que teve voltas de SC bastantes
    gaps: tuple[float, ...]
    cost_ratio: tuple[float, float]  # (razão central, desvio) da perda da parada sob SC
    n_sc_stops: int
    n_green_stops: int

    @property
    def rate(self) -> float:
        """SCs por corrida (estimativa pontual: o simulador sorteia da posteriori)."""
        return self.events / self.races if self.races else 0.0


@dataclass(frozen=True, slots=True)
class RetirementPrior:
    retirements: int
    starters: int
    fractions: tuple[float, ...]  # a fração da corrida da última volta de cada abandono

    @property
    def rate(self) -> float:
        return self.retirements / self.starters if self.starters else 0.0


@dataclass(frozen=True, slots=True)
class CompoundPrior:
    counts: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SimPriors:
    sc: SafetyCarPrior
    retirement: RetirementPrior
    compounds: CompoundPrior
    source: str


def _robust_sd(values: Sequence[float], floor: float) -> float:
    centre = median(values)
    return max(MAD_TO_SD * float(median(abs(v - centre) for v in values)), floor)


def leave_one_out_history(
    histories: Sequence[RaceHistory], exclude: str, *, earlier_only: bool = False
) -> SimPriors:
    """As constantes do simulador sem a corrida `exclude` (e, com `earlier_only`, só com as
    anteriores
    a ela no calendário, que é o que se saberia ao vivo)."""
    target = next((h for h in histories if h.label == exclude), None)
    if earlier_only and target is None:
        raise ValueError(f"corrida {exclude!r} não está no histórico: earlier_only não se aplica")
    others = [h for h in histories if h.label != exclude]
    if earlier_only and target is not None:
        others = [h for h in others if (h.year, h.round) < (target.year, target.round)]

    periods = [(h, p) for h in others for p in h.sc_periods]
    green = [x for h in others for x in h.green_stop_losses]
    under_sc = [x for h in others for x in h.sc_stop_losses]
    if len(green) >= MIN_STOPS and len(under_sc) >= MIN_STOPS:
        base = float(median(green))
        cost = (float(median(under_sc)) / base, _robust_sd(under_sc, 1.0) / base)
    else:
        cost = DEFAULT_COST_RATIO
    ratios = tuple(h.sc_lap_ratio for h in others if h.sc_lap_ratio is not None)
    gaps = tuple(h.sc_gap for h in others if h.sc_gap is not None)
    sc = SafetyCarPrior(
        events=len(periods),
        races=len(others),
        start_fractions=tuple(p[0] / h.total_laps for h, p in periods),
        durations=tuple(p[1] - p[0] + 1 for _, p in periods),
        lap_ratios=ratios or (DEFAULT_LAP_RATIO,),
        gaps=gaps or (DEFAULT_GAP_S,),
        cost_ratio=cost,
        n_sc_stops=len(under_sc),
        n_green_stops=len(green),
    )
    retirement = RetirementPrior(
        retirements=sum(len(h.retirement_laps) for h in others),
        starters=sum(h.starters for h in others),
        fractions=tuple(lap / h.total_laps for h in others for lap in h.retirement_laps),
    )
    counts: Counter[tuple[str, str]] = Counter(t for h in others for t in h.transitions)
    where = "só anteriores" if earlier_only else "todas as outras"
    return SimPriors(
        sc=sc,
        retirement=retirement,
        compounds=CompoundPrior(dict(counts)),
        source=f"sem {exclude}: {len(others)} corridas ({where})",
    )


def all_races_history(histories: Sequence[RaceHistory]) -> SimPriors:
    """As constantes com TODAS as corridas: o uso ao vivo, sem nada a avaliar (nunca num
    backtest)."""
    priors = leave_one_out_history(histories, exclude="")
    return SimPriors(priors.sc, priors.retirement, priors.compounds, DEFAULT_SOURCE)


# ---------------------------------------------------------------- relatório


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _periods(periods: Sequence[tuple[int, int]]) -> str:
    return ", ".join(f"{a}-{b}" for a, b in periods) or "-"


def history_table(histories: Sequence[RaceHistory]) -> list[list[str]]:
    header = [
        "corrida", "voltas", "SC (início-fim)", "VSC", "vermelha", "largaram", "abandonos",
        "ritmo no SC", "espaçamento (s)", "paradas verdes", "paradas sob SC",
    ]  # fmt: skip
    rows = [header]
    for h in histories:
        rows.append(
            [
                h.label,
                str(h.total_laps),
                _periods(h.sc_periods),
                _periods(h.vsc_periods),
                _periods(h.red_periods),
                str(h.starters),
                str(len(h.retirement_laps)),
                _fmt(h.sc_lap_ratio),
                _fmt(h.sc_gap),
                str(len(h.green_stop_losses)),
                str(len(h.sc_stop_losses)),
            ]
        )
    return rows


def summary_lines(histories: Sequence[RaceHistory]) -> list[str]:
    n = len(histories)
    events = [(h, p) for h in histories for p in h.sc_periods]
    with_sc = sum(bool(h.sc_periods) for h in histories)
    starts = sorted(p[0] for _, p in events)
    late = [p for _, p in events if p[0] > 10]
    durations = sorted(p[1] - p[0] + 1 for _, p in events)
    retired = sum(len(h.retirement_laps) for h in histories)
    starters = sum(h.starters for h in histories)
    green = [x for h in histories for x in h.green_stop_losses]
    under_sc = [x for h in histories for x in h.sc_stop_losses]
    lines = [
        f"- **Safety Car:** {len(events)} períodos em {with_sc} de {n} corridas secas "
        f"({len(events) / n:.2f} por corrida; probabilidade de ao menos um na corrida "
        f"{with_sc / n:.0%}). Voltas de início: {starts}.",
        f"- **Só {len(late)} dos {len(events)} começam depois da volta 10**; os demais são "
        "incidentes de largada. A chance de SC no meio da corrida vem desses poucos eventos e é "
        "muito incerta.",
        f"- **Duração** (voltas): {durations}; mediana {median(durations) if durations else '-'}.",
        f"- **VSC:** {sum(len(h.vsc_periods) for h in histories)} períodos (fora do simulador). "
        f"**Vermelha:** {sum(len(h.red_periods) for h in histories)} (fora do simulador).",
        f"- **Abandonos:** {retired} em {starters} largadas "
        f"({retired / starters:.1%} por piloto-corrida).",
    ]
    if green:
        if under_sc:
            ratio = median(under_sc) / median(green)
            sc_text = f"sob SC {median(under_sc):.1f} s (n={len(under_sc)}), razão {ratio:.2f}."
        else:
            sc_text = "sem paradas sob SC."
        lines.append(
            f"- **Perda da parada contra o campo:** verde {median(green):.1f} s "
            f"(n={len(green)}); {sc_text}"
        )
    ratios = [h.sc_lap_ratio for h in histories if h.sc_lap_ratio is not None]
    gaps = [h.sc_gap for h in histories if h.sc_gap is not None]
    if ratios:
        lines.append(
            f"- **Ritmo sob SC:** {median(ratios):.2f}x a volta verde (por corrida de "
            f"{min(ratios):.2f} a {max(ratios):.2f}, {len(ratios)} corridas). "
            f"**Espaçamento entre carros:** {median(gaps):.2f} s "
            f"(de {min(gaps):.2f} a {max(gaps):.2f})."
        )
    return lines


MARKDOWN_NOTES = """\
## Como ler

- **SC como evento da corrida.** O número de SCs é Poisson(lambda), o início sai das frações
  históricas e a duração das durações históricas; a incerteza de lambda (Gamma posteriori) entra em
  cada simulação. Detalhes em `strategy/history.py`.
- **A perda da parada sob SC é medida contra o CAMPO**, e não contra o ritmo próprio do piloto (que
  dá mais sob SC, porque todo o campo anda devagar). Em bandeira verde o método dá o mesmo que
  `docs/perda-no-pit.md`, o que o valida.
- **Poucos dados.** Todas as razões e probabilidades têm o `n` ao lado; o das paradas sob SC é
  pequeno e muito disperso, e a razão de custo tem um desvio grande.
- **Não está aqui:** o desgaste do pneu sob SC (não se mede num tempo de volta sob SC; é uma
  heurística declarada no simulador), VSC e bandeira vermelha.
"""


def build_report(
    histories: Sequence[RaceHistory], *, markdown: bool, generated_on: date | None = None
) -> str:
    table = history_table(histories)
    summary = "\n".join(summary_lines(histories))
    if not markdown:
        return f"{summary}\n\n{render_text(table)}\n"
    day = (generated_on or date.today()).isoformat()
    return (
        "# Safety Car, abandonos e trocas de composto\n\n"
        f"Gerado por `python -m strategy.history --markdown docs/safety-car.md` em {day}.\n\n"
        f"## Resumo\n\n{summary}\n\n{MARKDOWN_NOTES}\n## Por corrida\n\n{render_markdown(table)}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.history",
        description="Mostra o que as corridas secas dizem sobre SC, abandonos e trocas de "
        "composto.",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    parser.add_argument("--markdown", type=Path, default=None, help="grava o relatório (.md)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    with get_session_factory()() as session:
        histories = load_race_histories(session, args.year)
    if not histories:
        print(
            "Nenhuma corrida seca no banco. Ingira com: python -m ingestion --year 2024 --round 10"
        )
        return 1
    print(build_report(histories, markdown=False))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_report(histories, markdown=True), encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
