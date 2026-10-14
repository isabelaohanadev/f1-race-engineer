"""Qualidade dos dados de voltas, ANTES de qualquer modelagem: `python -m strategy.quality`.

Mede FATOS dos dados brutos (o que falta, o que é inconsistente, o que é raro). Não decide o que
excluir: as regras de limpeza são da etapa seguinte (strategy/cleaning.py) e ficam lá, num lugar só,
para este relatório não virar uma segunda cópia delas.

Por que inspecionar antes de modelar (boas para explicar em entrevista):
* Um modelo ajustado a dado sujo devolve números plausíveis e errados, sem erro nenhum. A inspeção
  é o que diz se "lap_time_ms" é confiável como variável de resposta e quais colunas são lixo.
* Cada verificação abaixo nasceu de uma pergunta que o modelo vai fazer ao dado: "a idade do pneu
  anda de 1 em 1?" (a regressão usa a idade como regressor), "o tempo de volta bate com o relógio
  da sessão?" (o intervalo entre carros sai do relógio, a resposta do modelo sai do tempo de volta),
  "a posição concorda com a ordem de chegada à linha?" (o undercut compara quem está à frente de
  quem).
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import median

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from db.models import Driver, Lap, PositionSample, Race, RaceEntry
from db.session import get_session_factory

# Códigos de status de pista do FastF1 (db/models.py: Lap.track_status).
STATUS_YELLOW, STATUS_SC, STATUS_RED, STATUS_VSC, STATUS_VSC_ENDING = "2", "4", "5", "6", "7"
STATUS_CODES = (STATUS_YELLOW, STATUS_SC, STATUS_RED, STATUS_VSC, STATUS_VSC_ENDING)
DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})

# O tempo de volta e o relógio da sessão são medidos separadamente e arredondados para o ms: até
# 2 ms de diferença é ruído de arredondamento, não inconsistência.
CLOCK_TOLERANCE_MS = 2
# "Volta lenta": mais de 125% da mediana da corrida. Não é regra de exclusão (o modelo robusto de
# ritmo lida com ela), é uma medida de quanta sujeira de tráfego/SC/erro há no dado.
SLOW_FACTOR = 1.25


@dataclass(frozen=True, slots=True)
class LapRow:
    """Uma volta, com só o que a inspeção lê (o mesmo que o modelo vai ler)."""

    driver: str
    lap_number: int
    lap_time_ms: int | None
    compound: str | None
    tyre_life: int | None
    stint: int | None
    position: int | None
    session_time_ms: int | None
    pit_in: bool | None
    pit_out: bool | None
    track_status: str | None


@dataclass(frozen=True, slots=True)
class EntryRow:
    """O resultado oficial de um piloto: é a "verdade" contra a qual o backtest mede a previsão."""

    driver: str
    finish_position: int | None
    status: str | None = None


@dataclass(slots=True)
class RaceQuality:
    label: str
    laps: int = 0
    drivers: int = 0
    total_laps: int | None = None  # o que a corrida diz que tem
    max_lap: int = 0  # a maior volta presente nos dados
    has_positions: bool = False  # se a ingestão foi completa (com trajetória)
    # Nulos por coluna ("não sei"), em número de voltas.
    missing: dict[str, int] = field(default_factory=dict)
    # Voltas de box e a armadilha: quantas delas TÊM tempo de volta (e contaminariam o modelo).
    pit_in_laps: int = 0
    pit_in_with_time: int = 0
    pit_out_laps: int = 0
    pit_out_with_time: int = 0
    # Voltas em que cada código de pista ocorreu (uma volta pode contar em vários).
    status_laps: dict[str, int] = field(default_factory=dict)
    green_laps: int = 0  # status exatamente "1": pista livre a volta toda
    compound_laps: dict[str, int] = field(default_factory=dict)
    # Inconsistências estruturais, em número de ocorrências.
    anomalies: dict[str, int] = field(default_factory=dict)
    # Saúde do resultado oficial (race_entries): o backtest o usa como a resposta certa.
    results: dict[str, int] = field(default_factory=dict)

    @property
    def dry_share(self) -> float:
        dry = sum(n for c, n in self.compound_laps.items() if c in DRY_COMPOUNDS)
        return dry / self.laps if self.laps else 0.0


def _by_driver(rows: Iterable[LapRow]) -> dict[str, list[LapRow]]:
    grouped: dict[str, list[LapRow]] = defaultdict(list)
    for row in rows:
        grouped[row.driver].append(row)
    for laps in grouped.values():
        laps.sort(key=lambda r: r.lap_number)
    return grouped


def count_lap_number_gaps(by_driver: dict[str, list[LapRow]]) -> int:
    """Voltas que faltam no meio da sequência de um piloto (ex.: tem a 5 e a 7, falta a 6).
    Quem abandona simplesmente para de ter voltas: isso NÃO é buraco."""
    gaps = 0
    for laps in by_driver.values():
        numbers = [lap.lap_number for lap in laps]
        gaps += (numbers[-1] - numbers[0] + 1) - len(numbers) if numbers else 0
    return gaps


def count_tyre_life_jumps(by_driver: dict[str, list[LapRow]]) -> int:
    """Voltas seguidas, no MESMO stint, em que a idade do pneu não subiu exatamente 1. A regressão
    usa a idade como regressor: se ela pula, o eixo X do modelo está errado."""
    jumps = 0
    for laps in by_driver.values():
        for prev, cur in zip(laps, laps[1:], strict=False):
            same_stint = prev.stint is not None and prev.stint == cur.stint
            contiguous = cur.lap_number == prev.lap_number + 1
            if (
                same_stint
                and contiguous
                and prev.tyre_life is not None
                and cur.tyre_life is not None
                and cur.tyre_life != prev.tyre_life + 1
            ):
                jumps += 1
    return jumps


def count_compound_changes_inside_stint(by_driver: dict[str, list[LapRow]]) -> int:
    """Troca de composto sem troca de stint: o stint deixa de identificar UM jogo de pneus."""
    changes = 0
    for laps in by_driver.values():
        for prev, cur in zip(laps, laps[1:], strict=False):
            if (
                prev.stint is not None
                and prev.stint == cur.stint
                and prev.compound is not None
                and cur.compound is not None
                and prev.compound != cur.compound
            ):
                changes += 1
    return changes


def count_session_time_not_increasing(by_driver: dict[str, list[LapRow]]) -> int:
    """O relógio da sessão no fim da volta tem de crescer a cada volta do piloto."""
    bad = 0
    for laps in by_driver.values():
        times = [lap.session_time_ms for lap in laps if lap.session_time_ms is not None]
        bad += sum(1 for a, b in zip(times, times[1:], strict=False) if b <= a)
    return bad


def count_clock_mismatches(by_driver: dict[str, list[LapRow]]) -> int:
    """Voltas em que o tempo de volta não bate com a diferença entre os relógios de fim de volta.
    Só voltas consecutivas e com os três valores. Esperado: ~0 (até 2 ms é arredondamento); o que
    sobra costuma ser bandeira vermelha ou volta de box, e é o que o intervalo entre carros NÃO pode
    usar às cegas."""
    bad = 0
    for laps in by_driver.values():
        for prev, cur in zip(laps, laps[1:], strict=False):
            if (
                cur.lap_number == prev.lap_number + 1
                and cur.lap_time_ms is not None
                and cur.session_time_ms is not None
                and prev.session_time_ms is not None
            ):
                elapsed = cur.session_time_ms - prev.session_time_ms
                if abs(elapsed - cur.lap_time_ms) > CLOCK_TOLERANCE_MS:
                    bad += 1
    return bad


def count_position_order_mismatches(rows: Sequence[LapRow]) -> int:
    """Voltas (número de volta) em que a ordem por `position` discorda da ordem de chegada à linha
    (`session_time_ms`). O undercut pergunta "quem está à frente de quem": se os dois discordam,
    é preciso escolher em qual confiar."""
    by_lap: dict[int, list[LapRow]] = defaultdict(list)
    for row in rows:
        if row.position is not None and row.session_time_ms is not None:
            by_lap[row.lap_number].append(row)
    mismatched = 0
    for laps in by_lap.values():
        ordered = sorted(laps, key=lambda r: (r.position, r.session_time_ms))
        times = [lap.session_time_ms for lap in ordered]
        if any(b < a for a, b in zip(times, times[1:], strict=False)):
            mismatched += 1
    return mismatched


def count_slow_laps(rows: Sequence[LapRow]) -> int:
    times = [r.lap_time_ms for r in rows if r.lap_time_ms is not None]
    if not times:
        return 0
    limit = SLOW_FACTOR * median(times)
    return sum(1 for t in times if t > limit)


def result_checks(entries: Sequence[EntryRow], rows: Sequence[LapRow]) -> dict[str, int]:
    """Saúde do resultado oficial. O backtest mede "quem venceu/subiu ao pódio" contra
    `finish_position`; se ele vier incompleto ou repetido, a nota do modelo vira ficção.

    * sem_posicao_final: pilotos sem `finish_position` (a fonte falhou, ou não classificou);
    * sem_status: pilotos sem `status` ("Finished", "Lapped", "Retired"...). O resultado OFICIAL vem
      de uma API externa (Ergast/Jolpica); quando ela não responde o FastF1 CALCULA a ordem a partir
      das voltas, que sai completa (1..N) mas sem status. Ou seja, `sem_status` > 0 é o sinal de que
      o resultado gravado é o calculado e não o oficial, o que o teste de buracos não enxerga;
    * posicao_final_repetida / com_buraco: a ordem de chegada tem de ser 1..N sem repetir nem pular;
    * final_vs_ultima_volta: o resultado oficial discorda da posição na última volta. É esperado
      para quem abandonou e para penalidades pós-corrida, por isso é uma medida, não um erro."""
    positions = [e.finish_position for e in entries if e.finish_position is not None]
    counts = Counter(positions)
    last_lap_position = {
        driver: laps[-1].position for driver, laps in _by_driver(rows).items() if laps[-1].position
    }
    return {
        "sem_posicao_final": len(entries) - len(positions),
        "sem_status": sum(1 for e in entries if e.status is None),
        "posicao_final_repetida": sum(1 for n in counts.values() if n > 1),
        "posicao_final_com_buraco": (max(positions) - len(counts)) if positions else 0,
        "final_vs_ultima_volta": sum(
            1
            for e in entries
            if e.finish_position is not None
            and e.driver in last_lap_position
            and last_lap_position[e.driver] != e.finish_position
        ),
    }


def inspect_laps(
    label: str,
    rows: Sequence[LapRow],
    *,
    entries: Sequence[EntryRow] = (),
    total_laps: int | None = None,
    has_positions: bool = False,
) -> RaceQuality:
    """O relatório de qualidade de UMA corrida. Função pura: recebe as voltas, devolve números."""
    quality = RaceQuality(
        label=label,
        laps=len(rows),
        drivers=len({r.driver for r in rows}),
        total_laps=total_laps,
        max_lap=max((r.lap_number for r in rows), default=0),
        has_positions=has_positions,
    )
    columns = (
        "lap_time_ms",
        "compound",
        "tyre_life",
        "stint",
        "position",
        "session_time_ms",
        "pit_in",
        "pit_out",
        "track_status",
    )
    quality.missing = {c: sum(1 for r in rows if getattr(r, c) is None) for c in columns}

    pit_in = [r for r in rows if r.pit_in]
    pit_out = [r for r in rows if r.pit_out]
    quality.pit_in_laps = len(pit_in)
    quality.pit_in_with_time = sum(1 for r in pit_in if r.lap_time_ms is not None)
    quality.pit_out_laps = len(pit_out)
    quality.pit_out_with_time = sum(1 for r in pit_out if r.lap_time_ms is not None)

    statuses = [r.track_status for r in rows if r.track_status is not None]
    quality.status_laps = {code: sum(1 for s in statuses if code in s) for code in STATUS_CODES}
    quality.green_laps = sum(1 for s in statuses if s == "1")
    quality.compound_laps = dict(Counter(r.compound or "?" for r in rows))

    by_driver = _by_driver(rows)
    quality.anomalies = {
        "buracos_de_volta": count_lap_number_gaps(by_driver),
        "idade_do_pneu_pula": count_tyre_life_jumps(by_driver),
        "composto_muda_no_stint": count_compound_changes_inside_stint(by_driver),
        "relogio_nao_cresce": count_session_time_not_increasing(by_driver),
        "relogio_vs_tempo_de_volta": count_clock_mismatches(by_driver),
        "posicao_vs_chegada": count_position_order_mismatches(rows),
        "voltas_lentas_acima_de_125pct": count_slow_laps(rows),
    }
    quality.results = result_checks(entries, rows)
    return quality


# ---------------------------------------------------------------- relatório


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def composition_table(qualities: Sequence[RaceQuality]) -> list[list[str]]:
    header = [
        "corrida",
        "voltas",
        "pilotos",
        "posições",
        "seco",
        "sem tempo",
        "box entrada (com tempo)",
        "box saída (com tempo)",
        "SC",
        "VSC",
        "vermelha",
        "amarela",
        "pista livre",
    ]
    rows = [header]
    for q in qualities:
        rows.append(
            [
                q.label,
                str(q.laps),
                str(q.drivers),
                "sim" if q.has_positions else "não",
                _pct(round(q.dry_share * q.laps), q.laps),
                _pct(q.missing.get("lap_time_ms", 0), q.laps),
                f"{q.pit_in_laps} ({q.pit_in_with_time})",
                f"{q.pit_out_laps} ({q.pit_out_with_time})",
                str(q.status_laps.get(STATUS_SC, 0)),
                str(q.status_laps.get(STATUS_VSC, 0)),
                str(q.status_laps.get(STATUS_RED, 0)),
                str(q.status_laps.get(STATUS_YELLOW, 0)),
                _pct(q.green_laps, q.laps),
            ]
        )
    return rows


def anomaly_table(qualities: Sequence[RaceQuality]) -> list[list[str]]:
    names = list(qualities[0].anomalies) if qualities else []
    results = list(qualities[0].results) if qualities else []
    unknown = ("pit_in", "pit_out", "track_status", "session_time_ms")
    rows = [["corrida", *names, *results, *[f"{c} nulo" for c in unknown]]]
    for q in qualities:
        rows.append(
            [
                q.label,
                *[str(q.anomalies[n]) for n in names],
                *[str(q.results[r]) for r in results],
                *[str(q.missing.get(c, 0)) for c in unknown],
            ]
        )
    return rows


def render_text(table: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def render_markdown(table: list[list[str]]) -> str:
    head, *body = table
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


# ---------------------------------------------------------------- banco


def load_qualities(session: Session, year: int | None = None) -> list[RaceQuality]:
    races = session.execute(
        select(Race.id, Race.year, Race.round, Race.name, Race.total_laps)
        .where(Race.year == year if year is not None else Race.id.is_not(None))
        .order_by(Race.year, Race.round)
    ).all()
    qualities = []
    for race in races:
        lap_rows = session.execute(
            select(
                Driver.code,
                Lap.lap_number,
                Lap.lap_time_ms,
                Lap.compound,
                Lap.tyre_life,
                Lap.stint,
                Lap.position,
                Lap.session_time_ms,
                Lap.pit_in,
                Lap.pit_out,
                Lap.track_status,
            )
            .join(Driver, Driver.id == Lap.driver_id)
            .where(Lap.race_id == race.id)
        ).all()
        entries = session.execute(
            select(Driver.code, RaceEntry.finish_position, RaceEntry.status)
            .join(Driver, Driver.id == RaceEntry.driver_id)
            .where(RaceEntry.race_id == race.id)
        ).all()
        # EXISTS para na primeira linha: contar ~800 mil posições só para saber "tem ou não" seria
        # trabalho jogado fora.
        has_positions = bool(
            session.scalar(select(exists().where(PositionSample.race_id == race.id)))
        )
        qualities.append(
            inspect_laps(
                f"{race.year} R{race.round:02d} {race.name.removesuffix(' Grand Prix')}",
                [LapRow(*row) for row in lap_rows],
                entries=[EntryRow(*row) for row in entries],
                total_laps=race.total_laps,
                has_positions=has_positions,
            )
        )
    return qualities


MARKDOWN_LEGEND = """\
## Como ler

**Composição e cobertura**

- **posições**: `sim` = ingestão completa (tem a trajetória); `não` = ingerida com `--laps-only`.
- **seco**: % das voltas em SOFT, MEDIUM ou HARD. Corrida com pouco "seco" não serve para modelar o
  desgaste do pneu de pista seca.
- **sem tempo**: % das voltas com tempo de volta nulo.
- **box entrada / saída (com tempo)**: quantas voltas terminam / começam nos boxes e, entre
  parênteses, quantas delas **têm** tempo de volta. É a armadilha: um modelo que só descartasse
  "voltas sem tempo" manteria essas voltas lentas na regressão.
- **SC, VSC, vermelha, amarela**: voltas em que o código de pista ocorreu, mesmo que por parte da
  volta (uma volta pode contar em mais de uma coluna).
- **pista livre**: % das voltas com status exatamente `1`.

**Inconsistências estruturais** (esperado: 0)

- **buracos_de_volta**: voltas que faltam no meio da sequência de um piloto (abandono não conta).
- **idade_do_pneu_pula**: a idade do pneu não subiu 1 entre voltas seguidas do mesmo stint.
- **composto_muda_no_stint**: o composto mudou sem o stint mudar.
- **relogio_nao_cresce**: o relógio da sessão no fim da volta não cresceu de uma volta para outra.
- **relogio_vs_tempo_de_volta**: o tempo de volta não bate (tolerância de 2 ms) com a diferença
  entre os relógios de duas voltas seguidas.
- **posicao_vs_chegada**: voltas em que a ordem por `position` discorda da ordem de chegada à linha.
- **voltas_lentas_acima_de_125pct**: voltas acima de 125% da mediana da corrida (medida de sujeira
  por tráfego, SC ou erro, não regra de exclusão).

**Resultado oficial** (a "verdade" do backtest)

- **sem_posicao_final / posicao_final_repetida / posicao_final_com_buraco**: a ordem de chegada
  tem de ser 1..N sem faltar, repetir nem pular.
- **sem_status**: pilotos sem `status`. O resultado oficial vem de uma API externa (Ergast/Jolpica);
  quando ela não responde, o FastF1 **calcula** a ordem a partir das voltas, que sai completa mas
  sem status. Acima de 0 = o resultado gravado é o calculado, e reingerir a corrida o corrige.
- **final_vs_ultima_volta**: o resultado oficial discorda da posição na última volta. É esperado
  para quem abandonou e para penalidades ou desqualificações pós-corrida (medida, não erro).

**Desconhecidos**: `pit_in`, `pit_out`, `track_status` e `session_time_ms` nulos são voltas gravadas
antes de existirem essas colunas, que a limpeza trata como "não sei" (e descarta).
"""


def build_report(
    qualities: Sequence[RaceQuality], *, markdown: bool, generated_on: date | None = None
) -> str:
    comp, anom = composition_table(qualities), anomaly_table(qualities)
    if markdown:
        day = (generated_on or date.today()).isoformat()
        return (
            "# Qualidade dos dados de voltas\n\n"
            "Gerado por `python -m strategy.quality --markdown docs/qualidade-dos-dados.md` em "
            f"{day}, sobre {len(qualities)} corridas. É uma inspeção dos dados **brutos**: não "
            "decide o que excluir (isso é da limpeza das voltas, `strategy/cleaning.py`).\n\n"
            f"{MARKDOWN_LEGEND}\n"
            "## Composição e cobertura\n\n"
            f"{render_markdown(comp)}\n\n"
            "## Inconsistências, resultado oficial e valores desconhecidos\n\n"
            f"{render_markdown(anom)}\n"
        )
    return (
        f"COMPOSIÇÃO E COBERTURA\n{render_text(comp)}\n\n"
        f"INCONSISTÊNCIAS, RESULTADO OFICIAL E DESCONHECIDOS\n{render_text(anom)}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.quality",
        description="Inspeciona a qualidade das voltas ingeridas (sem decidir o que excluir).",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="grava as tabelas neste arquivo .md"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        qualities = load_qualities(session, args.year)
    if not qualities:
        print("Nenhuma corrida no banco. Ingira com: python -m ingestion --year 2024 --round 21")
        return 1
    print(build_report(qualities, markdown=False))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_report(qualities, markdown=True), encoding="utf-8")
        print(f"Tabelas gravadas em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
