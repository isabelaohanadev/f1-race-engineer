"""A "verdade" do backtest: a ordem NA PISTA, com o resultado oficial reportado ao lado.
`python -m strategy.results` mostra, corrida a corrida, onde os dois divergem.

DECISÃO (registrada na Fase 3, a pedido, antes da etapa do backtest)
O backtest mede a previsão de vitória e de pódio contra a ORDEM NA PISTA: mais voltas completas
primeiro; empatando, quem cruzou a linha antes (a mesma regra de `replay.state.classify`). O
resultado oficial (`race_entries.finish_position`) é reportado AO LADO, nunca no lugar.

POR QUE NA PISTA
* O que o modelo simula é a corrida na pista: ritmo, pneu, paradas, safety car. Desqualificações e
  penalidades de tempo aplicadas DEPOIS da bandeira são decididas fora da pista, por pessoas, e
  nenhuma informação disponível na volta N as prevê. Avaliar a probabilidade de vitória contra o
  resultado oficial puniria o modelo por algo que ele não tem como saber.
* O caso que expôs o problema: no GP da Bélgica de 2024 o Russell venceu na pista e consta como
  `Disqualified`, o que desloca o pódio e as posições dos outros 19. Com o oficial como verdade, um
  modelo que previu a vitória do Russell (certa, na pista) seria contado como errado.
* Penalidades cumpridas DURANTE a corrida (passagem pelos boxes, parada e segue) já estão nos
  tempos de volta, então a ordem na pista as inclui. O que ela não inclui são os acréscimos de
  tempo aplicados ao resultado final.

O OFICIAL AO LADO: o relatório mostra os dois e o backtest calcula a pontuação com os dois. Em 23
das 24 corridas de 2024 o vencedor é o mesmo (ver docs/verdade-do-backtest.md).
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from db.session import get_session_factory
from strategy.laps import (
    EntryRow,
    LapRow,
    list_races,
    load_race_entries,
    load_race_laps,
    race_label,
)
from strategy.tables import render_markdown, render_text

# Status que são rotina (terminou, volta a menos, abandonou). Qualquer outro (Disqualified, Did not
# start...) aparece nas observações da corrida: é onde o oficial pode divergir da pista.
ROUTINE_STATUSES = frozenset({"Finished", "Lapped", "Retired"})
PODIUM = 3


def on_track_order(rows: Sequence[LapRow]) -> list[str]:
    """Pilotos na ordem de chegada NA PISTA. Quem não tem nenhuma volta com relógio (não largou) não
    entra: não há como ordená-lo, e a ordem oficial é que o põe no fim."""
    last: dict[str, LapRow] = {}
    for row in rows:
        if row.session_time_ms is None:
            continue
        current = last.get(row.driver)
        if current is None or row.lap_number > current.lap_number:
            last[row.driver] = row
    ranked = sorted(last.items(), key=lambda kv: (-kv[1].lap_number, kv[1].session_time_ms, kv[0]))
    return [driver for driver, _ in ranked]


def official_order(entries: Sequence[EntryRow]) -> list[str]:
    """Pilotos na ordem do resultado oficial (os sem posição final ficam de fora)."""
    classified = [e for e in entries if e.finish_position is not None]
    return [e.driver for e in sorted(classified, key=lambda e: (e.finish_position, e.driver))]


@dataclass(slots=True)
class ResultComparison:
    label: str
    on_track: list[str]
    official: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def winner_on_track(self) -> str | None:
        return self.on_track[0] if self.on_track else None

    @property
    def winner_official(self) -> str | None:
        return self.official[0] if self.official else None

    @property
    def podium_same(self) -> bool:
        """Mesmos três pilotos na mesma ordem."""
        return self.on_track[:PODIUM] == self.official[:PODIUM]

    @property
    def positions_differ(self) -> int:
        """Pilotos cuja posição muda entre as duas ordens, comparando só quem está nas duas (quem
        não largou não tem posição na pista, e contá-lo seria dizer que todo mundo mudou)."""
        common = set(self.on_track) & set(self.official)
        track = [d for d in self.on_track if d in common]
        official = [d for d in self.official if d in common]
        return sum(1 for a, b in zip(track, official, strict=True) if a != b)

    @property
    def compared(self) -> int:
        return len(set(self.on_track) & set(self.official))


def compare_results(
    label: str, rows: Sequence[LapRow], entries: Sequence[EntryRow]
) -> ResultComparison:
    notes = [
        f"{e.driver}: {e.status}"
        for e in sorted(entries, key=lambda e: e.driver)
        if e.status is not None and e.status not in ROUTINE_STATUSES
    ]
    return ResultComparison(label, on_track_order(rows), official_order(entries), notes)


# ---------------------------------------------------------------- relatório


def results_table(comparisons: Sequence[ResultComparison]) -> list[list[str]]:
    rows = [
        [
            "corrida",
            "vencedor na pista",
            "vencedor oficial",
            "pódio igual",
            "posições que diferem",
            "observações (status incomuns)",
        ]
    ]
    for c in comparisons:
        rows.append(
            [
                c.label,
                c.winner_on_track or "-",
                c.winner_official or "-",
                "sim" if c.podium_same else "NÃO",
                f"{c.positions_differ} de {c.compared}",
                "; ".join(c.notes) or "-",
            ]
        )
    return rows


def _summary(comparisons: Sequence[ResultComparison]) -> list[str]:
    total = len(comparisons)
    winners_differ = [c.label for c in comparisons if c.winner_on_track != c.winner_official]
    podium_differs = [c.label for c in comparisons if not c.podium_same]
    return [
        f"- O vencedor é o mesmo em {total - len(winners_differ)} de {total} corridas"
        + (f" (difere em: {', '.join(winners_differ)})." if winners_differ else "."),
        f"- O pódio (três pilotos, na ordem) é o mesmo em {total - len(podium_differs)} de {total}"
        + (f" (difere em: {', '.join(podium_differs)})." if podium_differs else "."),
    ]


MARKDOWN_DECISION = """\
## A decisão

O backtest mede a previsão de vitória e de pódio contra a **ordem na pista**. O **resultado
oficial é reportado ao lado**, nunca no lugar.

- **Ordem na pista**: mais voltas completas primeiro; empatando, quem cruzou a linha antes (a mesma
  regra da classificação do replay). Quem não largou não tem posição na pista.
- **Por que**: o modelo simula a corrida na pista (ritmo, pneu, paradas, safety car).
  Desqualificações e acréscimos de tempo aplicados depois da bandeira são decididos fora da pista e
  nada que se saiba na volta N os prevê: medir contra o oficial puniria o modelo por algo que ele
  não tem como saber. Penalidades cumpridas durante a corrida já estão nos tempos de volta.
- **O caso que expôs o problema**: Bélgica 2024. O Russell venceu na pista e consta como
  `Disqualified`; com o oficial como verdade, prever a vitória dele (certa na pista) contaria como
  erro, e todas as posições dos outros pilotos se deslocam.
- **Como ler a tabela**: `posições que diferem` compara só quem aparece nas duas ordens (quem não
  largou não tem posição na pista). `observações` lista os status que não são de rotina, onde o
  oficial pode divergir da pista.
"""


def build_report(
    comparisons: Sequence[ResultComparison], *, markdown: bool, generated_on: date | None = None
) -> str:
    table = results_table(comparisons)
    summary = "\n".join(_summary(comparisons))
    if not markdown:
        return f"ORDEM NA PISTA x RESULTADO OFICIAL\n{render_text(table)}\n\n{summary}\n"
    day = (generated_on or date.today()).isoformat()
    return (
        "# A verdade do backtest: ordem na pista, com o oficial ao lado\n\n"
        "Gerado por `python -m strategy.results --markdown docs/verdade-do-backtest.md` em "
        f"{day}, sobre {len(comparisons)} corridas. A regra está em `strategy/results.py`.\n\n"
        f"## Resumo\n\n{summary}\n\n{MARKDOWN_DECISION}\n"
        f"## Corrida a corrida\n\n{render_markdown(table)}\n"
    )


def load_comparisons(session, year: int | None = None) -> list[ResultComparison]:
    return [
        compare_results(
            race_label(r.year, r.round, r.name),
            load_race_laps(session, r.id),
            load_race_entries(session, r.id),
        )
        for r in list_races(session, year)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.results",
        description="Compara a ordem na pista (a verdade do backtest) com o resultado oficial.",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="grava o relatório neste arquivo .md"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        comparisons = load_comparisons(session, args.year)
    if not comparisons:
        print("Nenhuma corrida no banco. Ingira com: python -m ingestion --year 2024 --round 21")
        return 1
    print(build_report(comparisons, markdown=False))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_report(comparisons, markdown=True), encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
