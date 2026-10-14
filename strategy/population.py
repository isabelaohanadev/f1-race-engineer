"""Que corridas entram no backtest principal e quais ficam à parte (`strategy.population`).

POR QUE NEM TODA CORRIDA ENTRA
O modelo de pneu regride o tempo de volta contra a idade do pneu. Em pista molhada ou que seca
durante a corrida, o tempo cai por causa da pista, não da falta de desgaste, e o modelo
"descobriria" degradação negativa. Medir o erro do modelo nelas mediria o clima, não o modelo.

DECISÃO (pedida na Fase 3): o backtest PRINCIPAL usa só corridas secas (>= 99% das voltas em SOFT,
MEDIUM ou HARD). As demais são reportadas À PARTE, nunca misturadas à nota principal:
  * mista: parte seca, parte molhada. É o GP da Grã-Bretanha de 2024 (76% seca): tem voltas de
    pista seca suficientes para ajustar um modelo, e por isso mesmo é a que mais tenta ser
    incluída por engano, com a pista secando por baixo do ajuste;
  * molhada: menos da metade seca (Canadá e Brasil). Não há desgaste de pista seca a modelar.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from db.session import get_session_factory
from strategy.laps import DRY_COMPOUNDS, LapRow, list_races, load_race_laps, race_label

MIN_DRY_RACE_SHARE = 0.99  # a partir daqui a corrida é "seca" e entra no backtest principal
WET_BELOW = 0.50  # abaixo daqui a corrida é "molhada"; entre os dois, "mista"


class RaceKind(StrEnum):
    SECA = "seca"
    MISTA = "mista"
    MOLHADA = "molhada"


def dry_race_share(rows: Sequence[LapRow]) -> float:
    """Fração das voltas (com composto conhecido) em pneu de pista seca."""
    known = [lap for lap in rows if lap.compound is not None]
    return sum(lap.compound in DRY_COMPOUNDS for lap in known) / len(known) if known else 0.0


def classify_race(rows: Sequence[LapRow]) -> RaceKind:
    share = dry_race_share(rows)
    if share >= MIN_DRY_RACE_SHARE:
        return RaceKind.SECA
    return RaceKind.MISTA if share >= WET_BELOW else RaceKind.MOLHADA


@dataclass(frozen=True, slots=True)
class RaceRef:
    race_id: int
    label: str
    total_laps: int
    kind: RaceKind
    dry_share: float

    @property
    def is_main(self) -> bool:
        return self.kind is RaceKind.SECA

    @property
    def reason(self) -> str:
        """Por que a corrida está fora do backtest principal ("" se está dentro)."""
        return "" if self.is_main else f"pista {self.kind.value} ({self.dry_share:.0%} seca)"


def split_population(session, year: int | None = None) -> tuple[list[RaceRef], list[RaceRef]]:
    """(corridas do backtest principal, corridas reportadas à parte), em ordem de calendário."""
    main: list[RaceRef] = []
    separate: list[RaceRef] = []
    for race in list_races(session, year):
        rows = load_race_laps(session, race.id)
        total = race.total_laps or max((lap.lap_number for lap in rows), default=0)
        if not rows or not total:
            continue
        ref = RaceRef(
            race.id,
            race_label(race.year, race.round, race.name),
            total,
            classify_race(rows),
            dry_race_share(rows),
        )
        (main if ref.is_main else separate).append(ref)
    return main, separate


def build_report(main: Sequence[RaceRef], separate: Sequence[RaceRef]) -> str:
    lines = [f"BACKTEST PRINCIPAL ({len(main)} corridas secas)"]
    lines += [f"  {r.label:26s} {r.dry_share:5.0%} seca" for r in main]
    lines += ["", f"REPORTADAS À PARTE ({len(separate)}), fora da nota principal"]
    lines += [f"  {r.label:26s} {r.reason}" for r in separate]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.population",
        description="Mostra que corridas entram no backtest principal e quais ficam à parte.",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        main_races, separate = split_population(session, args.year)
    if not main_races and not separate:
        print("Nenhuma corrida no banco. Ingira com: python -m ingestion --year 2024 --round 10")
        return 1
    print(build_report(main_races, separate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
