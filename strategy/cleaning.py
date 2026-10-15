"""Limpeza das voltas: quais representam o RITMO do carro (`python -m strategy.cleaning`: o funil).

O modelo de pneu regride o tempo de volta contra a idade do pneu. Para a inclinação significar
"desgaste", só entram voltas em que o carro andou no seu ritmo, sem nada além do pneu e do
combustível mexendo no tempo. Todo o resto é descartado, por esta ordem (cada volta cai na PRIMEIRA
regra que a pega, então o funil soma sem contar nada duas vezes):

  1. desconhecido     pit_in, pit_out ou track_status nulos (volta gravada antes de existirem):
                      ausência de informação não é informação de ausência, então não se assume
                      "pista livre, sem box";
  2. sem_tempo        sem tempo de volta (a variável de resposta não existe);
  3. volta_1          largada: grid, combustível cheio, pneu frio e tráfego não dizem nada do ritmo;
  4. box              volta de entrada ou de saída dos boxes: lenta por definição (e, nas corridas
                      secas, 97% delas TÊM tempo de volta, então a regra 2 não as pega);
  5. sem_pneu         sem composto, idade do pneu ou stint: a volta não cabe no eixo do modelo;
  6. pista_nao_verde  status diferente de "1" por qualquer motivo além da amarela (safety car, VSC,
                      bandeira vermelha, ...);
  7. so_amarela       status só de verde e amarela ("12", "21", "2"): a amarela fica por último de
                      propósito, para o seu custo ser o MARGINAL, o que ela tira além do resto.

A REGRA DO STATUS É UMA SÓ: fica quem tem o status exatamente "1". Não procuramos "os códigos ruins"
um por um (4, 5, 6, 7, e os que a F1 inventar amanhã): o status vem como TEXTO com um código por
ocorrência na volta ("1", "12", "26", "671"...), então qualquer coisa além de "1" é uma volta em
que a pista não esteve livre a volta toda. A divisão entre as regras 6 e 7 existe SÓ para o
relatório mostrar quanto a amarela custa separadamente; ela não participa da decisão de excluir.
"""

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path

from db.session import get_session_factory
from strategy.laps import DRY_COMPOUNDS, LapRow, list_races, load_race_laps, race_label
from strategy.tables import pct, render_markdown, render_text

GREEN = "1"
# Códigos que, sozinhos, só indicam pista livre (1) ou amarela (2). Serve apenas para o relatório
# separar o custo da amarela (regra 7); a decisão de excluir é `!= GREEN`.
_GREEN_OR_YELLOW = frozenset({"1", "2"})

# "Se alguma corrida perder mais de 40% das voltas, sinalize": uma corrida que perde tanto deixa o
# modelo com pouca coisa, e o número que ele devolver não deve ser lido como se fosse firme.
MAX_LOSS_SHARE = 0.40
# Abaixo disto, as voltas úteis são de pista molhada: não há modelo de pneu de seco a ajustar.
MIN_DRY_SHARE = 0.50


class Reason(StrEnum):
    DESCONHECIDO = "desconhecido"
    SEM_TEMPO = "sem_tempo"
    VOLTA_1 = "volta_1"
    BOX = "box"
    SEM_PNEU = "sem_pneu"
    PISTA_NAO_VERDE = "pista_nao_verde"
    SO_AMARELA = "so_amarela"


FUNNEL_ORDER = tuple(Reason)


def exclusion_reason(lap: LapRow) -> Reason | None:
    """Por que a volta NÃO entra no modelo (a primeira regra que a pega), ou None se entra."""
    if lap.pit_in is None or lap.pit_out is None or lap.track_status is None:
        return Reason.DESCONHECIDO
    if lap.lap_time_ms is None:
        return Reason.SEM_TEMPO
    if lap.lap_number == 1:
        return Reason.VOLTA_1
    if lap.pit_in or lap.pit_out:
        return Reason.BOX
    if lap.compound is None or lap.tyre_life is None or lap.stint is None:
        return Reason.SEM_PNEU
    if lap.track_status != GREEN:
        # A decisão de excluir é a linha acima (`!= GREEN`); aqui só se rotula o motivo, para o
        # relatório separar o custo da amarela do das demais.
        if set(lap.track_status) <= _GREEN_OR_YELLOW:
            return Reason.SO_AMARELA
        return Reason.PISTA_NAO_VERDE
    return None


def clean_laps(rows: Sequence[LapRow]) -> list[LapRow]:
    """As voltas que representam o ritmo do carro."""
    return [lap for lap in rows if exclusion_reason(lap) is None]


@dataclass(slots=True)
class Funnel:
    """Quantas voltas cada regra tira numa corrida, na ordem de FUNNEL_ORDER."""

    label: str
    total: int = 0
    removed: dict[Reason, int] = field(default_factory=lambda: dict.fromkeys(FUNNEL_ORDER, 0))
    kept: int = 0
    kept_dry: int = 0  # úteis em SOFT, MEDIUM ou HARD

    @property
    def loss_share(self) -> float:
        return 1 - self.kept / self.total if self.total else 0.0

    @property
    def loss_share_if_yellow_kept(self) -> float:
        """A perda se a amarela fosse mantida: o custo do status "só amarela", à parte do resto."""
        if not self.total:
            return 0.0
        return 1 - (self.kept + self.removed[Reason.SO_AMARELA]) / self.total

    @property
    def dry_share_of_kept(self) -> float:
        return self.kept_dry / self.kept if self.kept else 0.0

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.loss_share > MAX_LOSS_SHARE:
            out.append(f"perde {100 * self.loss_share:.0f}% das voltas")
        if self.kept and self.dry_share_of_kept < MIN_DRY_SHARE:
            out.append("sem pista seca")
        elif self.kept and self.kept_dry < self.kept:
            # Seco e molhado na mesma corrida: a pista seca durante o stint e o tempo de volta cai
            # por isso, não por falta de desgaste. Péssima para validar um modelo de pneu.
            out.append(f"pista mista ({pct(1 - self.dry_share_of_kept)} molhado)")
        if not self.kept:
            out.append("nenhuma volta útil")
        return out


def build_funnel(label: str, rows: Sequence[LapRow]) -> Funnel:
    funnel = Funnel(label=label, total=len(rows))
    counts = Counter(exclusion_reason(lap) for lap in rows)
    for reason in FUNNEL_ORDER:
        funnel.removed[reason] = counts[reason]
    funnel.kept = counts[None]
    funnel.kept_dry = sum(
        1 for lap in rows if exclusion_reason(lap) is None and lap.compound in DRY_COMPOUNDS
    )
    return funnel


# ---------------------------------------------------------------- relatório


def funnel_table(funnels: Sequence[Funnel]) -> list[list[str]]:
    header = [
        "corrida",
        "voltas",
        *[r.value for r in FUNNEL_ORDER],
        "úteis",
        "perda",
        "perda se a amarela ficasse",
        "alerta",
    ]
    rows = [header]
    for f in funnels:
        rows.append(
            [
                f.label,
                str(f.total),
                *[str(f.removed[r]) for r in FUNNEL_ORDER],
                str(f.kept),
                pct(f.loss_share),
                pct(f.loss_share_if_yellow_kept),
                "; ".join(f.warnings) or "-",
            ]
        )
    return rows


def _summary(funnels: Sequence[Funnel]) -> list[str]:
    """Os números que a leitura do funil precisa, calculados (nunca escritos à mão)."""
    total = sum(f.total for f in funnels)
    removed = {r: sum(f.removed[r] for f in funnels) for r in FUNNEL_ORDER}
    kept = sum(f.kept for f in funnels)
    flagged = [f.label for f in funnels if f.loss_share > MAX_LOSS_SHARE]
    lines = [
        f"- Ao todo: {total} voltas, {kept} úteis ({pct(kept / total) if total else '-'}).",
        "- Tirado por cada regra (na ordem, sem contar duas vezes): "
        + ", ".join(f"{r.value} {removed[r]}" for r in FUNNEL_ORDER)
        + ".",
        f"- A amarela, sozinha, custa {removed[Reason.SO_AMARELA]} voltas "
        f"({pct(removed[Reason.SO_AMARELA] / total) if total else '-'} do total).",
        f"- Corridas que perdem mais de {pct(MAX_LOSS_SHARE)}: "
        + (", ".join(flagged) if flagged else "nenhuma")
        + ".",
    ]
    return lines


MARKDOWN_LEGEND = """\
## Como ler

Cada volta cai na **primeira** regra que a pega, na ordem das colunas, então a linha soma
(`voltas` = soma das regras + `úteis`) e nada é contado duas vezes.

- **desconhecido**: `pit_in`, `pit_out` ou `track_status` nulos (volta gravada antes de existirem).
- **sem_tempo**: sem tempo de volta.
- **volta_1**: a largada.
- **box**: volta de entrada ou de saída dos boxes.
- **sem_pneu**: sem composto, idade do pneu ou stint.
- **pista_nao_verde**: status diferente de `1` por algo além da amarela (SC, VSC, vermelha...).
- **so_amarela**: status só de verde e amarela (`12`, `21`, `2`). Fica por último para a coluna
  ser o custo **marginal** da amarela: o que ela tira além de todas as outras regras.
- **perda se a amarela ficasse**: a perda que a corrida teria se a amarela fosse mantida.
- **alerta**: `perde N%` se a corrida perde mais de 40% das voltas; `sem pista seca` se menos da
  metade das voltas úteis é de SOFT, MEDIUM ou HARD (não há desgaste de pneu seco a modelar);
  `pista mista` se há voltas úteis de pista molhada junto com as de seco (a pista seca durante o
  stint, e o tempo de volta cai por isso, não por falta de desgaste).

A regra de exclusão do status é uma só: fica quem tem o status exatamente `1`. A divisão entre
`pista_nao_verde` e `so_amarela` existe só para este relatório.
"""


def build_report(
    funnels: Sequence[Funnel], *, markdown: bool, generated_on: date | None = None
) -> str:
    table = funnel_table(funnels)
    if markdown:
        day = (generated_on or date.today()).isoformat()
        return (
            "# Limpeza das voltas: o funil de exclusão\n\n"
            "Gerado por `python -m strategy.cleaning --markdown docs/limpeza-das-voltas.md` em "
            f"{day}, sobre {len(funnels)} corridas. A regra está em `strategy/cleaning.py`.\n\n"
            "## Resumo\n\n" + "\n".join(_summary(funnels)) + "\n\n"
            f"{MARKDOWN_LEGEND}\n"
            "## Funil por corrida\n\n"
            f"{render_markdown(table)}\n"
        )
    return f"FUNIL DE EXCLUSÃO\n{render_text(table)}\n\n" + "\n".join(_summary(funnels)) + "\n"


def load_funnels(session, year: int | None = None) -> list[Funnel]:
    return [
        build_funnel(race_label(r.year, r.round, r.name), load_race_laps(session, r.id))
        for r in list_races(session, year)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strategy.cleaning",
        description="Mostra quantas voltas cada regra de limpeza tira, por corrida.",
    )
    parser.add_argument("--year", type=int, default=None, help="só esta temporada")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="grava o relatório neste arquivo .md"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with get_session_factory()() as session:
        funnels = load_funnels(session, args.year)
    if not funnels:
        print("Nenhuma corrida no banco. Ingira com: python -m ingestion --year 2024 --round 21")
        return 1
    print(build_report(funnels, markdown=False))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_report(funnels, markdown=True), encoding="utf-8")
        print(f"Relatório gravado em {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
