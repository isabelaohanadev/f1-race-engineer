"""voltas: box, status de pista e relógio da sessão

Fase 3 (estratégia): a limpeza das voltas antes de modelar o desgaste do pneu precisa saber se a
volta foi de entrada ou de saída dos boxes e se houve safety car, VSC ou bandeira vermelha. O FastF1
entrega os três (PitInTime, PitOutTime e TrackStatus), mas até aqui o loader os descartava. A
estratégia também precisa do intervalo entre carros na linha, que sai do relógio da sessão no fim da
volta (`Time`), gravado agora em `session_time_ms`: ao contrário de `completed_at`, ele não depende
de `t0_date`, que só existe com a telemetria carregada.

As colunas são NULÁVEIS e sem default de propósito: as voltas já gravadas ficam com NULL ("não sei")
até serem reingeridas, em vez de receberem um `false` que afirmaria "sem box" sem ninguém ter
olhado. A ingestão é idempotente, então reingerir uma corrida só preenche o que falta.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-14 16:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Só ADD COLUMN nulável sem default: no Postgres isso é uma mudança de catálogo (instantânea,
    # sem reescrever a tabela).
    op.add_column("laps", sa.Column("pit_in", sa.Boolean(), nullable=True))
    op.add_column("laps", sa.Column("pit_out", sa.Boolean(), nullable=True))
    op.add_column("laps", sa.Column("track_status", sa.String(length=16), nullable=True))
    op.add_column("laps", sa.Column("session_time_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("laps", "session_time_ms")
    op.drop_column("laps", "track_status")
    op.drop_column("laps", "pit_out")
    op.drop_column("laps", "pit_in")
