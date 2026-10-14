"""schema inicial

Tabelas relacionais (teams, drivers, races, race_entries, laps) geradas via autogenerate a
partir de db/models.py, mais os trechos específicos do TimescaleDB, que o Alembic não conhece:
a extensão e a conversão de `position_samples` em hypertable.

Revision ID: 0001
Revises:
Create Date: 2026-10-14 01:25:27.319829

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A imagem timescale/timescaledb já traz os binários; aqui garantimos que a extensão está
    # habilitada NESTE banco. IF NOT EXISTS torna o passo idempotente.
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "drivers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=3), nullable=False),
        sa.Column("full_name", sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drivers")),
        sa.UniqueConstraint("code", name=op.f("uq_drivers_code")),
    )
    op.create_table(
        "races",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("location", sa.String(length=100), nullable=True),
        sa.Column("country", sa.String(length=100), nullable=True),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_laps", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_races")),
        sa.UniqueConstraint("year", "round", name=op.f("uq_races_year_round")),
    )
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("color_hex", sa.String(length=6), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_teams")),
        sa.UniqueConstraint("name", name=op.f("uq_teams_name")),
    )
    op.create_table(
        "race_entries",
        sa.Column("race_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("driver_number", sa.Integer(), nullable=False),
        sa.Column("grid_position", sa.Integer(), nullable=True),
        sa.Column("finish_position", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(
            ["driver_id"], ["drivers.id"], name=op.f("fk_race_entries_driver_id_drivers")
        ),
        sa.ForeignKeyConstraint(
            ["race_id"],
            ["races.id"],
            name=op.f("fk_race_entries_race_id_races"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name=op.f("fk_race_entries_team_id_teams")
        ),
        sa.PrimaryKeyConstraint("race_id", "driver_id", name=op.f("pk_race_entries")),
        sa.UniqueConstraint(
            "race_id", "driver_number", name=op.f("uq_race_entries_race_id_driver_number")
        ),
    )
    op.create_table(
        "laps",
        sa.Column("race_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("driver_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("lap_number", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("lap_time_ms", sa.Integer(), nullable=True),
        sa.Column("compound", sa.String(length=20), nullable=True),
        sa.Column("tyre_life", sa.Integer(), nullable=True),
        sa.Column("stint", sa.Integer(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["race_id", "driver_id"],
            ["race_entries.race_id", "race_entries.driver_id"],
            name=op.f("fk_laps_race_id_driver_id_race_entries"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("race_id", "driver_id", "lap_number", name=op.f("pk_laps")),
    )
    op.create_table(
        "position_samples",
        sa.Column("race_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("driver_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("x", sa.Integer(), nullable=False),
        sa.Column("y", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["race_id", "driver_id"],
            ["race_entries.race_id", "race_entries.driver_id"],
            name=op.f("fk_position_samples_race_id_driver_id_race_entries"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("race_id", "driver_id", "time", name=op.f("pk_position_samples")),
    )

    # Hypertable: o Timescale particiona a tabela em "chunks" por faixa de `time`, mas ela segue
    # sendo consultada como uma tabela normal. Por que isso ajuda aqui:
    #  * escrita: o INSERT em lote da ingestão toca só o chunk "quente" (índices pequenos, que
    #    cabem em memória), em vez de um único índice gigante com anos de telemetria;
    #  * leitura: consultas por janela de tempo só abrem os chunks relevantes (chunk exclusion);
    #  * ciclo de vida: apagar/comprimir telemetria antiga vira operação por chunk (instantânea),
    #    em vez de DELETE linha a linha seguido de VACUUM.
    # chunk de 7 dias: o calendário da F1 tem no máximo ~1 corrida por semana, então cada corrida
    # cai em (quase sempre) um único chunk de ~650 mil linhas, um tamanho confortável.
    # create_default_indexes => FALSE: o índice automático só em `time` não atende nenhuma consulta
    # nossa (todas filtram por corrida) e só encareceria a escrita; criamos o índice certo abaixo.
    # Usamos a API by_range (a assinatura antiga com time_column_name está deprecated desde 2.13).
    # A tabela está vazia, então não precisa de migrate_data.
    op.execute(
        "SELECT create_hypertable('position_samples', by_range('time', INTERVAL '7 days'), "
        "create_default_indexes => FALSE, if_not_exists => TRUE)"
    )
    op.create_index(
        "ix_position_samples_race_id_time", "position_samples", ["race_id", "time"], unique=False
    )

    # Compressão e retenção ficaram de fora de propósito: re-ingerir uma corrida já comprimida
    # exigiria descomprimir o chunk. Entram quando houver necessidade e testes contra o Timescale.


def downgrade() -> None:
    # DROP TABLE numa hypertable remove também todos os chunks.
    op.drop_index("ix_position_samples_race_id_time", table_name="position_samples")
    op.drop_table("position_samples")
    op.drop_table("laps")
    op.drop_table("race_entries")
    op.drop_table("teams")
    op.drop_table("races")
    op.drop_table("drivers")
    # A extensão NÃO é removida: ela é do banco inteiro e outras migrations/objetos podem usá-la.
