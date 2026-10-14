"""Modelos ORM (SQLAlchemy 2.0 tipado com Mapped[]).

Visão geral do schema (1 corrida = 1 linha em `races`):

    teams ──┐
            ├── race_entries (quem correu, por qual equipe, com qual número) ──┬── laps
    drivers ┘          ▲                                                       └── position_samples
    races ─────────────┘                                                           (hypertable)

Decisões de design (boas para explicar em entrevista):

* Chaves naturais compostas em `laps` e `position_samples` (em vez de um `id` serial): a própria
  chave identifica a linha, então re-ingerir a mesma corrida cai em `ON CONFLICT` e não duplica
  nada (idempotência). Um `id` serial permitiria duplicatas silenciosas.
* A equipe fica em `race_entries`, não em `drivers`: o piloto troca de equipe entre temporadas
  (e até no meio dela). Guardar `team_id` no piloto reescreveria o passado a cada ingestão.
* Todo timestamp é `timestamptz` (ver `type_annotation_map`): evita o bug clássico de datetimes
  "ingênuos" interpretados no fuso errado. Posições e voltas usam o MESMO relógio absoluto,
  o que permite ordená-las juntas (necessário para reproduzir a corrida em ordem de tempo).
"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    MetaData,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Nomes determinísticos de constraints/índices. Sem isso o Postgres gera nomes automáticos e o
# Alembic não consegue alterá-los/removê-los por nome de forma portável entre ambientes.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # Todo `Mapped[datetime]` vira TIMESTAMP WITH TIME ZONE: impossível esquecer o timezone.
    type_annotation_map = {datetime: DateTime(timezone=True)}


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    # UNIQUE vira o alvo do upsert na ingestão (ON CONFLICT (name)) e já cria o índice de busca.
    name: Mapped[str] = mapped_column(String(100), unique=True)
    # Cor oficial da equipe sem '#', ex.: "3671C6" (usada depois para colorir os carros).
    color_hex: Mapped[str | None] = mapped_column(String(6))


class Driver(Base):
    __tablename__ = "drivers"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Abreviação de 3 letras ("VER", "HAM"). É a chave natural usada pelo FastF1 para voltas e
    # resultados, por isso é o alvo do upsert. Trade-off: assume que o código identifica o piloto.
    code: Mapped[str] = mapped_column(String(3), unique=True)
    full_name: Mapped[str] = mapped_column(String(100))


class Race(Base):
    __tablename__ = "races"

    id: Mapped[int] = mapped_column(primary_key=True)
    year: Mapped[int]
    round: Mapped[int]
    name: Mapped[str] = mapped_column(String(100))  # ex.: "São Paulo Grand Prix"
    location: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))
    starts_at: Mapped[datetime | None]
    total_laps: Mapped[int | None]

    # ÍNDICE: o endpoint é /races/{year}/{round}/laps, então toda consulta começa por
    # (year, round). Além de garantir unicidade, é o alvo do ON CONFLICT da ingestão.
    __table_args__ = (UniqueConstraint("year", "round"),)


class RaceEntry(Base):
    """Participação de um piloto numa corrida: liga piloto, equipe e número do carro."""

    __tablename__ = "race_entries"

    race_id: Mapped[int] = mapped_column(
        ForeignKey("races.id", ondelete="CASCADE"), primary_key=True
    )
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.id"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    # O FastF1 indexa a telemetria de posição pelo NÚMERO do carro, não pelo código do piloto.
    driver_number: Mapped[int]
    grid_position: Mapped[int | None]
    finish_position: Mapped[int | None]
    status: Mapped[str | None] = mapped_column(String(50))  # "Finished", "+1 Lap", "Collision"...

    # Dois pilotos não podem ter o mesmo número na mesma corrida (invariante de domínio).
    __table_args__ = (UniqueConstraint("race_id", "driver_number"),)


class Lap(Base):
    __tablename__ = "laps"

    race_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    driver_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    lap_number: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    # Inteiro em milissegundos: exato (sem erro de float), ordena e agrega fácil. Nulo em voltas
    # sem tempo válido (largada, entrada/saída dos boxes, bandeira vermelha).
    lap_time_ms: Mapped[int | None]
    # SOFT, MEDIUM, HARD, INTERMEDIATE, WET (ou UNKNOWN)
    compound: Mapped[str | None] = mapped_column(String(20))
    tyre_life: Mapped[int | None]  # voltas já rodadas com este jogo de pneus
    stint: Mapped[int | None]  # número do stint (muda a cada pit stop)
    position: Mapped[int | None]  # posição na corrida ao fim da volta
    # Instante absoluto em que a volta terminou: é o que permite ao replay (Fase 2) intercalar
    # voltas e posições na ordem correta de tempo.
    completed_at: Mapped[datetime | None]

    __table_args__ = (
        # ÍNDICE (a própria PK): (race_id, driver_id, lap_number) atende "todas as voltas da
        # corrida" (prefixo race_id), "voltas de um piloto" e "volta N de um piloto". Também é o
        # alvo do upsert. Não criamos índice extra por tempo: uma corrida tem ~1.400 voltas, e
        # ordenar 1.400 linhas em memória é mais barato que manter um índice a mais em cada escrita.
        # A FK composta garante que só existem voltas de quem realmente participou da corrida.
        ForeignKeyConstraint(
            ["race_id", "driver_id"],
            ["race_entries.race_id", "race_entries.driver_id"],
            ondelete="CASCADE",
        ),
    )


class PositionSample(Base):
    """Amostra de posição (x, y) de um carro. É a tabela grande: ~800 mil linhas por corrida
    (medido no GP do Brasil 2024: 797.642 linhas, ~75 MB com índices).

    Vira uma HYPERTABLE do TimescaleDB na migration (particionada por `time`).
    """

    __tablename__ = "position_samples"

    # A ordem das colunas da PK importa: (race_id, driver_id, time) deixa o índice utilizável por
    # prefixo para "corrida X", "corrida X + piloto Y" e "corrida X + piloto Y + janela de tempo".
    race_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    driver_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    # Coluna de particionamento da hypertable. O Timescale exige que ela faça parte de toda
    # chave única, por isso `time` está na PK.
    time: Mapped[datetime] = mapped_column(primary_key=True)
    # Coordenadas do circuito em décimos de metro (unidade do FastF1); cabem em int32 com folga.
    x: Mapped[int]
    y: Mapped[int]

    __table_args__ = (
        ForeignKeyConstraint(
            ["race_id", "driver_id"],
            ["race_entries.race_id", "race_entries.driver_id"],
            ondelete="CASCADE",
        ),
        # ÍNDICE 1 (PK: race_id, driver_id, time): trajetória de UM piloto numa janela de tempo
        # ("onde estava o VER entre 14:10 e 14:12?") e alvo do ON CONFLICT da ingestão.
        # ÍNDICE 2 abaixo: leitura da corrida INTEIRA em ordem de tempo (usado pelo replay).
        # Sem ele, ordenar ~800 mil linhas de todos os pilotos por `time` exigiria um sort enorme.
        # Desativamos o índice automático só em `time` (create_default_indexes=false na migration):
        # nenhuma consulta nossa filtra por tempo sem filtrar a corrida, então ele só custaria
        # escrita. Cada índice a mais deixa o INSERT em lote mais lento.
        Index("ix_position_samples_race_id_time", "race_id", "time"),
    )
