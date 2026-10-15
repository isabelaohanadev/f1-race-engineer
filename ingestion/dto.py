"""Objetos de transferência entre o loader (FastF1) e o writer (banco).

São dataclasses simples, só com tipos nativos do Python (nada de pandas/numpy). Essa é a
fronteira que separa as duas metades da ingestão:

    FastF1 ──► loader ──► [RaceData] ──► writer ──► Postgres

Com isso o writer é testável sem rede, sem FastF1 e sem pandas (basta montar um RaceData na mão),
e trocar a fonte de dados no futuro (outro provedor, arquivos) só exige outro loader.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RaceInfo:
    year: int
    round: int
    name: str
    location: str | None
    country: str | None
    starts_at: datetime | None  # horário OFICIAL agendado, em UTC
    total_laps: int | None


@dataclass(frozen=True, slots=True)
class EntryInfo:
    """Um piloto numa corrida, com sua equipe, número do carro e resultado."""

    driver_code: str
    driver_number: int
    full_name: str
    team_name: str
    team_color: str | None  # "RRGGBB" em maiúsculas, sem '#'
    grid_position: int | None
    finish_position: int | None
    status: str | None


@dataclass(frozen=True, slots=True)
class LapInfo:
    driver_code: str
    lap_number: int
    lap_time_ms: int | None
    compound: str | None
    tyre_life: int | None
    stint: int | None
    position: int | None
    completed_at: datetime | None  # UTC
    # Fase 3. Com default `None` ("não sei") para quem monta um LapInfo sem esses dados; o loader
    # sempre os preenche (True/False e o texto do status).
    pit_in: bool | None = None  # a volta termina nos boxes
    pit_out: bool | None = None  # a volta começa nos boxes
    track_status: str | None = None  # códigos de pista da volta, ex.: "1", "12", "124"
    session_time_ms: int | None = None  # relógio da sessão no fim da volta (não precisa de t0)


@dataclass(frozen=True, slots=True)
class PositionTrack:
    """Trajetória de UM piloto em formato colunar (3 listas paralelas, já ordenadas por tempo).

    Colunar de propósito: uma corrida tem ~800 mil amostras, e um objeto Python por amostra
    custaria centenas de MB de memória. Listas de int/datetime custam uma fração disso.
    """

    driver_code: str
    times: Sequence[datetime]  # UTC, timezone-aware
    xs: Sequence[int]
    ys: Sequence[int]

    def __len__(self) -> int:
        return len(self.times)


@dataclass(frozen=True, slots=True)
class RaceData:
    race: RaceInfo
    entries: Sequence[EntryInfo]
    laps: Sequence[LapInfo]
    tracks: Sequence[PositionTrack]
    # True quando a corrida foi carregada SEM telemetria (`--laps-only`): não há t0_date, então os
    # `completed_at` das voltas vêm nulos por falta de dado, não porque a volta não tenha horário.
    # O writer então NÃO grava essa coluna, para não apagar horários que uma ingestão completa
    # anterior já deixou no banco.
    laps_only: bool = False
