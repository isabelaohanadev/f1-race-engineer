"""Contratos da API (Pydantic). São separados dos models do banco de propósito: o banco pode
mudar (colunas, nomes) sem quebrar quem consome a API, e a API só expõe o que decidimos expor."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    detail: str


class HealthChecks(BaseModel):
    process: Literal["ok"]  # se o endpoint respondeu, o processo está vivo
    database: Literal["ok", "unavailable"]
    redis: Literal["ok", "unavailable"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: HealthChecks


class RaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    year: int
    round: int
    name: str
    location: str | None
    country: str | None
    starts_at: datetime | None
    total_laps: int | None


class LapOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    driver: str  # código de 3 letras, ex.: "VER"
    lap_number: int
    lap_time_ms: int | None  # nulo em voltas sem tempo válido (largada, box, bandeira vermelha)
    compound: str | None
    tyre_life: int | None
    stint: int | None
    position: int | None
    completed_at: datetime | None  # UTC


class RaceLapsResponse(BaseModel):
    race: RaceOut
    driver: str | None  # o filtro aplicado (normalizado em maiúsculas), ou null se não houve
    laps: list[LapOut]


# ---------------------------------------------------------------- replay

# Só estas velocidades (1x, 4x, 16x): um valor fora delas é 422 já na validação do corpo.
ReplaySpeed = Literal[1, 4, 16]


class StartReplayRequest(BaseModel):
    speed: ReplaySpeed = 1


class SetSpeedRequest(BaseModel):
    speed: ReplaySpeed


class ReplayStatusOut(BaseModel):
    """O status do replay como qualquer instância da API o enxerga (vem do Redis)."""

    state: Literal["pending", "running", "paused", "finished", "stopped", "failed"]
    speed: float
    race_time_ms: int | None  # onde o replay está, no tempo da corrida (ms desde a época Unix)
    lap: int | None  # volta do líder
    total_laps: int | None
    events_published: int
    error: str | None
    epoch: int  # número do replay (muda a cada start)
    owner: str  # a instância da API que roda o motor
    updated_at_ms: int  # quando o dono publicou este status (relógio dele)
