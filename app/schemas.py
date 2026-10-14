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
