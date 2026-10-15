"""Contratos da API (Pydantic). São separados dos models do banco de propósito: o banco pode
mudar (colunas, nomes) sem quebrar quem consome a API, e a API só expõe o que decidimos expor."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


# ---------------------------------------------------------------- estratégia (Fase 3)
#
# O contrato do resultado de estratégia (o MESMO documento do GET, do evento `strategy` do
# WebSocket e do que o consumidor grava no Redis: ver strategy/payload.py). As descrições abaixo
# são a documentação do campo no OpenAPI (/docs): é onde ficam as ressalvas do que o backtest mediu.


class StrategyBaselineOut(BaseModel):
    name: str = Field(description="O comparador: 'o líder atual vence'.")
    leader: str | None = Field(
        description="Quem lidera no instante do cálculo (o menor relógio entre os que completaram "
        "a volta). É quem recebe `baseline_win` = 1."
    )


class StrategyDriverOut(BaseModel):
    driver: str
    win: float = Field(description="Probabilidade de vitória segundo o Monte Carlo.")
    podium: float = Field(description="Probabilidade de pódio segundo o Monte Carlo.")
    win_error: float = Field(
        description="Erro-padrão do Monte Carlo (sqrt(p(1-p)/n)). Só o ruído da simulação: não "
        "inclui o erro do modelo."
    )
    podium_error: float
    baseline_win: float = Field(
        description="A probabilidade do comparador 'o líder atual vence': 1 para o líder, 0 para "
        "os outros. Mostre-a SEMPRE ao lado de `win`: no backtest o Monte Carlo não superou este "
        "comparador."
    )
    retire: float = Field(description="Probabilidade de abandonar até o fim.")
    expected_stops: float = Field(description="Paradas médias nas voltas que faltam.")
    source: str = Field(
        description="De onde veio o ritmo do piloto: 'modelo', 'troca recente' ou 'plano'."
    )


class StrategyForecastOut(BaseModel):
    n_sims: int
    variant: str = Field(description="A variante do coeficiente de combustível (`k`).")
    sc_probability: float = Field(description="P(há volta sob Safety Car de agora até o fim).")
    without_model: list[str] = Field(
        description="Pilotos na pista sem nenhum ritmo estimado: ficam de fora da simulação."
    )
    drivers: list[StrategyDriverOut] = Field(description="Ordenados por `win`, do maior ao menor.")


class UndercutAlertOut(BaseModel):
    attacker: str
    rival: str | None
    probability: float | None = Field(description="P(a margem do undercut ser positiva).")
    margin_s: float | None = Field(
        description="Margem esperada em segundos ao fim do horizonte (+ = o atacante termina à "
        "frente do rival, já contado o tráfego)."
    )
    margin_sd_s: float | None
    horizon_k: int | None = Field(description="O rival para daqui a K voltas.")
    pit_loss_s: float | None
    attacker_compounds: list[str | None] = Field(description="[composto atual, composto novo].")
    rival_compounds: list[str | None]
    blocked_by: str | None = Field(description="Quem prende o atacante depois da parada.")


class StrategyWarningOut(BaseModel):
    code: Literal["safety_car", "virtual_safety_car", "red_flag"]
    text: str


class StrategyInputsOut(BaseModel):
    cars_reported: int = Field(
        description="Quantos carros estavam no estado do cálculo (tinham reportado até o instante "
        "do líder). Menos que `cars_known` quer dizer carros fora da corrida ou sem relógio."
    )
    cars_known: int = Field(description="Pilotos que já apareceram no stream de voltas.")
    late_laps: int = Field(
        description="Voltas que chegaram DEPOIS do resultado do instante delas: ficam no estado "
        "para as voltas seguintes, mas não reabrem o resultado já calculado."
    )
    duplicates: int = Field(description="Entradas repetidas ignoradas (dedupe por piloto e volta).")
    wait_s: float = Field(description="A janela de espera depois que o líder completa a volta.")
    constants: str = Field(
        description="De quais corridas saíram as constantes (SC, abandonos, trocas): sem a corrida "
        "reproduzida."
    )


class StrategyOut(BaseModel):
    v: Literal[1] = Field(
        description="Versão do formato. Um cliente que vê uma versão que não conhece deve dizê-lo "
        "em vez de adivinhar."
    )
    type: Literal["strategy"]
    year: int
    round: int
    epoch: int = Field(description="Número do replay (muda a cada start).")
    lap: int = Field(description="A volta em que o LÍDER está: a que ele acabou de completar.")
    total_laps: int
    laps_left: int = Field(description="Voltas que faltam: total menos a volta do líder.")
    progress: float = Field(description="lap / total_laps.")
    as_of_ms: int | None = Field(
        description="O instante (relógio da sessão, ms) em que o líder completou a volta: o estado "
        "do cálculo só tem o que tinha acontecido até ali."
    )
    reliability: Literal["indicative", "experimental"] = Field(
        description="Nenhum dos dois valores quer dizer 'confiável'. `indicative` (até 80% da "
        "corrida): no backtest (docs/backtest.md) o Monte Carlo NÃO se distinguiu de 'o líder "
        "atual vence'. `experimental` (a partir de 80%): no backtest ele foi PIOR que o líder "
        "atual. Use `baseline_win` como a referência."
    )
    baseline: StrategyBaselineOut
    forecast: StrategyForecastOut = Field(
        description="Monte Carlo de vitória e pódio. No backtest (21 corridas secas de 2024) ele "
        "não superou 'o líder atual vence' em nenhuma fase da corrida, e o erro do tempo de volta "
        "do modelo não superou a persistência. As heurísticas de parada e de ultrapassagem não "
        "são medidas; VSC e bandeira vermelha ficam fora do simulador."
    )
    alerts: list[UndercutAlertOut] = Field(
        description="Alertas de undercut ligados (limiar 0,6, histerese em 0,5). RESSALVA DO VIÉS "
        "DE SELEÇÃO: só se observam as paradas que aconteceram, e as equipes tendem a parar "
        "quando funciona. A precisão medida é OTIMISTA, a revocação não é mensurável e a "
        "amostra é pequena. São avaliados no instante em que o líder completa a volta, a forma "
        "medida offline em docs/undercut-ao-vivo.md (10 de 12 alertas certos, intervalo de "
        "Wilson de 55% a 95%): o backtest mediu outra forma (o instante do atacante), e os "
        "números dele não valem para esta. O líder nunca é o rival: nenhum undercut contra o "
        "líder é alertado. O fator de inflação congelado (1,3) deixou o intervalo da margem "
        "curto."
    )
    warnings: list[StrategyWarningOut] = Field(
        description="SC em curso (modelado), VSC e bandeira vermelha (fora do simulador)."
    )
    inputs: StrategyInputsOut
    computed_at_ms: int = Field(description="Quando o cálculo terminou (relógio do consumidor).")
    duration_s: float | None = Field(description="Duração da simulação.")
