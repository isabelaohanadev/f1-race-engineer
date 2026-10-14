from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.deps import get_control, get_replay_manager
from app.schemas import (
    ErrorResponse,
    ReplayStatusOut,
    SetSpeedRequest,
    StartReplayRequest,
)
from replay.control import CommandResult, RedisControlPlane, ReplayInfo
from replay.errors import InvalidSpeed, RaceNotFound
from replay.manager import ReplayManager

router = APIRouter(prefix="/replay", tags=["replay"])

Year = Annotated[int, Path(ge=1950, le=2100, description="Temporada, ex.: 2024")]
Round = Annotated[int, Path(alias="round", ge=1, le=50, description="Etapa, ex.: 21")]

_ERRORS = {
    status.HTTP_409_CONFLICT: {"model": ErrorResponse, "description": "Estado incompatível."},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Redis fora."},
    status.HTTP_504_GATEWAY_TIMEOUT: {
        "model": ErrorResponse,
        "description": "O processo dono do replay não confirmou o comando a tempo.",
    },
}


def to_status_out(info: ReplayInfo) -> ReplayStatusOut:
    s = info.status
    return ReplayStatusOut(
        state=s.state.value,  # type: ignore[arg-type]
        speed=s.speed,
        race_time_ms=s.race_time_ms,
        lap=s.lap,
        total_laps=s.total_laps,
        events_published=s.events_published,
        error=s.error,
        epoch=info.epoch,
        owner=info.owner,
        updated_at_ms=info.updated_at_ms,
    )


def _result_out(result: CommandResult) -> ReplayStatusOut:
    if not result.ok or result.info is None:
        # O dono rejeitou o comando (ex.: velocidade inválida que passou pela validação do corpo).
        raise InvalidSpeed(result.detail or result.error or "comando rejeitado")
    return to_status_out(result.info)


@router.post(
    "/{year}/{round}/start",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReplayStatusOut,
    responses={
        **_ERRORS,
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Corrida não ingerida."},
        422: {
            "model": ErrorResponse,
            "description": "Velocidade inválida, ou a corrida não tem dados para reproduzir.",
        },
    },
)
async def start_replay(
    year: Year,
    round_number: Round,
    manager: Annotated[ReplayManager, Depends(get_replay_manager)],
    control: Annotated[RedisControlPlane, Depends(get_control)],
    body: StartReplayRequest | None = None,
) -> ReplayStatusOut:
    """Inicia o replay desta corrida NESTA instância, que passa a ser a dona dele.

    `202 Accepted`: o replay roda em segundo plano (dura a corrida inteira). Só um replay por
    corrida existe de cada vez, mesmo com várias instâncias da API: um segundo `start` recebe 409.
    """
    request = body or StartReplayRequest()
    try:
        await manager.start(year, round_number, speed=request.speed)
    except RaceNotFound as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=(
                f"{exc}. Ingira-a com: python -m ingestion --year {year} --round {round_number}"
            ),
        ) from exc
    info = await control.get_info(year, round_number)  # o status que o start acabou de publicar
    if info is None:  # expirou entre o start e a leitura (Redis lento demais): trata como fora
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "status do replay indisponível")
    return to_status_out(info)


@router.get(
    "/{year}/{round}",
    response_model=ReplayStatusOut,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}, **_ERRORS},
)
async def get_replay_status(
    year: Year, round_number: Round, control: Annotated[RedisControlPlane, Depends(get_control)]
) -> ReplayStatusOut:
    """O status do replay, lido do Redis: vale de QUALQUER instância, não só da dona."""
    info = await control.get_info(year, round_number)
    if info is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Não há replay da corrida {year}/{round_number} (nunca iniciado ou expirado).",
        )
    return to_status_out(info)


@router.post("/{year}/{round}/pause", response_model=ReplayStatusOut, responses=_ERRORS)
async def pause_replay(
    year: Year, round_number: Round, control: Annotated[RedisControlPlane, Depends(get_control)]
) -> ReplayStatusOut:
    """Congela o replay NO PONTO em que está (o tempo de corrida para). Idempotente."""
    return _result_out(await control.send_command(year, round_number, "pause"))


@router.post("/{year}/{round}/resume", response_model=ReplayStatusOut, responses=_ERRORS)
async def resume_replay(
    year: Year, round_number: Round, control: Annotated[RedisControlPlane, Depends(get_control)]
) -> ReplayStatusOut:
    """Retoma de ONDE parou (não reinicia a corrida). Idempotente."""
    return _result_out(await control.send_command(year, round_number, "resume"))


@router.post("/{year}/{round}/speed", response_model=ReplayStatusOut, responses=_ERRORS)
async def set_replay_speed(
    year: Year,
    round_number: Round,
    body: SetSpeedRequest,
    control: Annotated[RedisControlPlane, Depends(get_control)],
) -> ReplayStatusOut:
    """Muda para 1x, 4x ou 16x sem reiniciar nem pular. Se estiver pausado, vale ao retomar."""
    return _result_out(await control.send_command(year, round_number, "speed", body.speed))


@router.post("/{year}/{round}/stop", response_model=ReplayStatusOut, responses=_ERRORS)
async def stop_replay(
    year: Year, round_number: Round, control: Annotated[RedisControlPlane, Depends(get_control)]
) -> ReplayStatusOut:
    """Encerra o replay (não dá para retomar; um novo `start` recomeça a corrida do zero)."""
    return _result_out(await control.send_command(year, round_number, "stop"))
