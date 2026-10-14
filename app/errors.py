"""Tradução dos erros de domínio do replay (e do Redis) em respostas HTTP."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from replay.errors import (
    InvalidSpeed,
    RaceHasNoData,
    ReplayAlreadyRunning,
    ReplayNotRunning,
    ReplayTimeout,
)

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: dict[type[Exception], int] = {
    ReplayAlreadyRunning: status.HTTP_409_CONFLICT,  # já há um replay desta corrida
    ReplayNotRunning: status.HTTP_409_CONFLICT,  # não há replay ativo para pausar/retomar/...
    RaceHasNoData: 422,  # a corrida existe mas está vazia
    InvalidSpeed: 422,
    ReplayTimeout: status.HTTP_504_GATEWAY_TIMEOUT,  # o dono do replay não respondeu a tempo
}


def register_exception_handlers(app: FastAPI) -> None:
    def make_handler(code: int):
        async def handler(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=code)

        return handler

    for error_type, code in _STATUS_BY_ERROR.items():
        app.add_exception_handler(error_type, make_handler(code))

    @app.exception_handler(RedisError)
    async def redis_unavailable(request: Request, exc: RedisError) -> JSONResponse:
        # O detalhe (host, porta, mensagem do driver) vai para o LOG, nunca para a resposta.
        logger.error("Redis indisponível ao atender %s", request.url.path, exc_info=exc)
        return JSONResponse(
            {"detail": "Redis indisponível: o streaming está fora do ar. Tente novamente."},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
