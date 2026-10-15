"""Estratégia por volta: o resultado que o consumidor de estratégia calculou. SÓ LÊ.

Este endpoint nunca dispara cálculo nem fala com o banco de dados da corrida: o consumidor (um
processo à parte, `python -m strategy.service`) calcula quando o líder completa cada volta e grava
no Redis, e aqui só se lê. Duas razões: uma requisição HTTP não pode custar ~1 s de CPU de numpy
(e ninguém controla quantas chegam), e o resultado de uma volta tem de ser UM só, o que o WebSocket
anunciou, não um segundo cálculo que poderia divergir.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import ValidationError

from app.deps import get_strategy_store
from app.schemas import ErrorResponse, StrategyOut
from strategy.service.results import ResultStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/races", tags=["strategy"])

NOT_CALCULATED_HINT = (
    "O consumidor de estratégia calcula a volta N quando o líder a completa, a partir da volta 5, "
    "durante um replay (POST /replay/{year}/{round}/start). Confira se ele está rodando "
    "(`python -m strategy.service`)."
)


@router.get(
    "/{year}/{round}/strategy",
    response_model=StrategyOut,
    summary="Probabilidades de vitória/pódio e alertas de undercut de uma volta",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Ainda não há resultado para a corrida (ou para essa volta).",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "O resultado guardado tem um formato que esta API não entende.",
        },
    },
)
async def get_strategy(
    year: Annotated[int, Path(ge=1950, le=2100, description="Temporada, ex.: 2024")],
    round_number: Annotated[
        int, Path(alias="round", ge=1, le=50, description="Etapa do campeonato, ex.: 10")
    ],
    store: Annotated[ResultStore, Depends(get_strategy_store)],
    lap: Annotated[
        int | None,
        Query(
            ge=1,
            description="A volta do LÍDER. Sem ela, a mais recente calculada. Só existem as voltas "
            "que o consumidor chegou a calcular: uma volta é pulada se a seguinte chegou antes de "
            "o cálculo terminar.",
        ),
    ] = None,
) -> StrategyOut:
    """O resultado de estratégia do replay ATUAL da corrida: o Monte Carlo de vitória e pódio com o
    comparador "o líder atual vence" ao lado, as voltas que faltam, a confiabilidade, os alertas de
    undercut e os avisos de pista. Resultados de um replay que já passou não aparecem."""
    stored = await store.get(year, round_number, lap)
    if stored is None:
        latest = await store.get(year, round_number) if lap is not None else None
        if latest is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=f"Sem resultado de estratégia para {year}/{round_number}. "
                + NOT_CALCULATED_HINT,
            )
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=(
                f"A volta {lap} não tem resultado de estratégia; a mais recente é a "
                f"{latest.latest}. Uma volta é pulada se a seguinte chegou antes de o cálculo "
                "terminar."
            ),
        )
    try:
        return StrategyOut.model_validate_json(stored.payload)
    except ValidationError as exc:
        logger.error("Resultado de estratégia ilegível para esta versão da API: %s", exc)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="O resultado guardado tem um formato que esta versão da API não entende.",
        ) from exc
