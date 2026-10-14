import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.deps import get_db
from app.schemas import HealthChecks, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["infra"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "O processo está vivo, mas o banco está inacessível.",
        }
    },
)
def health(response: Response, db: Annotated[Session, Depends(get_db)]) -> HealthResponse:
    """Distingue "o processo está vivo" de "o banco está acessível".

    Os dois estados são independentes: se este endpoint responde, o processo está vivo (por isso
    `process` é sempre "ok"). O banco é verificado com um `SELECT 1`. Se falhar, devolvemos 503,
    o código que balanceadores e orquestradores entendem como "não mande tráfego para mim agora",
    e o corpo continua dizendo QUAL dependência falhou.
    """
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError:
        # O detalhe do erro (host, usuário, mensagem do driver) vai para o LOG, nunca para a
        # resposta: /health costuma ser público e isso vazaria informação da infraestrutura.
        logger.exception("Health check: banco de dados inacessível")
        database = "unavailable"

    if database != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        checks=HealthChecks(process="ok", database=database),
    )
