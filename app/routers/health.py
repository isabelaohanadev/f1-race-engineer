import asyncio
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from fastapi.concurrency import run_in_threadpool
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.deps import get_db, get_redis
from app.schemas import HealthChecks, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["infra"])

# Limite do PING no Redis. O cliente tem retry com backoff (~4 s para desistir); um health check não
# pode demorar isso: o healthcheck do Docker espera 3 s. Com 1 s, "Redis fora do ar" vira uma
# resposta rápida em vez de um timeout do próprio probe.
REDIS_HEALTH_TIMEOUT_S = 1.0

Check = Literal["ok", "unavailable"]


def _check_database(db: Session) -> Check:
    try:
        db.execute(text("SELECT 1"))
        return "ok"
    except SQLAlchemyError:
        # O detalhe do erro (host, usuário, mensagem do driver) vai para o LOG, nunca para a
        # resposta: /health costuma ser público e isso vazaria informação da infraestrutura.
        logger.exception("Health check: banco de dados inacessível")
        return "unavailable"


async def _check_redis(redis: Redis) -> Check:
    try:
        async with asyncio.timeout(REDIS_HEALTH_TIMEOUT_S):
            await redis.ping()
        return "ok"
    except (RedisError, OSError, TimeoutError):
        logger.warning("Health check: Redis inacessível", exc_info=True)
        return "unavailable"


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
async def health(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> HealthResponse:
    """Distingue "o processo está vivo" de "as dependências estão acessíveis".

    Os estados são independentes: se este endpoint responde, o processo está vivo (por isso
    `process` é sempre "ok"). O corpo diz QUAL dependência falhou, mas só o BANCO muda o código
    HTTP:

    * banco fora -> 503: a API REST inteira depende dele, "não mande tráfego para mim";
    * Redis fora -> continua 200, com `status: "degraded"` e `redis: "unavailable"` no corpo. Só o
      streaming (replay/WebSocket) para; as rotas REST seguem funcionando, e tirar a instância do
      balanceador por causa disso derrubaria o que ainda funciona.

    O endpoint é `async` para checar o Redis sem bloquear o loop; o banco (driver síncrono) roda
    numa thread, no mesmo threadpool que o FastAPI usaria para um endpoint `def`.
    """
    database = await run_in_threadpool(_check_database, db)
    redis_status = await _check_redis(redis)

    if database != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if database == redis_status == "ok" else "degraded",
        checks=HealthChecks(process="ok", database=database, redis=redis_status),
    )
