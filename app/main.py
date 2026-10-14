from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.redis_client import create_redis
from app.routers import health, races


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # O cliente Redis assíncrono pertence ao event loop do servidor: nasce e morre com ele. A
    # criação não conecta (é lazy), então a API sobe mesmo com o Redis fora do ar.
    app.state.redis = create_redis()
    try:
        yield
    finally:
        await app.state.redis.aclose()


app = FastAPI(
    title="F1 Race Engineer",
    description="API sobre dados de corridas de F1 ingeridos via FastF1.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(races.router)
