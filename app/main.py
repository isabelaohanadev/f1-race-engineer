from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import metrics
from app.errors import register_exception_handlers
from app.hub import HubRegistry
from app.redis_client import create_redis, create_sync_redis
from app.routers import health, races, replay, ws
from app.routers import metrics as metrics_router
from db.session import get_session_factory
from replay.control import RedisControlPlane
from replay.manager import ReplayManager
from replay.state import RaceStateStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # O cliente Redis assíncrono pertence ao event loop do servidor: nasce e morre com ele. A
    # criação não conecta (é lazy), então a API sobe mesmo com o Redis fora do ar.
    redis = create_redis()
    app.state.redis = redis
    app.state.control = RedisControlPlane(redis)
    app.state.replay_manager = ReplayManager(
        get_session_factory(), redis, on_events_published=metrics.EVENTS_PUBLISHED.inc
    )
    app.state.state_store = RaceStateStore(redis)
    app.state.hubs = HubRegistry(redis)
    # Collectors: leem o estado ATUAL quando o Prometheus raspa. Registrados aqui (e retirados ao
    # desligar) porque dependem de objetos criados neste lifespan.
    sync_redis = create_sync_redis()
    collectors = [metrics.StreamCollector(sync_redis), metrics.ClientsCollector(app.state.hubs)]
    for collector in collectors:
        metrics.REGISTRY.register(collector)
    try:
        yield
    finally:
        for collector in collectors:
            metrics.REGISTRY.unregister(collector)
        sync_redis.close()
        # Para os replays que ESTA instância roda e libera os locks (senão a corrida ficaria
        # "em replay em outro processo" até o TTL do lock expirar).
        await app.state.hubs.close()  # solta os leitores do stream e fecha as filas dos clientes
        await app.state.replay_manager.shutdown()
        await redis.aclose()


app = FastAPI(
    title="F1 Race Engineer",
    description="API sobre dados de corridas de F1 ingeridos via FastF1, com replay em streaming.",
    version="0.2.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(health.router)
app.include_router(races.router)
app.include_router(replay.router)
app.include_router(ws.router)
app.include_router(metrics_router.router)
