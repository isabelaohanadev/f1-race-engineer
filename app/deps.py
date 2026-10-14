from collections.abc import Iterator

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.orm import Session

from db.session import get_session_factory
from replay.control import RedisControlPlane
from replay.manager import ReplayManager


def get_db() -> Iterator[Session]:
    """Dependência do FastAPI: uma sessão por request, sempre fechada ao final.

    A sessão só abre conexão com o banco no primeiro uso (lazy). Por isso o /health consegue
    capturar a falha de conexão DENTRO do endpoint e responder 503, em vez de explodir antes.

    Nos testes, `app.dependency_overrides[get_db]` troca o banco real por um de teste.
    """
    with get_session_factory()() as session:
        yield session


def get_redis(request: Request) -> Redis:
    """Dependência do FastAPI: o cliente Redis ÚNICO da aplicação, criado no lifespan (main.py).

    Um cliente por processo (e não por request): ele já mantém um pool de conexões. Nos testes,
    `app.dependency_overrides[get_redis]` troca o Redis real por um de teste.
    """
    return request.app.state.redis


def get_replay_manager(request: Request) -> ReplayManager:
    """O gerenciador de replays DESTA instância (criado no lifespan). Só serve para `start`: o
    resto do controle vai pelo Redis (ver get_control), para funcionar de qualquer instância."""
    return request.app.state.replay_manager


def get_control(request: Request) -> RedisControlPlane:
    """O plano de controle (status e comandos) sobre o Redis: o mesmo em todas as instâncias."""
    return request.app.state.control
