"""Engine e sessões do SQLAlchemy (síncronos, com psycopg 3).

Por que síncrono: o FastF1 é síncrono e a ingestão é um job em lote; as consultas da API são
simples e o FastAPI roda endpoints `def` numa threadpool sem bloquear o event loop.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(
        get_settings().database_url,
        # pool_pre_ping: testa a conexão antes de usá-la, então um restart do Postgres não deixa
        # conexões "mortas" no pool causando um erro no primeiro request seguinte.
        pool_pre_ping=True,
        # Sem timeout, o libpq espera INDEFINIDAMENTE por um host que não responde (rede caída,
        # firewall descartando pacotes). Com 3s o /health responde 503 em vez de pendurar.
        connect_args={"connect_timeout": 3},
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    # expire_on_commit=False: objetos continuam legíveis depois do commit (sem nova ida ao banco).
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Uma transação: commit se tudo deu certo, rollback se qualquer coisa levantou exceção.

    É a base da atomicidade da ingestão: uma corrida entra inteira ou não entra.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
