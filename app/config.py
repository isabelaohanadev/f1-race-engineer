"""Configuração via variáveis de ambiente (princípio 12-factor: config fora do código)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    # Em Docker as variáveis vêm do compose; fora dele, de um .env opcional na raiz.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "f1"
    # Sem default de propósito: falhar rápido e alto é melhor que subir com senha "padrão".
    postgres_password: str
    postgres_db: str = "f1_race_engineer"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    fastf1_cache_dir: str = ".fastf1-cache"

    @property
    def database_url(self) -> URL:
        # URL.create escapa caracteres especiais da senha (@, /, :...), o que f-string não faz.
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    """Instância única (cacheada). Nos testes, use dependency_overrides ou limpe o cache."""
    return Settings()


class RedisSettings(BaseSettings):
    """Conexão com o Redis, em uma classe SEPARADA de `Settings` de propósito.

    O worker consumidor (Fase 2) só fala com o Redis: não tem por que exigir (nem receber) a senha
    do Postgres, e `Settings()` falha sem `POSTGRES_PASSWORD`. Separar dá menos privilégio para o
    worker e deixa os testes do Redis independentes do banco. O prefixo REDIS_ vem de env_prefix:
    `host` lê REDIS_HOST, `port` lê REDIS_PORT, e assim por diante.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    # Índice do banco lógico (0-15). A aplicação usa o 0; os testes de integração usam outro
    # (ver tests/integration/conftest.py) para nunca encostar nos dados de desenvolvimento.
    db: int = 0

    @property
    def url(self) -> str:
        # Sem senha de propósito: o Redis do Compose só publica a porta em 127.0.0.1 (ver lá).
        return f"redis://{self.host}:{self.port}/{self.db}"


@lru_cache
def get_redis_settings() -> RedisSettings:
    return RedisSettings()


class WorkerSettings(BaseSettings):
    """Configuração do processo worker (`python -m worker`). Prefixo WORKER_ nas variáveis."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="WORKER_")

    # Porta do /metrics do worker. O worker é um processo à parte da API, então expõe o SEU
    # /metrics numa porta própria (ver README: por que não o modo multiprocess). 0 desliga.
    metrics_port: int = 9108


@lru_cache
def get_worker_settings() -> WorkerSettings:
    return WorkerSettings()


class StrategySettings(BaseSettings):
    """Configuração do consumidor de estratégia (`python -m strategy.service`): STRATEGY_*."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="STRATEGY_")

    # /metrics do consumidor, numa porta própria (0 desliga), como o worker (WORKER_METRICS_PORT).
    metrics_port: int = 9109
    # A janela de espera depois que o líder completa a volta (strategy/live.py), em segundos.
    wait_s: float = 0.3
    # TTL do lock do consumidor ativo: o standby assume em até este tempo depois de uma queda.
    lock_ttl_ms: int = 10_000
    # Simulações por volta (o protocolo do backtest usa 5.000).
    n_sims: int = 5_000


@lru_cache
def get_strategy_settings() -> StrategySettings:
    return StrategySettings()
