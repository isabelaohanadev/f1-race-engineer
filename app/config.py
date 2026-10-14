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
