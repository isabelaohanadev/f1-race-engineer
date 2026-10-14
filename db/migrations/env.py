from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.config import get_settings
from db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata dos models: base do `alembic revision --autogenerate` e do `alembic check`.
target_metadata = Base.metadata

# Passamos a URL direto ao SQLAlchemy em vez de usar config.set_main_option(): o alembic.ini usa
# interpolação com "%", o que quebra senhas com caracteres especiais (que viram %40, %2F...).
_url = get_settings().database_url


def run_migrations_offline() -> None:
    """Gera o SQL sem conectar ao banco (`alembic upgrade head --sql`). Útil para revisar."""
    context.configure(
        url=_url.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # NullPool: a migration é um processo curto e único; não faz sentido manter pool aberto.
    connectable = create_engine(_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Detecta mudança de tipo de coluna (ex.: String(50) -> String(100)) no autogenerate.
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
