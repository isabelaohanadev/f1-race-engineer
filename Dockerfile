FROM python:3.12-slim

# PYTHONUNBUFFERED: logs aparecem na hora no `docker compose logs`.
# PYTHONDONTWRITEBYTECODE: não gera .pyc dentro do container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Usuário não-root: se o app for comprometido, o atacante não vira root no container.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /code /cache/fastf1 \
    && chown -R appuser:appuser /code /cache

WORKDIR /code

# Copiamos só o requirements ANTES do código: enquanto as dependências não mudarem, o Docker
# reaproveita esta camada (que é a mais lenta: pandas, numpy, scipy...) em todo rebuild.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
