from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics import REGISTRY

router = APIRouter(tags=["infra"])


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Métricas no formato de texto do Prometheus.

    `def` (e não `async def`) de propósito: os collectors leem o Redis com um cliente SÍNCRONO, e o
    FastAPI roda endpoints `def` numa thread do pool, então essa leitura nunca bloqueia o event
    loop que atende os WebSockets. Sem autenticação: não há dado sensível aqui, mas em produção
    restrinja o acesso na borda (proxy/rede), como se faz com qualquer /metrics."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
