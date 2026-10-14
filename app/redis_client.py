"""Fábrica do cliente Redis assíncrono, compartilhada por replay, worker e API.

NÃO é um singleton em `lru_cache` (como o engine do SQLAlchemy em db/session.py): um cliente
assíncrono fica preso ao event loop em que foi usado pela primeira vez, e cada teste (e cada
processo) tem o seu. Quem roda o loop é dono do cliente: cria com `create_redis()` na subida
(lifespan do FastAPI, `main()` do worker) e chama `await client.aclose()` ao terminar.
"""

from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.config import RedisSettings, get_redis_settings

# Timeout de LEITURA no socket. Detecta uma conexão morta (cabo puxado, Redis travado): sem ele, uma
# leitura numa conexão "meio aberta" pendura para sempre.
SOCKET_TIMEOUT_S = 10

# Teto para o BLOCK de qualquer XREAD/XREADGROUP, em ms. REGRA: o BLOCK tem de ser MENOR que o
# SOCKET_TIMEOUT_S. Uma leitura bloqueante legitimamente fica muda até o BLOCK acabar; se o
# timeout do socket vencer antes, o redis-py levanta TimeoutError numa conexão saudável.
MAX_BLOCK_MS = 5_000


def create_redis(settings: RedisSettings | None = None) -> Redis:
    """Cria o cliente. Não conecta aqui: a 1ª conexão acontece no 1º comando (lazy).

    `decode_responses=True`: respostas voltam como `str` em vez de `bytes`. Todo evento do stream é
    JSON em texto, então decodificar uma vez aqui evita `.decode()` espalhado pelo código.

    Retry: mantemos o padrão do redis-py (3 tentativas com backoff exponencial e jitter em erros de
    conexão). Uma queda breve do Redis não derruba o publisher nem o worker; o custo é que um Redis
    realmente fora do ar leva ~4 s para levantar `redis.exceptions.ConnectionError` (medido).
    """
    settings = settings or get_redis_settings()
    return Redis(
        host=settings.host,
        port=settings.port,
        db=settings.db,
        decode_responses=True,
        # Mesmo valor do connect_timeout do Postgres (db/session.py): Redis fora do ar deve
        # falhar em 3 s, não pendurar o /health nem a subida do worker.
        socket_connect_timeout=3,
        socket_timeout=SOCKET_TIMEOUT_S,
        # Equivalente ao pool_pre_ping do SQLAlchemy: conexão ociosa por mais de 30 s leva um PING
        # antes do próximo comando e é reconectada se estiver morta (ex.: restart do Redis).
        health_check_interval=30,
        # TCP keepalive: o sistema operacional detecta o outro lado sumido mesmo sem tráfego.
        socket_keepalive=True,
    )


def create_sync_redis(settings: RedisSettings | None = None) -> SyncRedis:
    """Cliente SÍNCRONO, usado SÓ pelo collector de métricas na hora do scrape.

    Por que síncrono: `Collector.collect()` do prometheus-client é síncrono e roda numa thread (o
    endpoint /metrics é `def`, então o FastAPI o executa no threadpool, sem bloquear o event loop).
    Timeouts CURTOS e SEM retry: um Redis lento ou fora não pode segurar o scrape (o Prometheus
    desiste em ~10 s e a API perderia TODAS as métricas, não só as do Redis). O retry com backoff
    do redis-py, ótimo para o publisher, fazia um scrape com o Redis fora levar ~3,4 s (medido). A
    próxima raspagem tenta de novo, então não há o que "persistir" aqui.
    """
    settings = settings or get_redis_settings()
    return SyncRedis(
        host=settings.host,
        port=settings.port,
        db=settings.db,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=2,
        retry=Retry(NoBackoff(), 0),
    )
