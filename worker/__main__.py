"""Entrada do processo: `python -m worker`."""

import asyncio
import logging
import signal

from prometheus_client import start_http_server

from app.config import get_worker_settings
from app.redis_client import create_redis
from worker import metrics
from worker.consumer import StateWorker


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    port = get_worker_settings().metrics_port
    if port:  # 0 desliga. O servidor roda numa thread própria e serve o registro DESTE processo.
        start_http_server(port, registry=metrics.REGISTRY)
        logging.getLogger(__name__).info("Métricas do worker em :%d/metrics", port)
    redis = create_redis()
    worker = StateWorker(redis)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # `docker stop` manda SIGTERM: termina o lote em andamento e sai (não é morto no meio).
        loop.add_signal_handler(sig, stop.set)
    try:
        await worker.run(stop)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
