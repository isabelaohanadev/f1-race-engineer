"""Entrada do processo: `python -m worker`."""

import asyncio
import logging
import signal

from app.redis_client import create_redis
from worker.consumer import StateWorker


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
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
