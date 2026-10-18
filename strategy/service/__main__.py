"""Entrada do processo: `python -m strategy.service`."""

import asyncio
import logging
import signal

from prometheus_client import start_http_server

from app.config import get_strategy_settings
from app.redis_client import create_redis
from db.session import get_session_factory
from strategy import metrics
from strategy.protocol import frozen_sim_config
from strategy.service.constants import ConstantsProvider
from strategy.service.consumer import StrategyConsumer


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = get_strategy_settings()
    if settings.metrics_port:  # 0 desliga. Uma thread própria serve o registro DESTE processo.
        start_http_server(settings.metrics_port, registry=metrics.REGISTRY)
        logging.getLogger(__name__).info(
            "Métricas da estratégia em :%d/metrics", settings.metrics_port
        )
    redis = create_redis()
    consumer = StrategyConsumer(
        redis,
        ConstantsProvider(get_session_factory()),
        wait_s=settings.wait_s,
        lock_ttl_ms=settings.lock_ttl_ms,
        sim_config=frozen_sim_config(n_sims=settings.n_sims),
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)  # `docker stop`: sai limpo e libera o lock
    try:
        await consumer.run(stop)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
