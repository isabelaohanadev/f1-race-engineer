"""Relógio injetável: o motor nunca chama `time` nem `asyncio.sleep` diretamente.

Em produção, `SystemClock` usa `time.monotonic()`: o relógio monotônico só anda para frente. O
relógio de parede (`time.time()`) pode PULAR (NTP, horário de verão, VM pausada) e bagunçaria o
ritmo. Nos testes, um relógio falso avança o tempo na hora, sem esperar de verdade.
"""

import asyncio
import contextlib
import time
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float:
        """Segundos desde um ponto arbitrário; só serve para diferenças."""
        ...

    async def wait(self, timeout: float, wake: asyncio.Event) -> None:
        """Espera até `timeout` segundos OU até `wake` ser sinalizado (o que vier primeiro).

        O `wake` é o que permite interromper uma espera longa quando o usuário muda a velocidade,
        pausa ou para o replay: sem ele, trocar de 1x para 16x durante uma pausa de 30 s no ritmo de
        1x só faria efeito 30 s depois.
        """
        ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    async def wait(self, timeout: float, wake: asyncio.Event) -> None:
        # TimeoutError = o prazo venceu sem ninguém acordar a espera: é o caso normal.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(wake.wait(), timeout)
