"""Publica eventos no Redis Stream da corrida.

POR QUE REDIS STREAMS E NÃO PUB/SUB (a pergunta clássica de entrevista):
* Pub/Sub é "dispara e esquece": se ninguém está escutando NO INSTANTE da publicação, a mensagem
  some, e um consumidor que caiu por 2 s perde 2 s de corrida para sempre. O Stream é um LOG
  persistente: cada entrada tem um ID crescente, e quem volta lê a partir do último ID que viu.
* Pub/Sub entrega a mensagem a TODOS os inscritos e não sabe nada sobre eles. O Stream tem consumer
  groups: o Redis lembra até onde cada grupo leu, quais entradas foram entregues e ainda não
  confirmadas (XACK), e permite repartir o trabalho entre workers (e medir o atraso: o "lag").
* Contrapartida: o Stream ocupa memória (por isso o MAXLEN abaixo) e dá mais trabalho de operar.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

from redis.asyncio import Redis

from replay.events import (
    STATE_FINISHED,
    STATE_STARTED,
    LapCompletedEvent,
    RaceStatusEvent,
    ReplayEvent,
)

logger = logging.getLogger(__name__)

# Tamanho máximo APROXIMADO do stream, em entradas. O stream é um BUFFER DE CATCH-UP, não o
# histórico (o histórico é o banco; o replay pode ser refeito a qualquer hora). O valor responde:
# "por quanto tempo um consumidor pode ficar parado/lento e ainda alcançar o stream sem perder
# eventos?" Quem fica além do limite perde as entradas aparadas (o Redis não espera consumidor).
#
#   * ritmo de eventos: ~110/s a 1x (600 mil amostras em ~1h30 de corrida) e ~1.760/s a 16x;
#   * 100.000 entradas = ~15 min de folga a 1x e ~1 min a 16x. O pior caso (16x) é o que dimensiona:
#     1 min cobre com sobra o restart de um worker e uma pausa de GC/rede;
#   * memória MEDIDA (Redis 8.6, eventos realistas): 100.000 entradas = 4,2 MB (~42 bytes cada, já
#     que o formato de campos planos reaproveita os nomes dos campos). Ou seja: o limite NÃO é
#     ditado pela memória (o --maxmemory do compose é 256 MB), e sim pela janela de catch-up que
#     queremos garantir. Se um dia for preciso mais folga, 1 milhão de entradas custa ~42 MB.
#   * com `~`, o stream passou de 100.000 em apenas 53 entradas no teste (o corte é em blocos).
#
# Quem passa mais que isso atrás precisa ressincronizar de qualquer jeito, pelo snapshot de estado
# (etapas seguintes), em vez de reler eventos antigos.
STREAM_MAXLEN = 100_000

# O stream de VOLTAS (ver `laps_key` em events.py) guarda só as voltas e o início e o fim do replay:
# ~1.400 entradas por corrida. 20.000 cobre com folga qualquer corrida (e o stream é apagado a cada
# replay novo); o corte existe só para o pior caso nunca virar um vazamento de memória.
LAPS_STREAM_MAXLEN = 20_000


class Publisher(Protocol):
    async def reset(self) -> None: ...

    async def publish_many(self, events: Sequence[ReplayEvent]) -> None: ...


def goes_to_laps_stream(event: ReplayEvent) -> bool:
    """O que o stream de voltas leva: as voltas, e o início e o fim do replay (o `started` traz o
    total de voltas e é o que avisa o consumidor de que um replay novo começou). As posições e as
    mudanças de volta do líder (`running`) não: a estratégia as deriva das voltas."""
    if isinstance(event, LapCompletedEvent):
        return True
    return isinstance(event, RaceStatusEvent) and event.state in (STATE_STARTED, STATE_FINISHED)


class StreamPublisher:
    def __init__(
        self,
        redis: Redis,
        stream: str,
        epoch: int,
        maxlen: int = STREAM_MAXLEN,
        *,
        laps_stream: str | None = None,
        laps_maxlen: int = LAPS_STREAM_MAXLEN,
    ) -> None:
        self._redis = redis
        self._stream = stream
        self._laps_stream = laps_stream
        self._epoch = str(epoch)  # grava em TODA entrada: ver StreamEntry em events.py
        self._maxlen = maxlen
        self._laps_maxlen = laps_maxlen

    async def reset(self) -> None:
        """Começa um replay com o stream LIMPO (DEL), para eventos de uma execução anterior não se
        misturarem aos da nova. O DEL também apaga os consumer groups do stream: os workers
        percebem (NOGROUP) e recriam o grupo, e o evento `started` zera o estado deles. O stream
        de voltas é apagado junto, pelo mesmo motivo (e o consumidor de estratégia também recria o
        grupo dele)."""
        await self._redis.delete(
            *([self._stream] + ([self._laps_stream] if self._laps_stream else []))
        )

    async def publish_many(self, events: Sequence[ReplayEvent]) -> None:
        """Publica o lote numa transação (MULTI/EXEC): ou entra tudo, ou nada.

        Em lote porque a 16x são ~1.760 eventos/s: um ida-e-volta por evento seria o gargalo. A
        transação evita publicar METADE do lote se a conexão cair no meio; sem ela, repetir o lote
        duplicaria as entradas que já tinham entrado.

        `MAXLEN ~ N` (approximate): o Redis apara o stream em blocos inteiros, só quando pode
        remover um bloco todo (~100 entradas), então o stream passa de N em até ~100 e o corte é
        barato. O `MAXLEN = N` exato removeria entrada por entrada a cada XADD e custaria bem mais.
        """
        if not events:
            return
        async with self._redis.pipeline(transaction=True) as pipe:
            for event in events:
                fields = event.to_fields() | {"epoch": self._epoch}
                pipe.xadd(self._stream, fields, maxlen=self._maxlen, approximate=True)
                if self._laps_stream is not None and goes_to_laps_stream(event):
                    # Na MESMA transação: a volta nunca está num stream e não no outro.
                    lap_fields = (
                        event.to_lap_fields()
                        if isinstance(event, LapCompletedEvent)
                        else event.to_fields()
                    ) | {"epoch": self._epoch}
                    pipe.xadd(
                        self._laps_stream, lap_fields, maxlen=self._laps_maxlen, approximate=True
                    )
            await pipe.execute()
