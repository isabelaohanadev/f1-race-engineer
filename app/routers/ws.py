"""O WebSocket de uma corrida: snapshot do estado atual e, depois, os eventos em tempo real.

COMO UM CLIENTE ENTRA SEM PERDER NEM CORROMPER NADA (a trava 2)
O cliente precisa do estado de agora E de todos os eventos que acontecerem depois, sem buraco. A
receita é: (a) anotar um ponto do stream, (b) ler o snapshot, (c) seguir o stream a partir do ponto
de (a). Repetições são inofensivas (a atualização é idempotente), buracos não. Os passos, na ordem:

  0. INSCREVER a fila do cliente no hub: ler `join_id` (o último ID que o hub já repassou) e
     registrar a fila acontecem SEM `await` no meio. Daí em diante, todo evento DEPOIS de `join_id`
     cai na fila. É o que fecha a ponta de CIMA da cobertura (o que o catch-up não alcança).
  a. MARCA D'ÁGUA: o ID até o qual tudo já foi APLICADO ao estado (replay/state.py). Aqui está o
     ponto sutil: NÃO é "o último ID do stream". O worker aplica DEPOIS da publicação; se o cliente
     anotasse o fim do stream e lesse os hashes, os eventos já publicados e ainda não aplicados
     (alguns ms, ou muito mais se os workers estiverem atrasados ou parados) não estariam no
     snapshot e também não viriam do XREAD (que só traz o que vem DEPOIS do ID): um buraco.
  b. SNAPSHOT: um HGETALL atômico dos hashes (consistente entre os 20 pilotos).
  c. CATCH-UP: o que está no stream entre a marca d'água e `join_id` (XRANGE), lido do PRÓPRIO
     stream: por isso a ordem entre inscrever e ler o snapshot não importa, só que o catch-up vá
     até `join_id`. É o que o snapshot pode não ter. Pode repetir eventos que o snapshot já
     refletia: inofensivo.
  d. AO VIVO: a fila do hub (tudo depois de `join_id`).
Entre o catch-up e o ao vivo vai o último resultado de estratégia do replay atual, se há.
Cobertura: <= marca d'água está no snapshot; (marca d'água, join_id] vem do catch-up; > join_id vem
da fila. Nenhum evento fica de fora, e os repetidos são idempotentes no cliente.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Path, WebSocket, WebSocketDisconnect
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app import metrics
from app.deps import get_control, get_hubs, get_redis, get_state_store, get_strategy_store
from app.hub import HubRegistry, Subscription
from app.routers.replay import to_status_out
from replay.control import RedisControlPlane
from replay.events import entry_to_json, id_tuple, parse_entry
from replay.latency import seconds_since
from replay.state import RaceSnapshot, RaceStateStore
from strategy.service.results import ResultStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

# Um `send` que não completa em 10 s é um cliente parado (TCP sem janela): fecha em vez de segurar
# a tarefa (e a memória dela) para sempre.
SEND_TIMEOUT_S = 10.0
CATCHUP_PAGE = 500

CLOSE_TRY_AGAIN_LATER = 1013  # o cliente não acompanha: reconecte e receba um snapshot novo
CLOSE_INTERNAL_ERROR = 1011


class SlowClient(Exception):
    """O cliente não consome na velocidade dos eventos (fila cheia ou envio parado)."""


class WebSocketLike(Protocol):
    async def send_text(self, data: str) -> None: ...
    async def receive(self) -> dict: ...


# ---------------------------------------------------------------- mensagens


def snapshot_json(snapshot: RaceSnapshot, replay: dict | None, as_of: str) -> str:
    """O 1º frame: o estado de agora. `as_of` é a marca d'água (informativo)."""
    message = {"type": "snapshot", "as_of": as_of, **snapshot.to_dict(), "replay": replay}
    return json.dumps(message, separators=(",", ":"))


async def catch_up(redis: Redis, stream: str, after_id: str, upto_id: str) -> AsyncIterator[str]:
    """Os eventos do stream em (after_id, upto_id], em páginas (memória constante)."""
    if id_tuple(after_id) >= id_tuple(upto_id):
        return  # nada entre os dois: o snapshot já cobre tudo até o ponto do hub
    cursor = f"({after_id}"  # "(" = exclusivo
    while True:
        page = await redis.xrange(stream, min=cursor, max=upto_id, count=CATCHUP_PAGE)
        for entry_id, fields in page:
            yield entry_to_json(parse_entry(entry_id, fields))
        if len(page) < CATCHUP_PAGE:
            return
        cursor = f"({page[-1][0]}"


# ---------------------------------------------------------------- envio e bombeamento


async def send_with_timeout(ws: WebSocketLike, payload: str, timeout_s: float) -> None:
    try:
        async with asyncio.timeout(timeout_s):
            await ws.send_text(payload)
    except TimeoutError as exc:
        raise SlowClient(f"o envio não completou em {timeout_s} s") from exc


async def _send_loop(ws: WebSocketLike, sub: Subscription, timeout_s: float) -> None:
    while True:
        delivery = await sub.queue.get()
        if delivery is None:  # a fila foi encerrada
            if sub.overflowed:
                raise SlowClient("a fila do cliente encheu")
            return
        await send_with_timeout(ws, delivery.payload, timeout_s)
        if delivery.published_ms is not None:
            metrics.DELIVERY_LATENCY.observe(seconds_since(delivery.published_ms))


async def _watch_disconnect(ws: WebSocketLike) -> None:
    """Fica lendo do cliente SÓ para perceber quando ele cai. Sem isto, um cliente que sumiu num
    replay parado (nenhum `send` falhando) deixaria a tarefa e a fila presas até o servidor
    reiniciar. O que o cliente enviar é ignorado."""
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return


async def pump(ws: WebSocketLike, sub: Subscription, *, send_timeout_s: float = SEND_TIMEOUT_S):
    """Entrega a fila ao cliente até ele desconectar, a fila fechar ou ele ficar para trás.

    Duas tarefas: uma envia, outra só espera o cliente desconectar. A primeira que terminar encerra
    a outra: nenhuma sobra rodando depois que a conexão acaba (sem vazamento)."""
    watcher = asyncio.create_task(_watch_disconnect(ws))
    sender = asyncio.create_task(_send_loop(ws, sub, send_timeout_s))
    try:
        done, _ = await asyncio.wait({watcher, sender}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()  # propaga SlowClient/erros; um retorno normal = fim da conexão
    finally:
        for task in (watcher, sender):
            task.cancel()
        await asyncio.gather(watcher, sender, return_exceptions=True)


# ---------------------------------------------------------------- o endpoint


@router.websocket("/ws/races/{year}/{round}")
async def race_websocket(
    websocket: WebSocket,
    year: Annotated[int, Path(ge=1950, le=2100)],
    round_number: Annotated[int, Path(alias="round", ge=1, le=50)],
    hubs: Annotated[HubRegistry, Depends(get_hubs)],
    store: Annotated[RaceStateStore, Depends(get_state_store)],
    control: Annotated[RedisControlPlane, Depends(get_control)],
    redis: Annotated[Redis, Depends(get_redis)],
    strategy: Annotated[ResultStore, Depends(get_strategy_store)],
) -> None:
    await websocket.accept()
    metrics.WS_CONNECTIONS.inc()
    hub = sub = None
    try:
        hub, sub = await hubs.subscribe(year, round_number)  # passo 0: a fila já está ouvindo
        anchor = await store.applied_watermark(year, round_number)  # a
        snapshot = await store.snapshot(year, round_number)  # b
        info = await control.get_info(year, round_number)
        replay = to_status_out(info).model_dump() if info else None
        await send_with_timeout(websocket, snapshot_json(snapshot, replay, anchor), SEND_TIMEOUT_S)
        async for payload in catch_up(redis, hub.stream, anchor, sub.join_id):  # c
            await send_with_timeout(websocket, payload, SEND_TIMEOUT_S)
        # O último resultado de estratégia do replay atual (se há): lido DEPOIS de inscrever a fila,
        # então um resultado novo publicado a partir daí chega também pela fila (o cliente ignora a
        # repetição: vale o maior `lap` de cada epoch). Não há catch-up do stream de resultados.
        latest = await strategy.get(year, round_number)
        if latest is not None:
            await send_with_timeout(websocket, latest.payload, SEND_TIMEOUT_S)
        await pump(websocket, sub)  # d
    except WebSocketDisconnect:
        pass  # o cliente foi embora: normal
    except SlowClient as exc:
        logger.info("Cliente WebSocket lento desconectado: %s", exc)
        metrics.WS_SLOW_DISCONNECTS.inc()
        with contextlib.suppress(Exception):
            await websocket.close(code=CLOSE_TRY_AGAIN_LATER, reason="cliente lento demais")
    except RedisError:
        logger.exception("Redis indisponível no WebSocket de %s/%s", year, round_number)
        with contextlib.suppress(Exception):
            await websocket.close(code=CLOSE_INTERNAL_ERROR, reason="Redis indisponível")
    except (RuntimeError, OSError):
        pass  # enviar para uma conexão que o cliente já fechou
    finally:
        if hub is not None and sub is not None:
            # SEMPRE: libera a fila e, se for o último cliente, o leitor do hub.
            hub.unsubscribe(sub)
