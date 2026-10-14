"""A sessão WebSocket (pump): entrega, cliente lento, cliente que cai e ausência de vazamento.
Um WebSocket falso permite TRAVAR o envio de propósito."""

import asyncio

import pytest

from app.hub import Subscription
from app.routers.ws import SlowClient, pump, snapshot_json
from replay.state import RaceSnapshot


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.stalled = False  # True: send_text nunca completa (TCP sem janela)
        self._incoming: asyncio.Queue[dict] = asyncio.Queue()

    async def send_text(self, data: str) -> None:
        if self.stalled:
            await asyncio.Event().wait()
        self.sent.append(data)

    async def receive(self) -> dict:
        return await self._incoming.get()

    def disconnect(self) -> None:
        self._incoming.put_nowait({"type": "websocket.disconnect", "code": 1001})


def no_leftover_tasks() -> bool:
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []


async def test_pump_delivers_queued_events_in_order_until_the_client_disconnects():
    ws, sub = FakeWebSocket(), Subscription(100, "0-0")
    for n in range(5):
        sub.offer(f"evento-{n}")
    task = asyncio.create_task(pump(ws, sub))

    for _ in range(20):
        await asyncio.sleep(0)
    ws.disconnect()
    await asyncio.wait_for(task, 1)

    assert ws.sent == [f"evento-{n}" for n in range(5)]
    assert no_leftover_tasks()  # nem o emissor nem o vigia ficaram rodando


async def test_a_client_that_disconnects_while_idle_ends_the_session_promptly():
    """Replay parado: nenhum `send` vai falhar para denunciar o cliente. O vigia é que percebe."""
    ws, sub = FakeWebSocket(), Subscription(100, "0-0")
    task = asyncio.create_task(pump(ws, sub))
    await asyncio.sleep(0)

    ws.disconnect()
    await asyncio.wait_for(task, 1)  # não fica esperando a fila para sempre

    assert no_leftover_tasks()


async def test_pump_raises_slow_client_when_the_queue_overflowed():
    ws, sub = FakeWebSocket(), Subscription(2, "0-0")
    sub.offer("a")
    sub.offer("b")
    assert sub.offer("c") is False  # fila cheia
    sub.close(overflowed=True)

    with pytest.raises(SlowClient, match="fila"):
        await asyncio.wait_for(pump(ws, sub), 1)

    assert no_leftover_tasks()


async def test_pump_raises_slow_client_when_a_send_stalls():
    ws, sub = FakeWebSocket(), Subscription(100, "0-0")
    ws.stalled = True
    sub.offer("evento")

    with pytest.raises(SlowClient, match="envio"):
        await asyncio.wait_for(pump(ws, sub, send_timeout_s=0.05), 2)  # o timeout é de verdade

    assert no_leftover_tasks()


async def test_a_normal_close_of_the_queue_just_ends_the_session():
    ws, sub = FakeWebSocket(), Subscription(100, "0-0")
    sub.offer("ultimo")
    sub.close()  # o servidor está desligando (não é cliente lento)

    await asyncio.wait_for(pump(ws, sub), 1)

    assert no_leftover_tasks()


async def test_many_connect_disconnect_cycles_leave_nothing_behind():
    for _ in range(200):
        ws, sub = FakeWebSocket(), Subscription(10, "0-0")
        sub.offer("x")
        task = asyncio.create_task(pump(ws, sub))
        await asyncio.sleep(0)
        ws.disconnect()
        await asyncio.wait_for(task, 1)

    assert no_leftover_tasks()  # nenhuma tarefa acumulou em 200 ciclos


def test_the_snapshot_frame_has_a_stable_shape():
    snapshot = RaceSnapshot(3, {"state": "running", "lap": 2}, {"VER": {"x": 1}}, ["VER"])

    import json

    message = json.loads(snapshot_json(snapshot, {"state": "running"}, "10-0"))

    assert message == {
        "type": "snapshot",
        "as_of": "10-0",
        "epoch": 3,
        "race": {"state": "running", "lap": 2},
        "cars": {"VER": {"x": 1}},
        "classification": ["VER"],
        "replay": {"state": "running"},
    }
