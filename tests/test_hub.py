"""O hub: distribuição para vários clientes, isolamento do cliente lento e ciclo de vida.

Sem Redis: as entradas são injetadas em `dispatch()` (o que o leitor do stream chamaria) e o
Redis é um dublê mínimo onde o registro precisa dele.
"""

import asyncio

import pytest

from app.hub import HubRegistry, RaceHub
from replay.events import PositionEvent, stream_key


def entry(n: int, driver: str = "VER") -> tuple[str, dict[str, str]]:
    fields = PositionEvent(driver, n, -n, n).to_fields() | {"epoch": "1"}
    return f"{n}-0", fields


def drain(queue: asyncio.Queue) -> list:
    """Esvazia a fila e devolve só as MENSAGENS (o `payload` de cada Delivery)."""
    items = []
    while not queue.empty():
        item = queue.get_nowait()
        items.append(item.payload if item is not None else None)
    return items


@pytest.fixture
def hub() -> RaceHub:
    hub = RaceHub(None, 2024, 21, queue_size=50)
    hub.mark_ready("0-0")
    return hub


async def test_every_client_receives_every_event_in_order(hub: RaceHub):
    subs = [await hub.subscribe() for _ in range(3)]

    hub.dispatch([entry(n) for n in range(1, 11)])

    received = [drain(sub.queue) for sub in subs]
    assert all(len(r) == 10 for r in received) and received[0] == received[1] == received[2]
    assert [__import__("json").loads(p)["id"] for p in received[0]] == [
        f"{n}-0" for n in range(1, 11)
    ]


async def test_each_event_is_serialized_once_and_shared_by_all_clients(hub: RaceHub):
    one, two = await hub.subscribe(), await hub.subscribe()

    hub.dispatch([entry(1)])

    # A MESMA string nas duas filas: serializar uma vez e entregar N vezes (não N serializações).
    assert one.queue.get_nowait().payload is two.queue.get_nowait().payload


async def test_events_carry_real_types_the_stream_id_and_the_epoch(hub: RaceHub):
    sub = await hub.subscribe()

    hub.dispatch([entry(7)])

    import json

    assert json.loads(sub.queue.get_nowait().payload) == {
        "type": "position",
        "id": "7-0",
        "epoch": 1,
        "driver": "VER",
        "x": 7,
        "y": -7,
        "t": 7,
    }


async def test_each_delivery_carries_the_publication_instant_from_the_stream_id(hub: RaceHub):
    """O ID do stream (`ms-seq`) é o instante em que o Redis registrou a entrada: é dele que sai a
    latência publicação->entrega, sem gravar relógio nenhum no evento."""
    sub = await hub.subscribe()

    hub.dispatch([("1730649000123-4", PositionEvent("VER", 1, 2, 3).to_fields() | {"epoch": "1"})])

    assert sub.queue.get_nowait().published_ms == 1_730_649_000_123


async def test_a_client_only_receives_events_after_the_point_it_joined(hub: RaceHub):
    hub.dispatch([entry(1), entry(2)])  # antes de o cliente chegar
    sub = await hub.subscribe()
    assert sub.join_id == "2-0"  # é daqui que o stream precisa completar (catch-up)

    hub.dispatch([entry(3)])

    assert [__import__("json").loads(p)["id"] for p in drain(sub.queue)] == ["3-0"]


async def test_a_slow_client_is_dropped_and_never_blocks_or_slows_the_others(hub: RaceHub):
    slow, fast = await hub.subscribe(), await hub.subscribe()  # a fila de cada um comporta 50
    delivered_to_fast: list[str] = []

    for n in range(1, 201):  # 200 eventos: o `slow` nunca consome
        hub.dispatch([entry(n)])
        delivered_to_fast += drain(fast.queue)  # o rápido consome na hora

    assert len(delivered_to_fast) == 200  # o rápido recebeu TUDO, no ritmo do stream
    assert slow.overflowed and hub.stats.dropped_slow == 1
    assert hub.clients == 1  # só o rápido continua inscrito


async def test_dispatch_never_awaits_so_a_client_cannot_stall_the_hub(hub: RaceHub):
    assert not asyncio.iscoroutinefunction(hub.dispatch)  # síncrono: não há onde esperar cliente


async def test_the_memory_per_client_is_bounded_no_matter_how_many_events_arrive(hub: RaceHub):
    stalled = await hub.subscribe()

    for n in range(1, 20_001):
        hub.dispatch([entry(n)])

    # Encheu em 50, foi desconectado e a fila foi esvaziada (só sobra o marcador de fim).
    assert stalled.queue.qsize() <= 1 and stalled.overflowed


async def test_malformed_entries_are_skipped_but_the_hub_keeps_advancing(hub: RaceHub):
    sub = await hub.subscribe()

    hub.dispatch([("1-0", {"type": "position"}), entry(2)])  # a 1ª é ilegível (sem epoch etc.)

    assert len(drain(sub.queue)) == 1 and hub.last_id == "2-0"


# ---------------------------------------------------------------- o registro


class QuietRedis:
    """Um Redis que não tem nada no stream e cujo XREAD bloqueia para sempre (stream parado)."""

    async def xrevrange(self, *args, **kwargs):
        return []

    async def xread(self, *args, **kwargs):
        await asyncio.Event().wait()


async def test_concurrent_clients_of_the_same_race_share_one_hub():
    registry = HubRegistry(QuietRedis())  # type: ignore[arg-type]

    pairs = await asyncio.gather(*(registry.subscribe(2024, 21) for _ in range(25)))

    assert len({id(hub) for hub, _ in pairs}) == 1  # UM leitor para os 25 clientes
    assert registry.clients == 25 and len(registry.hubs) == 1
    await registry.close()


async def test_the_hub_is_released_when_its_last_client_leaves():
    registry = HubRegistry(QuietRedis())  # type: ignore[arg-type]
    pairs = [await registry.subscribe(2024, 21) for _ in range(3)]
    hub = pairs[0][0]

    for _, sub in pairs:
        hub.unsubscribe(sub)

    def hub_tasks() -> list[asyncio.Task]:
        return [t for t in asyncio.all_tasks() if t.get_name().startswith("hub-")]

    for _ in range(20):  # o cancelamento do leitor leva alguns turnos do loop para completar
        if not hub_tasks():
            break
        await asyncio.sleep(0)

    assert registry.hubs == [] and hub.closed  # o leitor (e a conexão Redis dele) foram liberados
    assert hub_tasks() == []


async def test_different_races_get_different_hubs():
    registry = HubRegistry(QuietRedis())  # type: ignore[arg-type]

    (a, _), (b, _) = await registry.subscribe(2024, 21), await registry.subscribe(2023, 1)

    assert a is not b and a.stream == stream_key(2024, 21)
    await registry.close()


async def test_a_new_client_after_the_hub_was_released_gets_a_fresh_one():
    registry = HubRegistry(QuietRedis())  # type: ignore[arg-type]
    first, sub = await registry.subscribe(2024, 21)
    first.unsubscribe(sub)
    await asyncio.sleep(0)

    second, _ = await registry.subscribe(2024, 21)

    assert second is not first and not second.closed
    await registry.close()


# ---------------------------------------------------------------- o evento `strategy`


def result_entry(lap: int, epoch: int = 1, entry_id: str | None = None):
    import json

    document = {"v": 1, "type": "strategy", "epoch": epoch, "lap": lap}
    fields = {"v": "1", "epoch": str(epoch), "lap": str(lap), "data": json.dumps(document)}
    return entry_id or f"{lap}-0", fields


async def test_a_strategy_result_reaches_every_client_as_the_document_plus_the_stream_id(
    hub: RaceHub,
):
    import json

    one, two = await hub.subscribe(), await hub.subscribe()

    hub.dispatch_strategy([result_entry(12, epoch=3, entry_id="900-1")])

    for sub in (one, two):
        message = json.loads(sub.queue.get_nowait().payload)
        assert message == {"v": 1, "type": "strategy", "epoch": 3, "lap": 12, "id": "900-1"}
    assert hub.stats.dispatched == 1


async def test_strategy_results_share_the_clients_queue_with_the_race_events_in_arrival_order(
    hub: RaceHub,
):
    import json

    sub = await hub.subscribe()

    hub.dispatch([entry(1)])
    hub.dispatch_strategy([result_entry(5)])
    hub.dispatch([entry(2)])

    assert [json.loads(p)["type"] for p in drain(sub.queue)] == ["position", "strategy", "position"]


async def test_a_strategy_result_is_serialized_once_and_the_slow_client_policy_is_the_same(
    hub: RaceHub,
):
    one, two = await hub.subscribe(), await hub.subscribe()
    hub.dispatch_strategy([result_entry(5)])
    assert one.queue.get_nowait().payload is two.queue.get_nowait().payload

    for lap in range(6, 80):  # 50 de fila: o cliente que não consome é desconectado, como sempre
        hub.dispatch_strategy([result_entry(lap)])
        drain(one.queue)

    assert two.overflowed and hub.stats.dropped_slow == 1 and hub.clients == 1


async def test_an_unreadable_strategy_entry_is_skipped_and_the_cursor_still_advances(hub: RaceHub):
    sub = await hub.subscribe()

    hub.dispatch_strategy(
        [("1-0", {"data": "isto não é json"}), ("2-0", {"v": "1"}), result_entry(7)]
    )

    assert len(drain(sub.queue)) == 1  # só o legível
    assert hub._last_strategy_id == "7-0"  # e o hub não ficou preso nas ilegíveis
