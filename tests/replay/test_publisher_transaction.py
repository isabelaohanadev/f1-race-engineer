"""O publisher escreve o stream principal e o de voltas na MESMA transação (MULTI/EXEC): uma volta
nunca fica num stream e não no outro. Um espião no pipeline prova a estrutura; a atomicidade em si é
garantia do Redis (testada nos testes de integração)."""

import pytest

from replay.events import LapCompletedEvent, PositionEvent, RaceStatusEvent
from replay.publisher import StreamPublisher

MAIN, LAPS = "race:2024:21", "laps:2024:21"


class SpyPipeline:
    def __init__(self, log: list) -> None:
        self.log = log
        self.id = len([x for x in log if x[0] == "open"])

    async def __aenter__(self) -> "SpyPipeline":
        self.log.append(("open", self.id))
        return self

    async def __aexit__(self, *exc) -> None:
        self.log.append(("close", self.id))

    def xadd(self, stream, fields, **kwargs) -> None:
        self.log.append(("xadd", self.id, stream, fields["type"]))

    async def execute(self) -> list:
        self.log.append(("execute", self.id))
        return []


class SpyRedis:
    def __init__(self) -> None:
        self.log: list = []
        self.transactions: list[bool] = []

    def pipeline(self, transaction: bool = True) -> SpyPipeline:
        self.transactions.append(transaction)
        return SpyPipeline(self.log)


EVENTS = [
    RaceStatusEvent("started", 1, 1_000, 3),
    PositionEvent("VER", 1, 2, 1_001),
    LapCompletedEvent("VER", 1, 1_500, 90_000, "SOFT", 1, session_time_ms=90_000),
    RaceStatusEvent("running", 2, 1_500, 3),
    RaceStatusEvent("finished", 3, 9_000, 3),
]


async def test_both_streams_are_written_in_one_transaction_executed_once():
    redis = SpyRedis()

    await StreamPublisher(redis, MAIN, 1, laps_stream=LAPS).publish_many(EVENTS)  # type: ignore[arg-type]

    assert redis.transactions == [True]  # um pipeline, e em transação (MULTI/EXEC)
    pipelines = {entry[1] for entry in redis.log}
    assert pipelines == {0}
    assert [e[0] for e in redis.log if e[0] in ("open", "execute", "close")] == [
        "open", "execute", "close",
    ]  # fmt: skip


async def test_the_main_stream_gets_every_event_and_the_laps_stream_only_laps_and_the_ends():
    redis = SpyRedis()

    await StreamPublisher(redis, MAIN, 1, laps_stream=LAPS).publish_many(EVENTS)  # type: ignore[arg-type]

    main = [e[3] for e in redis.log if e[0] == "xadd" and e[2] == MAIN]
    laps = [e[3] for e in redis.log if e[0] == "xadd" and e[2] == LAPS]
    assert main == [e.type for e in EVENTS]
    assert laps == ["race_status", "lap_completed", "race_status"]  # started, a volta, finished


async def test_an_empty_batch_opens_no_transaction():
    redis = SpyRedis()

    await StreamPublisher(redis, MAIN, 1, laps_stream=LAPS).publish_many([])  # type: ignore[arg-type]

    assert redis.log == [] and redis.transactions == []


@pytest.mark.parametrize("laps_stream", [None])
async def test_without_a_laps_stream_the_transaction_has_only_the_main_stream(laps_stream):
    redis = SpyRedis()

    await StreamPublisher(redis, MAIN, 1, laps_stream=laps_stream).publish_many(EVENTS)  # type: ignore[arg-type]

    assert {e[2] for e in redis.log if e[0] == "xadd"} == {MAIN}
