from datetime import UTC, datetime, timedelta, timezone

import pytest

from replay.events import (
    STATE_RUNNING,
    LapCompletedEvent,
    MalformedEvent,
    PositionEvent,
    RaceStatusEvent,
    parse_event,
    stream_key,
    to_epoch_ms,
)


def test_stream_key_has_the_documented_format():
    assert stream_key(2024, 21) == "race:2024:21"


@pytest.mark.parametrize(
    "event",
    [
        PositionEvent(driver="VER", x=-120, y=3400, t=1_730_649_000_123),
        LapCompletedEvent("NOR", 5, 1_730_649_100_000, 87134, "SOFT", 3),
        LapCompletedEvent("ALB", 1, 1_730_649_100_000),  # tudo que é opcional, ausente
        RaceStatusEvent(STATE_RUNNING, 2, 1_730_649_200_000, 71),
        RaceStatusEvent("finished", 71, 1_730_654_000_000),  # sem total_laps
    ],
)
def test_every_event_survives_the_trip_through_redis_fields(event):
    fields = event.to_fields()

    assert all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items())  # só strings
    assert parse_event(fields) == event


def test_missing_values_are_omitted_not_written_as_empty_strings():
    fields = LapCompletedEvent("ALB", 1, 5).to_fields()

    assert fields == {"type": "lap_completed", "driver": "ALB", "lap": "1", "t": "5"}


def test_unknown_fields_are_ignored_so_the_contract_can_grow():
    fields = PositionEvent("VER", 1, 2, 3).to_fields() | {"campo_novo": "x"}

    assert parse_event(fields) == PositionEvent("VER", 1, 2, 3)


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "pit_stop", "driver": "VER"},  # tipo desconhecido
        {"driver": "VER"},  # sem tipo
        {"type": "position", "driver": "VER", "x": "1", "y": "2"},  # falta t
        {"type": "position", "driver": "VER", "x": "um", "y": "2", "t": "3"},  # x não é inteiro
    ],
)
def test_malformed_entries_raise_a_specific_error(fields):
    with pytest.raises(MalformedEvent):
        parse_event(fields)


def test_epoch_ms_is_exact_integer_arithmetic():
    moment = datetime(2024, 11, 3, 15, 50, 0, 123_999, tzinfo=UTC)

    assert to_epoch_ms(moment) == 1_730_649_000_123  # trunca os µs, sem erro de float
    assert isinstance(to_epoch_ms(moment), int)


def test_naive_datetimes_are_treated_as_utc_and_offsets_are_respected():
    naive = datetime(2024, 11, 3, 15, 50)
    brasilia = datetime(2024, 11, 3, 12, 50, tzinfo=timezone(timedelta(hours=-3)))

    assert to_epoch_ms(naive) == to_epoch_ms(naive.replace(tzinfo=UTC))
    assert to_epoch_ms(brasilia) == to_epoch_ms(naive)  # mesmo instante, fusos diferentes
