from datetime import UTC, datetime, timedelta, timezone

import pytest

from replay.events import (
    LAPS_STREAM_VERSION,
    STATE_RUNNING,
    LapCompletedEvent,
    MalformedEvent,
    PositionEvent,
    RaceStatusEvent,
    StreamEntry,
    entry_to_json,
    laps_key,
    laps_stream_of,
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


# ---------------------------------------------------------------- o stream de voltas

FULL_LAP = LapCompletedEvent(
    "NOR",
    7,
    1_730_649_100_000,
    87_134,
    "SOFT",
    3,
    session_time_ms=5_432_100,
    tyre_life=6,
    stint=2,
    pit_in=False,
    pit_out=True,
    track_status="124",
)


def test_the_laps_stream_has_its_own_key_derived_from_the_race_stream():
    assert laps_key(2024, 21) == "laps:2024:21"
    assert laps_stream_of(stream_key(2024, 21)) == laps_key(2024, 21)
    with pytest.raises(ValueError):
        laps_stream_of("laps:2024:21")


def test_a_lap_survives_the_trip_through_the_laps_stream_with_every_strategy_field():
    fields = FULL_LAP.to_lap_fields()

    assert all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items())
    assert fields["v"] == str(LAPS_STREAM_VERSION)
    assert fields["pit_in"] == "0" and fields["pit_out"] == "1"  # falso não some: é um dado
    assert parse_event(fields) == FULL_LAP


def test_the_main_stream_format_of_a_lap_does_not_change():
    """O stream principal e o worker de estado seguem lendo o que sempre leram: os campos da
    estratégia só existem no stream de voltas."""
    fields = FULL_LAP.to_fields()

    assert set(fields) == {"type", "driver", "lap", "t", "lap_time_ms", "compound", "position"}
    assert parse_event(fields) == LapCompletedEvent("NOR", 7, 1_730_649_100_000, 87_134, "SOFT", 3)


def test_unknown_strategy_fields_are_absent_not_empty_for_old_laps():
    lap = LapCompletedEvent("ALB", 1, 5, session_time_ms=1_000)  # corrida ingerida sem o resto

    fields = lap.to_lap_fields()

    assert "tyre_life" not in fields and "pit_in" not in fields and "track_status" not in fields
    assert parse_event(fields) == lap


def test_a_flag_that_is_not_zero_or_one_is_malformed():
    fields = FULL_LAP.to_lap_fields() | {"pit_in": "talvez"}

    with pytest.raises(MalformedEvent):
        parse_event(fields)


def test_the_websocket_message_of_a_lap_does_not_carry_the_strategy_fields():
    """O contrato do WebSocket não muda por causa do stream de voltas."""
    entry = StreamEntry("1-0", 3, FULL_LAP)

    import json

    message = json.loads(entry_to_json(entry))

    assert set(message) == {
        "type", "id", "epoch", "driver", "lap", "t", "lap_time_ms", "compound", "position"
    }  # fmt: skip
