"""Partes do estado compartilhado que NÃO precisam de Redis: tradução evento -> atualização,
classificação, montagem do snapshot e utilitários de ID. O comportamento do script Lua (a trava de
ordem) é testado contra o Redis real em tests/integration/test_state_worker.py."""

import pytest

from replay.events import (
    LapCompletedEvent,
    MalformedEvent,
    PositionEvent,
    RaceStatusEvent,
    id_before,
    id_ms,
    id_tuple,
    parse_entry,
)
from replay.state import build_snapshot, classify, to_update


def test_a_position_updates_x_y_under_its_own_version():
    update = to_update(PositionEvent("VER", 10, -20, 5_000))

    assert (update.prefix, update.version_field, update.version) == ("VER:", "pos_t", 5_000)
    assert update.fields == (("x", "10"), ("y", "-20"))


def test_a_lap_has_a_version_separate_from_the_position():
    """Posição e volta têm versões independentes: uma volta "nova" não pode descartar uma posição
    legítima que chegou depois dela na ordem de processamento."""
    lap = to_update(LapCompletedEvent("VER", 3, 9_000, 87_134, "SOFT", 2))
    pos = to_update(PositionEvent("VER", 1, 1, 8_000))

    assert lap.version_field == "lap_t" and pos.version_field == "pos_t"
    assert dict(lap.fields) == {
        "lap": "3",
        "lap_time_ms": "87134",
        "compound": "SOFT",
        "position": "2",
    }


def test_missing_lap_values_become_empty_strings_that_remove_the_old_field():
    update = to_update(LapCompletedEvent("ALB", 1, 9_000))

    # "" = o script REMOVE o campo (um tempo de volta nulo não pode herdar o da volta anterior).
    assert dict(update.fields) == {"lap": "1", "lap_time_ms": "", "compound": "", "position": ""}


def test_status_versions_order_started_running_finished_even_in_the_same_millisecond():
    started = to_update(RaceStatusEvent("started", 1, 1_000, 71)).version
    running = to_update(RaceStatusEvent("running", 2, 1_000, 71)).version
    finished = to_update(RaceStatusEvent("finished", 71, 1_000, 71)).version

    assert started < running < finished  # o `finished` do último ms não pode ser descartado
    assert to_update(RaceStatusEvent("running", 2, 1_001, 71)).version > finished


def test_classification_orders_by_laps_then_by_who_crossed_the_line_first():
    cars = {
        "VER": {"lap": 5, "lap_t": 900},
        "NOR": {"lap": 5, "lap_t": 800},  # mesma volta, cruzou antes
        "HAM": {"lap": 6, "lap_t": 1_000},  # mais voltas: lidera mesmo cruzando depois
        "ALB": {"x": 1},  # ainda sem volta completa
        "BOT": {"x": 2},
    }

    assert classify(cars) == ["HAM", "NOR", "VER", "ALB", "BOT"]


def test_snapshot_parses_types_and_hides_internal_fields():
    raw = {
        "epoch": "3",
        "VER:x": "10",
        "VER:y": "-5",
        "VER:pos_t": "7000",
        "VER:lap": "2",
        "VER:lap_t": "6000",
        "VER:compound": "SOFT",
        "race:state": "running",
        "race:lap": "3",
        "race:total_laps": "71",
        "race:status_v": "123",
    }

    snapshot = build_snapshot(3, raw)

    assert snapshot.epoch == 3
    assert snapshot.race == {"state": "running", "lap": 3, "total_laps": 71}  # sem status_v
    assert snapshot.cars["VER"] == {
        "x": 10,
        "y": -5,
        "pos_t": 7000,
        "lap": 2,
        "lap_t": 6000,
        "compound": "SOFT",
    }
    assert snapshot.classification == ["VER"]
    assert snapshot.to_dict()["epoch"] == 3


@pytest.mark.parametrize(
    ("entry_id", "expected"),
    [
        ("1730649000123-5", "1730649000123-4"),
        ("1730649000123-0", "1730649000122-18446744073709551615"),
        ("0-3", "0-2"),
        ("0-0", "0-0"),  # não há anterior ao primeiro
    ],
)
def test_id_before_is_the_immediate_predecessor(entry_id, expected):
    assert id_before(entry_id) == expected
    assert id_tuple(id_before(entry_id)) <= id_tuple(entry_id)


def test_ids_compare_as_numbers_not_as_strings():
    assert id_tuple("9-0") < id_tuple("10-0")  # como string, "9-0" > "10-0"
    assert id_tuple("5-2") < id_tuple("5-10")
    assert id_ms("1730649000123-7") == 1_730_649_000_123


def test_an_entry_without_epoch_is_malformed():
    fields = PositionEvent("VER", 1, 2, 3).to_fields()  # sem "epoch"

    with pytest.raises(MalformedEvent, match="epoch"):
        parse_entry("1-0", fields)

    entry = parse_entry("1-0", fields | {"epoch": "4"})
    assert (entry.id, entry.epoch, entry.event) == ("1-0", 4, PositionEvent("VER", 1, 2, 3))
