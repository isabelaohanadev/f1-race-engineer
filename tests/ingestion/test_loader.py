"""Testes do loader com DataFrames sintéticos no formato do FastF1 (sem rede, sem FastF1 real).

A ordem segue o fluxo do loader: helpers -> corrida -> inscrições -> voltas -> trajetórias ->
orquestração (`load_race`) com o módulo `fastf1` inteiro substituído.
"""

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from ingestion import loader
from ingestion.dto import RaceData
from tests.ingestion.factories import (
    NAN,
    make_laps,
    make_positions,
    make_results,
    make_session,
)

T0 = pd.Timestamp("2024-11-03 14:38:02.746")  # t0_date real do GP do Brasil 2024


def td(text: str) -> pd.Timedelta:
    return pd.Timedelta(text)


# ------------------------------------------------------------------ helpers de conversão


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3.0, 3), (0.0, 0), (NAN, None), (None, None), (pd.NA, None)],
)
def test_int_or_none(value, expected):
    assert loader._int_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("SOFT", "SOFT"), ("", None), (None, None), (NAN, None), (pd.NA, None)],
)
def test_str_or_none(value, expected):
    assert loader._str_or_none(value) == expected


def test_to_utc_marks_the_naive_fastf1_timestamp_as_utc_without_shifting_it():
    result = loader._to_utc(pd.Timestamp("2024-11-03 15:30:00"))

    assert result == datetime(2024, 11, 3, 15, 30, tzinfo=UTC)
    assert result.utcoffset() == timedelta(0)  # com fuso: vira timestamptz sem ambiguidade


# ------------------------------------------------------------------ corrida


def test_build_race_maps_event_fields_and_uses_scheduled_utc_date():
    session = make_session(results=make_results(), laps=make_laps(), pos_data={})

    race = loader._build_race(session, 2024, 21)

    assert (race.year, race.round, race.name) == (2024, 21, "São Paulo Grand Prix")
    assert (race.location, race.country, race.total_laps) == ("São Paulo", "Brazil", 69)
    assert race.starts_at == datetime(2024, 11, 3, 15, 30, tzinfo=UTC)


def test_build_race_falls_back_to_the_highest_lap_number_when_total_laps_is_missing():
    laps = make_laps({"Driver": "VER", "LapNumber": 1.0}, {"Driver": "VER", "LapNumber": 57.0})
    session = make_session(results=make_results(), laps=laps, pos_data={}, total_laps=None)

    assert loader._build_race(session, 2023, 1).total_laps == 57


def test_build_race_total_laps_is_none_when_nothing_is_known():
    session = make_session(results=make_results(), laps=make_laps(), pos_data={}, total_laps=None)

    assert loader._build_race(session, 2024, 1).total_laps is None


def test_build_race_turns_missing_event_text_into_none():
    event = {"EventName": "Test GP", "Location": NAN, "Country": ""}
    session = make_session(results=make_results(), laps=make_laps(), pos_data={}, event=event)

    race = loader._build_race(session, 2024, 1)

    assert (race.location, race.country) == (None, None)


# ------------------------------------------------------------------ inscrições (resultados)


def result_row(**overrides):
    row = {
        "DriverNumber": "1",
        "Abbreviation": "VER",
        "FullName": "Max Verstappen",
        "TeamName": "Red Bull Racing",
        "TeamColor": "3671C6",
        "GridPosition": 17.0,
        "Position": 1.0,
        "Status": "Finished",
    }
    return row | overrides


def test_build_entries_converts_types_from_the_real_fastf1_shapes():
    results = make_results(result_row(), result_row(DriverNumber="44", Abbreviation="HAM"))

    entries = loader._build_entries(results)

    assert [(e.driver_code, e.driver_number) for e in entries] == [("VER", 1), ("HAM", 44)]
    first = entries[0]
    assert (first.full_name, first.team_name) == ("Max Verstappen", "Red Bull Racing")
    assert (first.grid_position, first.finish_position, first.status) == (17, 1, "Finished")


def test_build_entries_uppercases_the_mixed_case_team_colors_of_the_api():
    # A API real devolve "0093cc" (Alpine) junto de "3671C6" (Red Bull).
    entries = loader._build_entries(make_results(result_row(TeamColor="0093cc")))

    assert entries[0].team_color == "0093CC"


@pytest.mark.parametrize("bad_color", [NAN, None, "", "FFF", "#3671C6"])
def test_build_entries_discards_missing_or_malformed_team_colors(bad_color):
    entries = loader._build_entries(make_results(result_row(TeamColor=bad_color)))

    assert entries[0].team_color is None


def test_build_entries_keeps_pit_lane_start_as_zero_and_turns_nan_into_none():
    results = make_results(
        result_row(Abbreviation="PIT", DriverNumber="2", GridPosition=0.0),  # largou dos boxes
        result_row(
            Abbreviation="DNS", DriverNumber="3", GridPosition=NAN, Position=NAN, Status=NAN
        ),
    )

    pit, dns = loader._build_entries(results)

    assert pit.grid_position == 0  # 0 NÃO é "sem dado": é a convenção de largada pelos boxes
    assert (dns.grid_position, dns.finish_position, dns.status) == (None, None, None)


# ------------------------------------------------------------------ voltas


def lap_row(**overrides):
    row = {
        "Driver": "VER",
        "LapNumber": 1.0,
        "LapTime": td("0 days 00:01:39.161"),
        "Time": td("0 days 01:13:34.207"),
        "Compound": "INTERMEDIATE",
        "TyreLife": 1.0,
        "Stint": 1.0,
        "Position": 11.0,
    }
    return row | overrides


def test_build_laps_converts_values_like_the_real_brazil_2024_first_lap():
    laps = loader._build_laps(make_laps(lap_row()), T0)

    lap = laps[0]
    assert (lap.driver_code, lap.lap_number) == ("VER", 1)
    assert lap.lap_time_ms == 99161
    assert (lap.compound, lap.tyre_life, lap.stint, lap.position) == ("INTERMEDIATE", 1, 1, 11)
    # fim da volta = t0_date + Time (14:38:02.746 + 1:13:34.207), e em UTC.
    assert lap.completed_at == datetime(2024, 11, 3, 15, 51, 36, 953000, tzinfo=UTC)


@pytest.mark.parametrize("milliseconds", [64002, 64005, 64008])
def test_build_laps_lap_time_survives_float_representation_error(milliseconds):
    """Em ~1% dos milissegundos, total_seconds()*1000 cai ABAIXO do inteiro: 1:04.002 vira
    64001.99999999999 e um int() ingênuo gravaria 64001 (1 ms a menos). round() corrige."""
    lap_time = pd.Timedelta(milliseconds=milliseconds)
    # Premissa do teste: estes valores REALMENTE exibem o erro (senão o teste seria decorativo).
    assert int(lap_time.total_seconds() * 1000) != milliseconds

    lap = loader._build_laps(make_laps(lap_row(LapTime=lap_time)), T0)[0]

    assert lap.lap_time_ms == milliseconds


def test_build_laps_lap_without_time_nat_becomes_none_but_keeps_the_other_fields():
    """Volta de largada/box/bandeira vermelha: LapTime é NaT (37 de 1.134 no GP do Brasil)."""
    lap = loader._build_laps(make_laps(lap_row(LapTime=pd.NaT)), T0)[0]

    assert lap.lap_time_ms is None
    assert lap.completed_at is not None  # o instante de fim existe mesmo sem tempo de volta
    assert (lap.compound, lap.stint, lap.tyre_life) == ("INTERMEDIATE", 1, 1)


def test_build_laps_lap_without_session_time_nat_has_no_completion_instant():
    lap = loader._build_laps(make_laps(lap_row(Time=pd.NaT)), T0)[0]

    assert lap.completed_at is None
    assert lap.lap_time_ms == 99161


@pytest.mark.parametrize("missing", [None, NAN, ""])
def test_build_laps_driver_without_tyre_compound_becomes_none(missing):
    lap = loader._build_laps(make_laps(lap_row(Compound=missing)), T0)[0]

    assert lap.compound is None


def test_build_laps_keeps_unknown_compound_as_the_literal_the_api_sends():
    lap = loader._build_laps(make_laps(lap_row(Compound="UNKNOWN")), T0)[0]

    assert lap.compound == "UNKNOWN"  # a API usa essa string para "não sei"; não é nulo


def test_build_laps_null_numeric_fields_become_none():
    lap = loader._build_laps(make_laps(lap_row(Position=NAN, Stint=NAN, TyreLife=NAN)), T0)[0]

    assert (lap.position, lap.stint, lap.tyre_life) == (None, None, None)
    assert lap.lap_number == 1  # a chave nunca é nula


def test_build_laps_row_full_of_nulls_still_produces_a_valid_record():
    row = lap_row(LapTime=pd.NaT, Time=pd.NaT, Compound=None, TyreLife=NAN, Stint=NAN, Position=NAN)

    lap = loader._build_laps(make_laps(row), T0)[0]

    assert (lap.driver_code, lap.lap_number) == ("VER", 1)
    assert (lap.lap_time_ms, lap.completed_at, lap.compound) == (None, None, None)


def test_build_laps_converts_every_row_and_keeps_the_order():
    laps = make_laps(lap_row(LapNumber=1.0), lap_row(LapNumber=2.0), lap_row(Driver="NOR"))

    result = loader._build_laps(laps, T0)

    assert [(r.driver_code, r.lap_number) for r in result] == [("VER", 1), ("VER", 2), ("NOR", 1)]


# ------------------------------------------------------------------ trajetórias (posições)

CODES = {1: "VER", 4: "NOR"}


def test_build_tracks_converts_to_utc_datetimes_and_python_ints():
    pos = make_positions(
        ("2024-11-03 14:51:47.202", -1234.0, 567.0),
        ("2024-11-03 14:51:47.422", -1230.0, 570.0),
    )

    (track,) = loader._build_tracks({"1": pos}, CODES)

    assert track.driver_code == "VER"
    assert track.times == [
        datetime(2024, 11, 3, 14, 51, 47, 202000, tzinfo=UTC),
        datetime(2024, 11, 3, 14, 51, 47, 422000, tzinfo=UTC),
    ]
    assert track.xs == [-1234, -1230]
    assert track.ys == [567, 570]
    assert all(type(v) is int for v in track.xs + track.ys)  # int nativo, não numpy.int64


def test_build_tracks_drops_the_zero_zero_no_gps_placeholder_everywhere_in_the_series():
    """~14% das amostras reais são (0, 0): antes do sinal E no meio da corrida."""
    pos = make_positions(
        ("2024-11-03 14:38:07.701", 0.0, 0.0),  # antes do primeiro sinal
        ("2024-11-03 14:51:47.202", 100.0, 200.0),
        ("2024-11-03 14:51:47.422", 0.0, 0.0),  # perda de sinal no meio
        ("2024-11-03 14:51:47.642", 110.0, 210.0),
    )

    (track,) = loader._build_tracks({"1": pos}, CODES)

    assert (track.xs, track.ys) == ([100, 110], [200, 210])


def test_build_tracks_keeps_samples_where_only_one_coordinate_is_zero():
    # (0, 5) e (5, 0) são posições legítimas (o carro está sobre um eixo). Só (0, 0) é placeholder.
    pos = make_positions(
        ("2024-11-03 14:51:47.202", 0.0, 5.0),
        ("2024-11-03 14:51:47.422", 5.0, 0.0),
    )

    (track,) = loader._build_tracks({"1": pos}, CODES)

    assert (track.xs, track.ys) == ([0, 5], [5, 0])


@pytest.mark.parametrize("null_in", ["X", "Y", "both"])
def test_build_tracks_drops_samples_with_null_coordinates_instead_of_crashing(null_in):
    """Nulos nas posições: NaN em X/Y não pode derrubar a ingestão (int(NaN) levanta erro)."""
    x_null = NAN if null_in in ("X", "both") else 20.0
    y_null = NAN if null_in in ("Y", "both") else 21.0
    pos = make_positions(
        ("2024-11-03 14:51:47.202", 10.0, 11.0),
        ("2024-11-03 14:51:47.422", x_null, y_null),
        ("2024-11-03 14:51:47.642", 30.0, 31.0),
    )

    (track,) = loader._build_tracks({"1": pos}, CODES)

    assert (track.xs, track.ys) == ([10, 30], [11, 31])
    assert len(track.times) == 2


def test_build_tracks_drops_samples_with_null_timestamp():
    pos = make_positions(
        ("2024-11-03 14:51:47.202", 10.0, 11.0),
        ("NaT", 20.0, 21.0),  # sem instante não dá para ordenar nem gravar (time é PK)
    )

    (track,) = loader._build_tracks({"1": pos}, CODES)

    assert track.xs == [10]


def test_build_tracks_orders_samples_by_time():
    pos = make_positions(
        ("2024-11-03 14:51:47.642", 3.0, 3.0),
        ("2024-11-03 14:51:47.202", 1.0, 1.0),
        ("2024-11-03 14:51:47.422", 2.0, 2.0),
    )

    (track,) = loader._build_tracks({"1": pos}, CODES)

    assert track.xs == [1, 2, 3]
    assert track.times == sorted(track.times)


def test_build_tracks_driver_who_never_got_a_gps_fix_yields_an_empty_track():
    # Ex.: ALB (não largou): as 46.559 amostras reais são todas (0, 0).
    pos = make_positions(
        ("2024-11-03 14:38:07.701", 0.0, 0.0), ("2024-11-03 14:38:07.921", 0.0, 0.0)
    )

    (track,) = loader._build_tracks({"1": pos}, CODES)

    assert len(track) == 0
    assert track.times == [] and track.xs == [] and track.ys == []


def test_build_tracks_maps_car_number_keys_to_driver_codes():
    pos = make_positions(("2024-11-03 14:51:47.202", 1.0, 1.0))

    tracks = loader._build_tracks({"1": pos, "4": pos}, CODES)

    assert [t.driver_code for t in tracks] == ["VER", "NOR"]


def test_build_tracks_skips_cars_without_an_entry_and_warns(caplog):
    """O FastF1 já avisou de 'drivers' 241/242/243 (veículos de apoio) em dados reais."""
    pos = make_positions(("2024-11-03 14:51:47.202", 1.0, 1.0))

    with caplog.at_level(logging.WARNING, logger="ingestion.loader"):
        tracks = loader._build_tracks({"1": pos, "241": pos}, CODES)

    assert [t.driver_code for t in tracks] == ["VER"]
    assert "#241" in caplog.text


# ------------------------------------------------------------------ orquestração (load_race)


@pytest.fixture
def fake_fastf1(monkeypatch: pytest.MonkeyPatch):
    """Troca o módulo `fastf1` do loader por um falso: nenhum download acontece."""
    session = make_session(
        results=make_results(
            result_row(),
            result_row(
                DriverNumber="4", Abbreviation="NOR", FullName="Lando Norris", TeamName="McLaren"
            ),
        ),
        laps=make_laps(lap_row(), lap_row(Driver="NOR", LapTime=pd.NaT, Compound=None)),
        pos_data={
            "1": make_positions(("2024-11-03 14:51:47.202", 10.0, 11.0)),
            "4": make_positions(
                ("2024-11-03 14:51:47.202", 0.0, 0.0), ("2024-11-03 14:51:47.422", 5.0, 6.0)
            ),
            "241": make_positions(("2024-11-03 14:51:47.202", 9.0, 9.0)),  # carro sem inscrição
        },
    )
    session.load = Mock()
    fake = Mock()
    fake.get_session.return_value = session
    monkeypatch.setattr(loader, "fastf1", fake)
    return fake, session


def test_load_race_assembles_race_data_from_the_session(fake_fastf1, tmp_path):
    _, session = fake_fastf1

    data = loader.load_race(2024, 21, str(tmp_path / "cache"))

    assert isinstance(data, RaceData)
    assert data.race.name == "São Paulo Grand Prix"
    assert [e.driver_code for e in data.entries] == ["VER", "NOR"]
    assert [(lap.driver_code, lap.lap_time_ms) for lap in data.laps] == [
        ("VER", 99161),
        ("NOR", None),
    ]
    # NOR perdeu o (0, 0) do começo; #241 não tem inscrição e foi descartado.
    assert [(t.driver_code, t.xs) for t in data.tracks] == [("VER", [10]), ("NOR", [5])]
    session.load.assert_called_once_with(laps=True, telemetry=True, weather=False, messages=False)


def test_load_race_asks_fastf1_for_the_race_session_only(fake_fastf1, tmp_path):
    fake, _ = fake_fastf1

    loader.load_race(2024, 21, str(tmp_path / "cache"))

    fake.get_session.assert_called_once_with(2024, 21, "R")  # 'R' = corrida (não sprint/qualy)


def test_load_race_creates_the_cache_dir_and_enables_the_fastf1_cache(fake_fastf1, tmp_path):
    fake, _ = fake_fastf1
    cache_dir = tmp_path / "does" / "not" / "exist"

    loader.load_race(2024, 21, str(cache_dir))

    assert cache_dir.is_dir()  # o FastF1 exige que a pasta exista; o loader a cria
    fake.Cache.enable_cache.assert_called_once_with(str(cache_dir))


def test_load_race_fails_clearly_when_fastf1_cannot_determine_t0(fake_fastf1, tmp_path):
    _, session = fake_fastf1
    session.t0_date = None  # o FastF1 emite só um warning e segue com None

    with pytest.raises(RuntimeError, match="t0_date"):
        loader.load_race(2024, 21, str(tmp_path))
