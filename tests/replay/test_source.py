import asyncio
import random

import pytest
from sqlalchemy.dialects import postgresql

from replay import source as source_module
from replay.errors import RaceHasNoData, RaceNotFound
from replay.events import (
    LapCompletedEvent,
    PositionEvent,
    RaceStatusEvent,
    to_epoch_ms,
)
from replay.source import RaceSource, position_page_query
from tests.replay.conftest import ALB, NOR, VER, at


def ms(offset: int) -> int:
    return to_epoch_ms(at(offset))


async def collect(session_factory, *, page_size=5000, year=2024, round_number=21):
    source = await RaceSource.open(session_factory, year, round_number, page_size=page_size)
    return [e async for e in source.events()]


# ---------------------------------------------------------------- ordem


async def test_positions_and_laps_come_out_in_time_order_with_status_events(seed, session_factory):
    seed(
        positions=[
            # Inseridos DESORDENADOS de propósito: a ordem tem de vir da consulta, não da inserção.
            (NOR, at(200), 5, 5),
            (VER, at(400), 4, 4),
            (VER, at(0), 1, 1),
            (NOR, at(0), 2, 2),  # empate de horário com o VER: desempata por driver_id
            (VER, at(200), 3, 3),
        ],
        laps=[
            (VER, 1, at(200), 90_000, "SOFT", 1),  # no MESMO instante de duas posições
            (NOR, 1, at(300), 91_000, "HARD", 2),
            (VER, 2, at(500), 89_000, "SOFT", 1),
            (VER, 3, at(600), 88_000, "SOFT", 1),  # volta final: limitada ao total (3)
            (ALB, 1, None, None, None, None),  # sem horário de término: fica de fora
        ],
        total_laps=3,
    )

    events = await collect(session_factory)

    assert events == [
        RaceStatusEvent("started", 1, ms(0), 3),
        PositionEvent("VER", 1, 1, ms(0)),
        PositionEvent("NOR", 2, 2, ms(0)),
        PositionEvent("VER", 3, 3, ms(200)),
        PositionEvent("NOR", 5, 5, ms(200)),
        # volta no mesmo instante das posições: DEPOIS delas; e o líder passa à volta 2
        LapCompletedEvent("VER", 1, ms(200), 90_000, "SOFT", 1),
        RaceStatusEvent("running", 2, ms(200), 3),
        LapCompletedEvent("NOR", 1, ms(300), 91_000, "HARD", 2),  # retardatário: sem status
        PositionEvent("VER", 4, 4, ms(400)),
        LapCompletedEvent("VER", 2, ms(500), 89_000, "SOFT", 1),
        RaceStatusEvent("running", 3, ms(500), 3),
        LapCompletedEvent("VER", 3, ms(600), 88_000, "SOFT", 1),  # volta final: sem status novo
        RaceStatusEvent("finished", 3, ms(600), 3),
    ]


async def test_without_total_laps_the_leader_lap_is_not_capped(seed, session_factory):
    seed(
        positions=[(VER, at(0), 0, 0)],
        laps=[(VER, 1, at(10), 1, None, 1), (VER, 2, at(20), 1, None, 1)],
        total_laps=None,
    )

    statuses = [e for e in await collect(session_factory) if isinstance(e, RaceStatusEvent)]

    assert [(s.state, s.lap, s.total_laps) for s in statuses] == [
        ("started", 1, None),
        ("running", 2, None),
        ("running", 3, None),
        ("finished", 3, None),
    ]


async def test_laps_only_race_is_replayable(seed, session_factory):
    seed(positions=[], laps=[(VER, 1, at(1000), 90_000, "SOFT", 1)])

    events = await collect(session_factory)

    assert [type(e).__name__ for e in events] == [
        "RaceStatusEvent",
        "LapCompletedEvent",
        "RaceStatusEvent",
        "RaceStatusEvent",
    ]


async def test_origin_is_the_earliest_of_first_position_and_first_lap(seed, session_factory):
    seed(
        positions=[(VER, at(500), 0, 0)],
        laps=[(NOR, 1, at(100), None, None, None)],  # a volta vem ANTES da primeira posição
    )

    source = await RaceSource.open(session_factory, 2024, 21)

    assert source.t0_ms == ms(100)


# ---------------------------------------------------------------- paginação keyset


def _grid(seed):
    """40 instantes x 3 pilotos = 120 posições, com TODOS os instantes repetidos entre pilotos
    (empates em `time`, o caso que quebra uma paginação feita só por horário)."""
    rows = [(d, at(t * 220), t, d) for t in range(40) for d in (VER, NOR, ALB)]
    random.Random(7).shuffle(rows)  # inserção fora de ordem
    seed(positions=rows)


@pytest.mark.parametrize("page_size", [1, 2, 3, 7, 50, 119, 120, 121, 5000])
async def test_pagination_gives_the_same_events_for_every_page_size(
    seed, session_factory, page_size
):
    """Páginas cortando bem no meio de grupos de empate não podem perder nem repetir linha."""
    _grid(seed)

    paged = await collect(session_factory, page_size=page_size)
    whole = await collect(session_factory, page_size=5000)

    assert paged == whole
    positions = [e for e in paged if isinstance(e, PositionEvent)]
    assert len(positions) == 120
    assert len({(p.driver, p.t) for p in positions}) == 120  # nenhuma repetida
    assert [p.t for p in positions] == sorted(p.t for p in positions)  # em ordem de tempo


async def test_no_query_reads_more_than_one_page(seed, session_factory, monkeypatch):
    _grid(seed)
    fetched: list[int] = []
    original = source_module._fetch_position_page

    def spy(*args, **kwargs):
        rows = original(*args, **kwargs)
        fetched.append(len(rows))
        return rows

    monkeypatch.setattr(source_module, "_fetch_position_page", spy)

    await collect(session_factory, page_size=50)

    assert fetched == [50, 50, 20]  # 120 linhas lidas em 3 páginas; nenhuma passou de 50
    assert sum(fetched) == 120


async def test_a_row_count_that_is_a_multiple_of_the_page_size_still_terminates(
    seed, session_factory
):
    seed(positions=[(VER, at(i), i, i) for i in range(8)])

    events = await collect(session_factory, page_size=4)

    assert len([e for e in events if isinstance(e, PositionEvent)]) == 8  # sem laço infinito


def test_page_query_is_keyset_and_limit_never_offset():
    """Compila a consulta para o dialeto do POSTGRES (o de produção; o SQLite emite um
    `OFFSET 0` por peculiaridade do dialeto dele) e confere a forma: filtro de corrida, condição
    de faixa em `time` para o índice, comparação de tupla, ordem e LIMIT. Sem OFFSET."""
    stmt = position_page_query(race_id=1, after=(at(500), 2), limit=5000)

    sql = str(stmt.compile(dialect=postgresql.dialect())).upper()

    assert "WHERE POSITION_SAMPLES.RACE_ID =" in sql
    assert "POSITION_SAMPLES.TIME >=" in sql  # faixa simples: o planejador a usa no índice
    assert "(POSITION_SAMPLES.TIME, POSITION_SAMPLES.DRIVER_ID) >" in sql  # a chave keyset
    assert "ORDER BY POSITION_SAMPLES.TIME, POSITION_SAMPLES.DRIVER_ID" in sql
    assert "LIMIT" in sql
    assert "OFFSET" not in sql  # OFFSET faria cada página custar mais que a anterior


def test_first_page_has_no_keyset_condition():
    sql = str(position_page_query(1, None, 10).compile(dialect=postgresql.dialect())).upper()

    assert "POSITION_SAMPLES.TIME >=" not in sql and "LIMIT" in sql


async def test_closing_the_stream_early_leaves_no_orphan_read(seed, session_factory):
    _grid(seed)
    source = await RaceSource.open(session_factory, 2024, 21, page_size=10)
    stream = source.events()

    for _ in range(3):  # lê um pedaço da 1ª página (o prefetch da 2ª já foi disparado)
        await anext(stream)
    await stream.aclose()  # o motor faz isto ao parar o replay

    leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert leftover == []


async def test_a_source_is_single_use(seed, session_factory):
    seed(positions=[(VER, at(0), 0, 0)])
    source = await RaceSource.open(session_factory, 2024, 21)
    [_ async for _ in source.events()]

    with pytest.raises(RuntimeError, match="uso único"):
        [_ async for _ in source.events()]


# ---------------------------------------------------------------- erros na abertura


async def test_unknown_race_raises_race_not_found(session_factory):
    with pytest.raises(RaceNotFound, match="2099/1"):
        await RaceSource.open(session_factory, 2099, 1)


async def test_race_without_any_data_raises_race_has_no_data(seed, session_factory):
    seed(positions=[], laps=[(VER, 1, None, None, None, None)])  # só uma volta sem horário

    with pytest.raises(RaceHasNoData):
        await RaceSource.open(session_factory, 2024, 21)


async def test_invalid_page_size_is_rejected(seed, session_factory):
    seed(positions=[(VER, at(0), 0, 0)])

    with pytest.raises(ValueError):
        await RaceSource.open(session_factory, 2024, 21, page_size=0)


async def test_another_race_in_the_same_database_does_not_leak_in(seed, session_factory):
    seed(positions=[(VER, at(0), 1, 1)], year=2024, round_number=21, race_id=1)
    seed(positions=[(VER, at(0), 99, 99)], year=2023, round_number=1, race_id=2)

    positions = [e for e in await collect(session_factory) if isinstance(e, PositionEvent)]

    assert [(p.x, p.y) for p in positions] == [(1, 1)]
