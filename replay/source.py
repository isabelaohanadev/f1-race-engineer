"""Lê uma corrida do banco como um fluxo de eventos em ordem de tempo, com memória constante.

O problema: `position_samples` tem ~800 mil linhas por corrida (~600 mil depois de descartar os
sinais de GPS nulos). Carregar tudo seria ~100 MB só de tuplas Python antes de publicar o 1º evento.

A solução, em 3 partes:

1. PAGINAÇÃO POR CHAVE (keyset), não por OFFSET. `OFFSET n` obriga o Postgres a produzir e
   descartar as n primeiras linhas a cada página: a página 100 custa 100x a página 1. No keyset a
   próxima página começa "depois da última linha que vi" e o índice leva direto até lá: o custo de
   TODA página é o mesmo. A chave é `(time, driver_id)`: só `time` não serve, porque dois carros
   podem ter amostras no mesmo instante e uma página cortada entre eles perderia ou repetiria
   linhas. `(time, driver_id)` é total (com race_id fixo é único pela PK) e a ordem de leitura é a
   do índice `(race_id, time)`.
2. SESSÃO CURTA POR PÁGINA. Cada página abre e fecha a própria sessão: a conexão volta ao pool na
   hora, em vez de ficar presa por ~2 horas de replay (e sem cursor/transação longa no servidor).
3. PREFETCH. Enquanto o motor publica a página N, a N+1 já está sendo lida numa thread (o driver é
   síncrono). Memória: no máximo 2 páginas. A leitura do banco nunca atrasa o ritmo do replay.

As ~1.100 voltas cabem em memória (poucos KB) e são mescladas, em ordem de tempo, ao fluxo de
posições.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Select, select, tuple_
from sqlalchemy.orm import Session, sessionmaker

from db.models import Driver, Lap, PositionSample, Race, RaceEntry
from replay.errors import RaceHasNoData, RaceNotFound
from replay.events import (
    STATE_FINISHED,
    STATE_RUNNING,
    STATE_STARTED,
    LapCompletedEvent,
    PositionEvent,
    RaceStatusEvent,
    ReplayEvent,
    to_epoch_ms,
)

# 5.000 linhas ≈ 0,5 MB por página; a 16x são ~3 s de corrida, tempo de sobra para o prefetch.
DEFAULT_PAGE_SIZE = 5_000

PageKey = tuple[datetime, int]  # (time, driver_id) da última linha da página anterior


@dataclass(frozen=True, slots=True)
class _PositionRow:
    time: datetime
    driver_id: int
    x: int
    y: int


def position_page_query(race_id: int, after: PageKey | None, limit: int) -> Select[Any]:
    """A consulta de UMA página de posições, em ordem de (time, driver_id), DEPOIS de `after`."""
    stmt = select(
        PositionSample.time, PositionSample.driver_id, PositionSample.x, PositionSample.y
    ).where(PositionSample.race_id == race_id)
    if after is not None:
        last_time, last_driver = after
        stmt = stmt.where(
            # `time >=` é logicamente redundante com a comparação de tupla da linha de baixo (o
            # Postgres até deriva essa faixa sozinho para o índice). Fica por causa do TimescaleDB:
            # só uma condição SIMPLES em `time` é usada para EXCLUIR CHUNKS no planejamento. Medido
            # com 6 chunks: com ela o plano toca 3 (os anteriores ao cursor são descartados), sem
            # ela toca os 6. Com a hypertable crescendo (1 chunk por semana de corrida), é isso que
            # mantém uma página custando o mesmo. Há teste de regressão em tests/integration.
            PositionSample.time >= last_time,
            tuple_(PositionSample.time, PositionSample.driver_id) > tuple_(last_time, last_driver),
        )
    return stmt.order_by(PositionSample.time, PositionSample.driver_id).limit(limit)


def _fetch_position_page(
    session_factory: sessionmaker[Session], race_id: int, after: PageKey | None, limit: int
) -> list[_PositionRow]:
    with session_factory() as session:
        rows = session.execute(position_page_query(race_id, after, limit))
        return [_PositionRow(row.time, row.driver_id, row.x, row.y) for row in rows]


class _LeaderTracker:
    """Deduz "em que volta o líder está" a partir das voltas concluídas.

    Quando o PRIMEIRO carro conclui a volta N, o líder passa a estar na volta N+1 (limitada ao total
    de voltas). Carros retardatários concluindo voltas antigas não mudam nada.
    """

    def __init__(self, total_laps: int | None) -> None:
        self._total = total_laps
        self.current_lap = 1
        self._best_completed = 0

    def observe(self, lap: LapCompletedEvent) -> RaceStatusEvent | None:
        self._best_completed = max(self._best_completed, lap.lap)
        next_lap = self._best_completed + 1
        if self._total is not None:
            next_lap = min(next_lap, self._total)
        if next_lap <= self.current_lap:
            return None
        self.current_lap = next_lap
        return RaceStatusEvent(STATE_RUNNING, next_lap, lap.t, self._total)


class RaceSource:
    """Eventos de uma corrida, em ordem de tempo. De uso ÚNICO: `events()` consome a leitura."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        race_id: int,
        name: str,
        total_laps: int | None,
        driver_codes: dict[int, str],
        laps: list[LapCompletedEvent],
        first_page: list[_PositionRow],
        page_size: int,
    ) -> None:
        self._session_factory = session_factory
        self._race_id = race_id
        self._driver_codes = driver_codes
        self._laps = laps
        self._first_page = first_page
        self._page_size = page_size
        self._consumed = False
        self.name = name
        self.total_laps = total_laps

        candidates = [to_epoch_ms(first_page[0].time)] if first_page else []
        if laps:
            candidates.append(laps[0].t)
        if not candidates:
            raise RaceHasNoData(f"corrida {name!r} não tem posições nem voltas para reproduzir")
        # Origem do tempo de corrida: o primeiro evento (o replay não espera a "largada").
        self.t0_ms = min(candidates)

    @classmethod
    async def open(
        cls,
        session_factory: sessionmaker[Session],
        year: int,
        round_number: int,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> "RaceSource":
        """Valida que a corrida existe e tem dados, e lê o que cabe em memória (metadados, voltas
        e a 1ª página de posições). Falha AQUI, antes de o replay começar, não no meio dele."""
        if page_size < 1:
            raise ValueError("page_size deve ser >= 1")
        return await asyncio.to_thread(
            cls._open_sync, session_factory, year, round_number, page_size
        )

    @classmethod
    def _open_sync(
        cls, session_factory: sessionmaker[Session], year: int, round_number: int, page_size: int
    ) -> "RaceSource":
        with session_factory() as session:
            race = session.scalar(select(Race).where(Race.year == year, Race.round == round_number))
            if race is None:
                raise RaceNotFound(f"corrida {year}/{round_number} não encontrada no banco")
            codes = {
                row.driver_id: row.code
                for row in session.execute(
                    select(RaceEntry.driver_id, Driver.code)
                    .join(Driver, Driver.id == RaceEntry.driver_id)
                    .where(RaceEntry.race_id == race.id)
                )
            }
            lap_rows = session.execute(
                select(
                    Driver.code,
                    Lap.lap_number,
                    Lap.lap_time_ms,
                    Lap.compound,
                    Lap.position,
                    Lap.completed_at,
                )
                .join(Driver, Driver.id == Lap.driver_id)
                # Volta sem horário de término não tem lugar na linha do tempo: fica de fora.
                .where(Lap.race_id == race.id, Lap.completed_at.is_not(None))
            ).all()
            race_id, name, total_laps = race.id, race.name, race.total_laps

        laps = sorted(
            (
                LapCompletedEvent(
                    driver=row.code,
                    lap=row.lap_number,
                    t=to_epoch_ms(row.completed_at),
                    lap_time_ms=row.lap_time_ms,
                    compound=row.compound,
                    position=row.position,
                )
                for row in lap_rows
            ),
            key=lambda e: (e.t, e.driver, e.lap),
        )
        first_page = _fetch_position_page(session_factory, race_id, None, page_size)
        return cls(
            session_factory=session_factory,
            race_id=race_id,
            name=name,
            total_laps=total_laps,
            driver_codes=codes,
            laps=laps,
            first_page=first_page,
            page_size=page_size,
        )

    async def _positions(self) -> AsyncIterator[PositionEvent]:
        page, self._first_page = self._first_page, []  # solta a referência: não segura 2 cópias
        prefetch: asyncio.Task[list[_PositionRow]] | None = None
        try:
            while page:
                last = page[-1]
                # Página cheia: pode haver mais; já lê a próxima ENQUANTO esta é publicada.
                # Página curta: era a última, não há o que buscar.
                prefetch = (
                    asyncio.create_task(
                        asyncio.to_thread(
                            _fetch_position_page,
                            self._session_factory,
                            self._race_id,
                            (last.time, last.driver_id),
                            self._page_size,
                        )
                    )
                    if len(page) == self._page_size
                    else None
                )
                for row in page:
                    yield PositionEvent(
                        driver=self._driver_codes[row.driver_id],
                        x=row.x,
                        y=row.y,
                        t=to_epoch_ms(row.time),
                    )
                page = await prefetch if prefetch is not None else []
                prefetch = None
        finally:
            # Replay parado no meio (stop/erro): não deixa a leitura em andamento "órfã".
            if prefetch is not None:
                prefetch.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await prefetch

    async def events(self) -> AsyncIterator[ReplayEvent]:
        """Todos os eventos da corrida em ordem de tempo.

        Ordem total e determinística: por `t` (ms); no MESMO `t`, posições antes de voltas, e cada
        volta é seguida do `race_status` que ela provoca. `started` abre e `finished` fecha.
        """
        if self._consumed:
            raise RuntimeError("RaceSource é de uso único: crie outra com RaceSource.open()")
        self._consumed = True

        tracker = _LeaderTracker(self.total_laps)
        yield RaceStatusEvent(STATE_STARTED, 1, self.t0_ms, self.total_laps)
        last_t = self.t0_ms
        laps, next_lap = self._laps, 0

        def lap_events(lap: LapCompletedEvent) -> list[ReplayEvent]:
            status = tracker.observe(lap)
            return [lap] if status is None else [lap, status]

        # `aclosing`: se o motor fechar ESTE gerador no meio (stop/erro), o `_positions()` interno
        # também fecha NA HORA e cancela o prefetch. Sem isso ele só seria finalizado quando o
        # coletor de lixo decidisse, e a leitura do banco ficaria rodando órfã até lá.
        async with aclosing(self._positions()) as positions:
            async for position in positions:
                while next_lap < len(laps) and laps[next_lap].t < position.t:
                    for event in lap_events(laps[next_lap]):
                        last_t = max(last_t, event.t)
                        yield event
                    next_lap += 1
                last_t = max(last_t, position.t)
                yield position
        for lap in laps[next_lap:]:
            for event in lap_events(lap):
                last_t = max(last_t, event.t)
                yield event
        yield RaceStatusEvent(STATE_FINISHED, tracker.current_lap, last_t, self.total_laps)
