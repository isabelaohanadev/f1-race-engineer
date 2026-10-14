"""Writer: grava um RaceData no banco de forma IDEMPOTENTE e ATÔMICA.

Idempotência (rodar duas vezes não duplica nem muda nada) vem de três escolhas combinadas:

1. Chaves naturais únicas no schema (ver db/models.py): o banco é quem garante a unicidade,
   não o código. Mesmo duas ingestões concorrentes não conseguem duplicar linhas.
2. Todo INSERT é um UPSERT (`INSERT ... ON CONFLICT`), nunca "checa se existe e depois insere"
   (que tem corrida entre o SELECT e o INSERT). A estratégia varia conforme o dado:
     * races/teams/drivers: DO UPDATE (precisamos do id de volta e o dado pode ter sido corrigido);
     * race_entries/laps: DO UPDATE só `WHERE` algo mudou, então a 2ª execução não reescreve linhas
       idênticas (sem versões mortas no Postgres) e o relatório mostra 0 alterações;
     * position_samples: DO NOTHING. Uma posição num instante é um fato imutável, e pular o
       conflito é o caminho mais barato para ~800 mil linhas.
3. Uma única transação por corrida (o commit é do chamador, via session_scope): ou a corrida
   entra inteira ou nada entra. Se a ingestão cair no meio, o banco não fica com uma corrida
   pela metade, e basta rodar de novo.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

from sqlalchemy import Connection, or_
from sqlalchemy.dialects.postgresql import Insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.models import Driver, Lap, PositionSample, Race, RaceEntry, Team
from ingestion.dto import EntryInfo, LapInfo, PositionTrack, RaceData, RaceInfo

logger = logging.getLogger(__name__)

# 10.000 linhas x 5 colunas = 50.000 parâmetros por INSERT, abaixo do limite de 65.535 do
# protocolo do Postgres. Lotes grandes reduzem idas e vindas ao banco; o limite fixa o teto.
POSITION_CHUNK_SIZE = 10_000


@dataclass(frozen=True, slots=True)
class IngestionReport:
    race_id: int
    entries_changed: int
    laps_offered: int
    laps_changed: int  # inseridas ou atualizadas; 0 numa re-execução idêntica
    positions_offered: int
    positions_inserted: int  # 0 numa re-execução idêntica
    tracks_skipped: int  # trajetórias de carros sem inscrição (não dá para gravar: FK)


def write_race(
    session: Session, data: RaceData, *, chunk_size: int = POSITION_CHUNK_SIZE
) -> IngestionReport:
    """Grava a corrida na sessão recebida. NÃO faz commit: a transação é de quem chamou."""
    # Core, não ORM: numa carga em lote o unit-of-work/identity map do ORM não agrega nada, só
    # custo. `session.connection()` é a MESMA transação da sessão, então o commit/rollback
    # continua sendo de quem chamou (session_scope).
    conn = session.connection()
    race_id = _upsert_race(conn, data.race)
    team_ids = _upsert_teams(conn, data.entries)
    driver_ids = _upsert_drivers(conn, data.entries)

    entries_changed = _upsert_entries(conn, race_id, data.entries, driver_ids, team_ids)
    laps_offered, laps_changed = _upsert_laps(
        conn, race_id, data.laps, driver_ids, keep_completed_at=data.laps_only
    )
    offered, inserted, skipped = _insert_positions(
        conn, race_id, data.tracks, driver_ids, chunk_size
    )

    return IngestionReport(
        race_id=race_id,
        entries_changed=entries_changed,
        laps_offered=laps_offered,
        laps_changed=laps_changed,
        positions_offered=offered,
        positions_inserted=inserted,
        tracks_skipped=skipped,
    )


def _execute_count(conn: Connection, stmt: Insert) -> int:
    """Executa o INSERT e devolve quantas linhas foram de fato inseridas/alteradas.

    O SQLAlchemy só captura o rowcount de UPDATE/DELETE; para INSERT ele devolve -1, a menos que
    se peça `preserve_rowcount`. É esse número que prova a idempotência no relatório (0 na 2ª vez).
    """
    return conn.execute(stmt.execution_options(preserve_rowcount=True)).rowcount


def _changed(model: type, stmt: Insert, fields: Iterable[str]):
    """Condição "alguma coluna difere do valor novo". IS DISTINCT FROM trata NULL corretamente
    (NULL = NULL é desconhecido no SQL; IS DISTINCT FROM considera dois NULL como iguais)."""
    return or_(*(getattr(model, f).is_distinct_from(stmt.excluded[f]) for f in fields))


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    it = iter(items)
    while batch := tuple(islice(it, size)):
        yield batch


def _upsert_race(conn: Connection, race: RaceInfo) -> int:
    fields = ("name", "location", "country", "starts_at", "total_laps")
    stmt = pg_insert(Race).values(
        year=race.year,
        round=race.round,
        **{f: getattr(race, f) for f in fields},
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Race.year, Race.round],
        set_={f: stmt.excluded[f] for f in fields},
    ).returning(Race.id)
    return conn.execute(stmt).scalar_one()


def _upsert_teams(conn: Connection, entries: Sequence[EntryInfo]) -> dict[str, int]:
    # Dedup antes: várias entradas por equipe. Um ON CONFLICT DO UPDATE não pode atingir a mesma
    # linha duas vezes no mesmo comando (erro "cannot affect row a second time").
    unique = {e.team_name: e.team_color for e in entries}
    stmt = pg_insert(Team).values([{"name": n, "color_hex": c} for n, c in unique.items()])
    stmt = stmt.on_conflict_do_update(
        index_elements=[Team.name], set_={"color_hex": stmt.excluded.color_hex}
    ).returning(Team.id, Team.name)
    return {name: id_ for id_, name in conn.execute(stmt)}


def _upsert_drivers(conn: Connection, entries: Sequence[EntryInfo]) -> dict[str, int]:
    unique = {e.driver_code: e.full_name for e in entries}
    stmt = pg_insert(Driver).values([{"code": c, "full_name": n} for c, n in unique.items()])
    stmt = stmt.on_conflict_do_update(
        index_elements=[Driver.code], set_={"full_name": stmt.excluded.full_name}
    ).returning(Driver.id, Driver.code)
    return {code: id_ for id_, code in conn.execute(stmt)}


def _upsert_entries(
    conn: Connection,
    race_id: int,
    entries: Sequence[EntryInfo],
    driver_ids: dict[str, int],
    team_ids: dict[str, int],
) -> int:
    fields = ("team_id", "driver_number", "grid_position", "finish_position", "status")
    rows = [
        {
            "race_id": race_id,
            "driver_id": driver_ids[e.driver_code],
            "team_id": team_ids[e.team_name],
            "driver_number": e.driver_number,
            "grid_position": e.grid_position,
            "finish_position": e.finish_position,
            "status": e.status,
        }
        for e in entries
    ]
    stmt = pg_insert(RaceEntry).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[RaceEntry.race_id, RaceEntry.driver_id],
        set_={f: stmt.excluded[f] for f in fields},
        where=_changed(RaceEntry, stmt, fields),
    )
    return _execute_count(conn, stmt)


LAP_FIELDS = (
    "lap_time_ms",
    "compound",
    "tyre_life",
    "stint",
    "position",
    "completed_at",
    "pit_in",
    "pit_out",
    "track_status",
    "session_time_ms",
)


def _upsert_laps(
    conn: Connection,
    race_id: int,
    laps: Sequence[LapInfo],
    driver_ids: dict[str, int],
    *,
    keep_completed_at: bool = False,
) -> tuple[int, int]:
    # `keep_completed_at` (ingestão --laps-only): a coluna fica FORA do INSERT e do UPDATE. Sem
    # telemetria não há t0_date, então o valor nulo que o loader traz não significa "a volta não tem
    # horário", e gravá-lo apagaria os horários que uma ingestão completa anterior já deixou lá (o
    # replay depende deles). Volta nova entra com completed_at nulo (o default da coluna).
    fields = tuple(f for f in LAP_FIELDS if not (keep_completed_at and f == "completed_at"))
    # Dict pela chave natural: se o FastF1 mandasse a mesma volta duas vezes, ficaria a última.
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for lap in laps:
        driver_id = driver_ids.get(lap.driver_code)
        if driver_id is None:
            logger.warning(
                "Volta %s/%s ignorada: piloto fora dos resultados", lap.driver_code, lap.lap_number
            )
            continue
        rows[(driver_id, lap.lap_number)] = {
            "race_id": race_id,
            "driver_id": driver_id,
            "lap_number": lap.lap_number,
            **{f: getattr(lap, f) for f in fields},
        }

    changed = 0
    for batch in _batched(list(rows.values()), 2_000):
        stmt = pg_insert(Lap).values(list(batch))
        stmt = stmt.on_conflict_do_update(
            index_elements=[Lap.race_id, Lap.driver_id, Lap.lap_number],
            set_={f: stmt.excluded[f] for f in fields},
            where=_changed(Lap, stmt, fields),
        )
        changed += _execute_count(conn, stmt)
    return len(rows), changed


def _insert_positions(
    conn: Connection,
    race_id: int,
    tracks: Sequence[PositionTrack],
    driver_ids: dict[str, int],
    chunk_size: int,
) -> tuple[int, int, int]:
    offered = inserted = skipped = 0
    for track in tracks:
        driver_id = driver_ids.get(track.driver_code)
        if driver_id is None:
            skipped += 1
            logger.warning(
                "Trajetória de %s ignorada: piloto fora dos resultados", track.driver_code
            )
            continue

        driver_inserted = 0
        for start in range(0, len(track), chunk_size):
            end = start + chunk_size
            rows = [
                {"race_id": race_id, "driver_id": driver_id, "time": t, "x": x, "y": y}
                for t, x, y in zip(
                    track.times[start:end], track.xs[start:end], track.ys[start:end], strict=True
                )
            ]
            # DO NOTHING: para um fato imutável é o mais barato. O rowcount de um INSERT ... ON
            # CONFLICT DO NOTHING conta só as linhas realmente inseridas (as puladas não contam).
            stmt = pg_insert(PositionSample).values(rows).on_conflict_do_nothing()
            offered += len(rows)
            driver_inserted += _execute_count(conn, stmt)
        inserted += driver_inserted
        logger.info("  %s: %d amostras (%d novas)", track.driver_code, len(track), driver_inserted)
    return offered, inserted, skipped
