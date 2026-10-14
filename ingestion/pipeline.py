from app.config import get_settings
from db.session import session_scope
from ingestion.loader import load_race
from ingestion.writer import IngestionReport, write_race


def ingest_race(year: int, round_number: int, cache_dir: str | None = None) -> IngestionReport:
    """Baixa a corrida e grava no banco. Seguro para rodar quantas vezes quiser (idempotente)."""
    cache_dir = cache_dir or get_settings().fastf1_cache_dir

    # 1) Rede/CPU primeiro, SEM transação aberta: o download pode levar minutos, e manter uma
    #    transação (e uma conexão do pool) parada esse tempo todo é desperdício e risco.
    data = load_race(year, round_number, cache_dir)

    # 2) Só então abrimos a transação: gravação rápida, tudo-ou-nada.
    with session_scope() as session:
        return write_race(session, data)
