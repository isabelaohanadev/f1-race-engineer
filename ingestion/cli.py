import argparse
import logging
import sys
import time

from ingestion.pipeline import ingest_race


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion",
        description="Baixa uma corrida de F1 (FastF1) e grava no banco. Idempotente.",
    )
    parser.add_argument("--year", type=int, required=True, help="temporada, ex.: 2024")
    parser.add_argument(
        "--round", type=int, required=True, dest="round_number", help="etapa, ex.: 21"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="cache do FastF1 (default: FASTF1_CACHE_DIR do ambiente)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="logs detalhados (inclui FastF1)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    if not args.verbose:
        # O FastF1 é muito verboso no INFO; só mostramos avisos e erros dele.
        logging.getLogger("fastf1").setLevel(logging.WARNING)

    log = logging.getLogger("ingestion")
    log.info("Ingerindo %s, etapa %s ...", args.year, args.round_number)
    started = time.perf_counter()
    try:
        report = ingest_race(args.year, args.round_number, args.cache_dir)
    except Exception:
        log.exception("Ingestão falhou; nada foi gravado (a transação sofreu rollback).")
        return 1

    log.info("Concluído em %.1fs (race_id=%d)", time.perf_counter() - started, report.race_id)
    log.info("  inscrições alteradas : %d", report.entries_changed)
    log.info(
        "  voltas               : %d alteradas de %d", report.laps_changed, report.laps_offered
    )
    log.info(
        "  posições             : %d inseridas de %d (0 novas = já estava tudo no banco)",
        report.positions_inserted,
        report.positions_offered,
    )
    if report.tracks_skipped:
        log.warning("  trajetórias ignoradas: %d", report.tracks_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
