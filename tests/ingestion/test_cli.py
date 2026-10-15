import logging
from unittest.mock import Mock

import pytest

from ingestion import cli
from ingestion.writer import IngestionReport

REPORT = IngestionReport(
    race_id=7,
    entries_changed=20,
    laps_offered=1134,
    laps_changed=1134,
    positions_offered=797642,
    positions_inserted=797642,
    tracks_skipped=0,
)


def test_main_returns_zero_and_forwards_the_arguments(monkeypatch: pytest.MonkeyPatch, caplog):
    ingest = Mock(return_value=REPORT)
    monkeypatch.setattr(cli, "ingest_race", ingest)

    with caplog.at_level(logging.INFO, logger="ingestion"):
        code = cli.main(["--year", "2024", "--round", "21", "--cache-dir", "/tmp/cache"])

    assert code == 0
    ingest.assert_called_once_with(2024, 21, "/tmp/cache", laps_only=False)
    assert "797642 inseridas de 797642" in caplog.text
    assert "só voltas" not in caplog.text


def test_main_cache_dir_defaults_to_none_so_the_settings_value_applies(monkeypatch):
    ingest = Mock(return_value=REPORT)
    monkeypatch.setattr(cli, "ingest_race", ingest)

    cli.main(["--year", "2024", "--round", "21"])

    ingest.assert_called_once_with(2024, 21, None, laps_only=False)


def test_main_laps_only_flag_is_forwarded_and_announced(monkeypatch, caplog):
    ingest = Mock(return_value=REPORT)
    monkeypatch.setattr(cli, "ingest_race", ingest)

    with caplog.at_level(logging.INFO, logger="ingestion"):
        cli.main(["--year", "2024", "--round", "6", "--laps-only"])

    ingest.assert_called_once_with(2024, 6, None, laps_only=True)
    assert "só voltas" in caplog.text


def test_main_returns_one_and_logs_when_the_ingestion_fails(monkeypatch, caplog):
    monkeypatch.setattr(cli, "ingest_race", Mock(side_effect=RuntimeError("boom")))

    with caplog.at_level(logging.ERROR, logger="ingestion"):
        code = cli.main(["--year", "2024", "--round", "21"])

    assert code == 1  # o shell/CI enxerga a falha pelo exit code
    assert "rollback" in caplog.text


@pytest.mark.parametrize(
    "argv", [[], ["--year", "2024"], ["--round", "21"], ["--year", "x", "--round", "1"]]
)
def test_main_rejects_missing_or_invalid_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 2  # convenção do argparse para erro de uso
