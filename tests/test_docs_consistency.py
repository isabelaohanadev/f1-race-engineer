"""Consistência entre os textos: o que o backtest mediu (a forma antiga do alerta) não pode aparecer
como se valesse para o alerta que o consumidor entrega (a forma ao vivo)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Os textos que DESCREVEM o alerta que roda ao vivo. A precisão da forma antiga (a do backtest) fica
# só no relatório do backtest; aqui ela não pode reaparecer, nem por engano.
LIVE_ALERT_TEXTS = [
    "README.md",
    "app/schemas.py",
    "frontend/index.html",
    "frontend/race_view.js",
    "docs/undercut.md",
    "docs/estrategia-ao-vivo.md",
    "docs/montecarlo.md",
]


@pytest.mark.parametrize("path", LIVE_ALERT_TEXTS)
def test_the_old_form_precision_is_not_quoted_in_the_texts_about_the_live_alert(path: str):
    text = (ROOT / path).read_text()

    assert "83%" not in text and "83 %" not in text, path
    assert not re.search(r"18 alertas", text), path  # o n da forma antiga também


def test_the_backtest_report_labels_its_undercut_numbers_as_the_old_form():
    text = " ".join((ROOT / "docs" / "backtest.md").read_text().split())

    assert "Esta é a forma ANTIGA do alerta" in text
    assert "Undercut, forma antiga" in text and "não vale para o consumidor" in text
    assert "undercut, forma antiga: alertas / precisão" in text


def test_every_text_that_points_to_the_live_evaluation_points_to_a_file_that_exists():
    assert (ROOT / "docs" / "undercut-ao-vivo.md").is_file()
    for path in ("README.md", "app/schemas.py", "docs/undercut.md", "docs/estrategia-ao-vivo.md"):
        assert "docs/undercut-ao-vivo.md" in (ROOT / path).read_text(), path


def test_the_live_evaluation_lists_no_post_result_adjustment_and_keeps_the_frozen_values():
    text = " ".join((ROOT / "docs" / "undercut-ao-vivo.md").read_text().split())

    assert "## Ajustes pós-resultado Nenhum." in text
    assert "o congelado (1.30)" in text  # o fator em uso é o congelado
    assert "Diagnóstico, não aplicado" in text


def readme() -> str:
    return (ROOT / "README.md").read_text()


def test_the_backtest_results_come_before_the_limitations_and_say_the_bad_news_plainly():
    text = readme()
    results = text.index("## Resultados do backtest")

    assert results < text.index("## Limitações conhecidas")
    section = " ".join(text[results : text.index("## Limitações conhecidas")].split())
    assert "O resultado é ruim" in section
    assert (
        "Não supera em nenhuma das 10 faixas" in section
        or "**Não.** Não supera em nenhuma" in section
    )
    assert "pior depois de 80%" in section
    assert "baseline_win" in section and "reliability" in section


def test_the_opening_of_the_readme_already_says_the_monte_carlo_does_not_beat_the_leader():
    opening = " ".join(readme()[: readme().index("## Arquitetura")].split())

    assert "não supera" in opening and "pior depois de 80% da corrida" in opening


def test_the_full_flow_diagram_has_both_tracks_from_the_database_to_the_client():
    text = readme()
    start = text.index("### O fluxo completo")
    diagram = text[start : text.index("### Como um cliente entra no meio do replay")]

    for piece in (
        "TimescaleDB", "ReplayEngine", "StreamPublisher", "Stream race:ano:etapa",
        "Stream laps:ano:etapa", "worker", "state-workers", "Hash state:ano:etapa",
        "strategy: 2 cópias, 1 ativa", "strategy:consumer:lock", "strategy:result:ano:etapa",
        "strategy:events:ano:etapa", "Hub por corrida", "WebSocket",
        "GET /races/ano/etapa/strategy", "standby", "cada lap_completed",
    ):  # fmt: skip
        assert piece in diagram, piece
    assert diagram.count("```mermaid") == 1
