"""A página de teste do WebSocket (frontend/): servida pela API e com as MESMAS regras do protocolo
que o cliente Python de referência (conformidade, rodando o JS de verdade no node)."""

import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from scripts.watch_race import RaceView

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


# ---------------------------------------------------------------- servida pelo FastAPI


@pytest.fixture
def api() -> TestClient:
    with TestClient(app) as client:
        yield client


def test_the_page_is_served_by_the_api_at_frontend(api: TestClient):
    response = api.get("/frontend/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # Conecta na MESMA origem (sem CORS) e usa o módulo do protocolo, não um servidor externo.
    assert "/ws/races/" in response.text and "location.host" in response.text
    assert '<script src="race_view.js">' in response.text


def test_the_protocol_module_is_served_as_javascript(api: TestClient):
    response = api.get("/frontend/race_view.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "class RaceView" in response.text


def test_frontend_without_the_trailing_slash_redirects_to_the_page(api: TestClient):
    response = api.get("/frontend", follow_redirects=False)

    assert response.status_code in (307, 308) and response.headers["location"].endswith(
        "/frontend/"
    )


def test_the_page_never_injects_network_data_as_html():
    """Tudo o que chega do WebSocket vai para a tela por textContent. Atribuir a `.innerHTML` (ou
    avaliar texto como código) com dado de rede seria uma porta para XSS."""
    html = (ROOT / "frontend" / "index.html").read_text()
    js = (ROOT / "frontend" / "race_view.js").read_text()

    dangerous = re.compile(
        r"\.innerHTML|\.outerHTML|insertAdjacentHTML\s*\(|\beval\s*\(|document\.write"
    )
    assert not dangerous.search(html) and not dangerous.search(js)


# ---------------------------------------------------------------- conformidade Python x JS


def random_scenario(rng: random.Random) -> dict:
    """Eventos de até 3 replays (epochs), com repetições, atrasos, campos nulos e ordem embaralhada,
    precedidos (às vezes) por um snapshot do estado depois de um prefixo dos eventos."""
    events = []
    for epoch in (1, 2, 3):
        if rng.random() < 0.3:
            continue
        for driver in ("VER", "NOR", "HAM"):
            for t in rng.sample(range(1_000, 5_000), 25):
                events.append(
                    {"type": "position", "epoch": epoch, "driver": driver, "x": t, "y": -t, "t": t}
                )
            for n in range(1, rng.randint(2, 6)):
                events.append(
                    {
                        "type": "lap_completed",
                        "epoch": epoch,
                        "driver": driver,
                        "lap": n,
                        "t": 1_000 * n + rng.randint(0, 400),
                        "lap_time_ms": rng.choice([None, 80_000 + n]),
                        "compound": rng.choice([None, "SOFT", "HARD"]),
                        "position": rng.choice([None, rng.randint(1, 20)]),
                    }
                )
        for _ in range(rng.randint(0, 4)):  # resultados de estratégia, de voltas e versões variadas
            events.append(
                {
                    "type": "strategy",
                    "epoch": epoch,
                    "v": rng.choice([1, 1, 1, 2]),  # a 2 é uma versão que o cliente não conhece
                    "lap": rng.randint(5, 30),
                    "reliability": rng.choice(["indicative", "experimental"]),
                }
            )
        for lap_number, state in [(1, "started"), (2, "running"), (3, "running"), (3, "finished")]:
            events.append(
                {
                    "type": "race_status",
                    "epoch": epoch,
                    "state": state,
                    "lap": lap_number,
                    "t": lap_number * 1_000,
                    "total_laps": rng.choice([None, 69]),
                }
            )
    events += rng.sample(events, k=min(40, len(events)))  # repetições (o catch-up repete)
    rng.shuffle(events)
    snapshot = None
    if rng.random() < 0.6 and events:
        prefix = events[: rng.randint(0, len(events) // 2)]
        seed_view = RaceView()
        for event in prefix:
            seed_view.apply_event(event)
        snapshot = {"epoch": seed_view.epoch, "race": seed_view.race, "cars": seed_view.cars}
    return {"snapshot": snapshot, "events": events}


@pytest.mark.skipif(NODE is None, reason="node não está instalado")
def test_the_page_and_the_python_client_reach_the_same_state_for_any_event_order():
    """Roda 60 cenários aleatórios no RaceView do navegador (JS de verdade, no node) e no cliente
    Python de referência e exige o mesmo estado final, inclusive a classificação."""
    assert NODE is not None
    rng = random.Random(2024)
    scenarios = [random_scenario(rng) for _ in range(60)]

    run = subprocess.run(
        [NODE, str(ROOT / "tests" / "frontend" / "apply_scenarios.cjs")],
        input=json.dumps(scenarios),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    for index, (scenario, js) in enumerate(zip(scenarios, json.loads(run.stdout), strict=True)):
        view = RaceView()
        if scenario["snapshot"]:
            view.apply_snapshot(scenario["snapshot"])
        for event in scenario["events"]:
            view.apply_event(event)
        python = {
            "epoch": view.epoch,
            "race": view.race,
            "cars": view.cars,
            "classification": view.classification,
            "strategy": view.strategy,
            "strategyUnsupported": view.strategy_unsupported,
        }
        assert js == python, f"cenário {index}: o JS divergiu do cliente Python"


# ---------------------------------------------------------------- o evento `strategy` na página


def strategy_message(**overrides) -> dict:
    from tests.test_strategy_api import document

    return document(30, **overrides)


def strategy_views(*cases: dict) -> list[dict]:
    assert NODE is not None
    run = subprocess.run(
        [NODE, str(ROOT / "tests" / "frontend" / "strategy_view.cjs")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(run.stdout)


def test_the_page_has_the_strategy_panel_with_both_columns_and_the_honesty_text():
    html = (ROOT / "frontend" / "index.html").read_text()

    assert 'id="strategy"' in html and "Monte Carlo ao lado de" in html
    assert "Vence (Monte Carlo)" in html and "Vence (líder atual)" in html
    assert "RaceView.strategyView(" in html  # a visão é a testada no node, não uma cópia


@pytest.mark.skipif(NODE is None, reason="node não está instalado")
def test_the_view_always_says_the_monte_carlo_did_not_beat_the_current_leader():
    indicative, experimental = strategy_views(
        {"strategy": strategy_message(reliability="indicative")},
        {"strategy": strategy_message(reliability="experimental")},
    )

    for view in (indicative, experimental):
        assert "NÃO superou" in view["honesty"] and "líder atual vence" in view["honesty"]
        assert "docs/backtest.md" in view["honesty"]
    assert (
        "EXPERIMENTAL" in experimental["reliabilityText"]
        and "PIOR" in experimental["reliabilityText"]
    )
    assert "não se distinguiu" in indicative["reliabilityText"]
    assert "EXPERIMENTAL" not in indicative["reliabilityText"]


@pytest.mark.skipif(NODE is None, reason="node não está instalado")
def test_the_view_puts_the_baseline_beside_the_forecast_and_marks_the_leader():
    (view,) = strategy_views({"strategy": strategy_message()})

    assert view["lapLine"] == "volta 30/66, faltam 36"
    assert view["rows"][0] == {
        "driver": "VER", "mc": "62.0%", "baseline": "100.0%", "podium": "97.0%", "leader": True,
    }  # fmt: skip
    assert view["rows"][1]["baseline"] == "0.0%" and view["rows"][1]["leader"] is False


@pytest.mark.skipif(NODE is None, reason="node não está instalado")
def test_the_view_lists_alerts_with_the_selection_bias_note_and_the_track_warnings():
    alert = {
        "attacker": "NOR", "rival": "VER", "probability": 0.72, "margin_s": 1.34,
        "margin_sd_s": 1.2, "horizon_k": 5, "pit_loss_s": 21.0,
        "attacker_compounds": ["MEDIUM", "HARD"], "rival_compounds": ["MEDIUM", "HARD"],
        "blocked_by": None,
    }  # fmt: skip
    warning = {"code": "virtual_safety_car", "text": "VSC em curso: fora do simulador."}
    (view,) = strategy_views({"strategy": strategy_message(alerts=[alert], warnings=[warning])})

    assert view["alerts"] == ["NOR sobre VER: p=72.0%, margem +1.3 s (K=5)"]
    assert "otimista" in view["alertNote"] and "só nas paradas que aconteceram" in view["alertNote"]
    assert view["warnings"] == ["VSC em curso: fora do simulador."]
    assert view["inputs"] == "carros que reportaram: 20/20; voltas atrasadas: 0"


@pytest.mark.skipif(NODE is None, reason="node não está instalado")
def test_the_view_says_what_is_missing_instead_of_showing_nothing():
    nothing, newer_version, newer_but_one_known = strategy_views(
        {"strategy": None},
        {"strategy": None, "unsupported": 2},
        {"strategy": strategy_message(), "unsupported": 2},
    )

    assert "sem resultado" in nothing["empty"]
    assert "versão 2" in newer_version["empty"] and "não suportada" in newer_version["empty"]
    assert "empty" not in newer_but_one_known and newer_but_one_known["unsupported"] is True


def run_page(messages: list[dict]) -> dict:
    assert NODE is not None
    snapshot = {
        "type": "snapshot",
        "as_of": "0-0",
        "epoch": 3,
        "race": {},
        "cars": {},
        "replay": None,
    }
    run = subprocess.run(
        [NODE, str(ROOT / "tests" / "frontend" / "page_smoke.cjs")],
        input=json.dumps({"snapshot": snapshot, "messages": messages}),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(run.stdout)


@pytest.mark.skipif(NODE is None, reason="node não está instalado")
def test_the_page_script_runs_and_draws_the_strategy_panel_with_a_fake_dom():
    screen = run_page([strategy_message(reliability="experimental")])

    assert screen["empty"] == "" and screen["bodyHidden"] is False
    assert "NÃO superou" in screen["honesty"]
    assert screen["reliabilityClass"] == "reliability experimental"
    assert screen["lap"] == "volta 30/66, faltam 36"
    assert screen["rows"][0] == ["VER", "62.0%", "100.0%", "97.0%"]
    assert screen["leaderRows"][0] == "leader" and screen["alerts"] == ["nenhum alerta ligado"]
    assert screen["warnings"] == ["pista sem avisos"] and screen["unsupportedHidden"] is True


@pytest.mark.skipif(NODE is None, reason="node não está instalado")
def test_the_page_keeps_the_empty_state_until_a_result_arrives_and_flags_an_unknown_version():
    nothing = run_page([])
    newer = run_page([strategy_message(v=2)])

    assert "sem resultado" in nothing["empty"] and nothing["bodyHidden"] is True
    assert "versão 2" in newer["empty"] and newer["bodyHidden"] is True
