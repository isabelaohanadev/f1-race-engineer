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
        }
        assert js == python, f"cenário {index}: o JS divergiu do cliente Python"
