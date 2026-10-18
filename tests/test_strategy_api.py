"""GET /races/{year}/{round}/strategy: só lê, valida o contrato e diz o que o backtest mediu."""

import json

import pytest
from fastapi.testclient import TestClient

from app.deps import get_strategy_store
from app.main import app
from strategy.service.results import StoredResult


def document(lap: int = 30, **overrides) -> dict:
    drivers = [
        {
            "driver": "VER", "win": 0.62, "podium": 0.97, "win_error": 0.007, "podium_error": 0.003,
            "baseline_win": 1.0, "retire": 0.01, "expected_stops": 1.0, "source": "modelo",
        },
        {
            "driver": "NOR", "win": 0.38, "podium": 0.9, "win_error": 0.007, "podium_error": 0.004,
            "baseline_win": 0.0, "retire": 0.01, "expected_stops": 1.1, "source": "modelo",
        },
    ]  # fmt: skip
    base = {
        "v": 1, "type": "strategy", "year": 2024, "round": 10, "epoch": 3, "lap": lap,
        "total_laps": 66, "laps_left": 66 - lap, "progress": round(lap / 66, 4), "as_of_ms": 123456,
        "reliability": "indicative", "baseline": {"name": "o líder atual vence", "leader": "VER"},
        "forecast": {
            "n_sims": 5000, "variant": "encolhimento", "sc_probability": 0.12,
            "without_model": [], "drivers": drivers,
        },
        "alerts": [], "warnings": [],
        "inputs": {
            "cars_reported": 20, "cars_known": 20, "late_laps": 0, "duplicates": 0,
            "wait_s": 0.3, "constants": "sem 2024 R10 Spanish: 20 corridas (todas as outras)",
        },
        "computed_at_ms": 1_700_000_000_000, "duration_s": 0.8,
    }  # fmt: skip
    return base | overrides


class FakeStore:
    """O armazém só com a LEITURA: se o endpoint tentasse escrever (ou calcular), levantaria."""

    def __init__(self, results: dict[int, str], epoch: int = 3) -> None:
        self.results, self.epoch = results, epoch
        self.reads: list[tuple[int, int, int | None]] = []

    async def get(self, year: int, round_number: int, lap: int | None = None):
        self.reads.append((year, round_number, lap))
        if not self.results:
            return None
        latest = max(self.results)
        wanted = latest if lap is None else lap
        payload = self.results.get(wanted)
        return None if payload is None else StoredResult(self.epoch, wanted, payload, latest)


@pytest.fixture
def store() -> FakeStore:
    fake = FakeStore({29: json.dumps(document(29)), 30: json.dumps(document(30))})
    app.dependency_overrides[get_strategy_store] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_strategy_store, None)


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_without_lap_it_returns_the_latest_result_with_the_baseline_beside_the_forecast(
    client: TestClient, store: FakeStore
):
    response = client.get("/races/2024/10/strategy")

    assert response.status_code == 200
    body = response.json()
    assert (body["lap"], body["laps_left"], body["v"], body["epoch"]) == (30, 36, 1, 3)
    assert body["baseline"] == {"name": "o líder atual vence", "leader": "VER"}
    assert [d["baseline_win"] for d in body["forecast"]["drivers"]] == [1.0, 0.0]
    assert body["reliability"] == "indicative"
    assert store.reads == [(2024, 10, None)]


def test_a_specific_lap_is_returned_as_calculated(client: TestClient, store: FakeStore):
    body = client.get("/races/2024/10/strategy?lap=29").json()

    assert body["lap"] == 29 and body["laps_left"] == 37
    assert store.reads == [(2024, 10, 29)]


def test_the_response_is_what_was_stored_nothing_is_recalculated(
    client: TestClient, store: FakeStore
):
    stored = json.loads(store.results[30])

    assert client.get("/races/2024/10/strategy?lap=30").json() == stored  # o mesmo documento


def test_a_lap_without_a_result_says_so_and_names_the_latest(client: TestClient, store: FakeStore):
    response = client.get("/races/2024/10/strategy?lap=12")

    assert response.status_code == 404
    assert "volta 12" in response.json()["detail"] and "30" in response.json()["detail"]


def test_a_race_without_any_result_explains_where_it_comes_from(client: TestClient):
    fake = FakeStore({})
    app.dependency_overrides[get_strategy_store] = lambda: fake
    try:
        response = client.get("/races/2024/10/strategy")
    finally:
        app.dependency_overrides.pop(get_strategy_store, None)

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "python -m strategy.service" in detail and "volta 5" in detail
    assert fake.reads == [(2024, 10, None)]


@pytest.mark.parametrize("query", ["lap=0", "lap=-3", "lap=abc"])
def test_an_invalid_lap_is_rejected_before_touching_the_store(
    client: TestClient, store: FakeStore, query: str
):
    assert client.get(f"/races/2024/10/strategy?{query}").status_code == 422
    assert store.reads == []


def test_a_result_in_a_format_this_api_does_not_know_is_a_clear_500_not_a_guess(
    client: TestClient,
):
    fake = FakeStore({30: json.dumps(document(30, v=2))})
    app.dependency_overrides[get_strategy_store] = lambda: fake
    try:
        response = client.get("/races/2024/10/strategy")
    finally:
        app.dependency_overrides.pop(get_strategy_store, None)

    assert response.status_code == 500
    assert "não entende" in response.json()["detail"]


def test_the_openapi_documents_the_backtest_caveats_on_the_fields(client: TestClient):
    """A ressalva do viés de seleção dos alertas vive na documentação do campo, não no payload; e a
    do Monte Carlo contra o comparador, no `reliability` e no `forecast`."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    alerts = schemas["StrategyOut"]["properties"]["alerts"]["description"]
    reliability = schemas["StrategyOut"]["properties"]["reliability"]["description"]
    forecast = schemas["StrategyOut"]["properties"]["forecast"]["description"]
    baseline = schemas["StrategyDriverOut"]["properties"]["baseline_win"]["description"]

    assert "VIÉS DE SELEÇÃO" in alerts and "OTIMISTA" in alerts and "revocação" in alerts
    assert "experimental" in reliability and "indicative" in reliability
    assert "PIOR" in reliability and "NÃO se distinguiu" in reliability
    assert "não superou" in forecast and "VSC" in forecast
    assert "SEMPRE" in baseline
    payload_keys = set(document()) | set(document()["forecast"]) | set(document()["inputs"])
    assert not any("viés" in key.lower() for key in payload_keys)  # a ressalva não está no payload
