import pytest
from fastapi.testclient import TestClient

LAPS_URL = "/races/2024/21/laps"


def test_returns_all_laps_ordered_by_driver_then_lap(client: TestClient):
    response = client.get(LAPS_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["driver"] is None
    # NOR vem antes de VER (ordem alfabética do código); dentro de cada piloto, por volta.
    assert [(lap["driver"], lap["lap_number"]) for lap in body["laps"]] == [
        ("NOR", 1),
        ("NOR", 2),
        ("VER", 1),
        ("VER", 2),
        ("VER", 3),
    ]


def test_response_includes_race_metadata(client: TestClient):
    race = client.get(LAPS_URL).json()["race"]

    assert race["year"] == 2024
    assert race["round"] == 21
    assert race["name"] == "São Paulo Grand Prix"
    assert race["country"] == "Brazil"
    assert race["total_laps"] == 69


def test_lap_fields_are_exposed_and_null_lap_time_is_null(client: TestClient):
    laps = client.get(LAPS_URL, params={"driver": "VER"}).json()["laps"]

    assert laps[0]["lap_time_ms"] is None  # volta sem tempo válido
    assert laps[1] == {
        "driver": "VER",
        "lap_number": 2,
        "lap_time_ms": 87134,
        "compound": "INTERMEDIATE",
        "tyre_life": 2,
        "stint": 1,
        "position": 10,
        "completed_at": laps[1]["completed_at"],  # formato do timestamp: ver teste de integração
    }
    assert laps[2]["compound"] == "WET"
    assert laps[2]["stint"] == 2
    assert laps[1]["completed_at"].startswith("2024-11-03T15:50:00")


@pytest.mark.parametrize("code", ["VER", "ver", "Ver"])
def test_driver_filter_returns_only_that_driver_case_insensitive(client: TestClient, code: str):
    body = client.get(LAPS_URL, params={"driver": code}).json()

    assert body["driver"] == "VER"  # o filtro aplicado vem normalizado
    assert [lap["lap_number"] for lap in body["laps"]] == [1, 2, 3]
    assert {lap["driver"] for lap in body["laps"]} == {"VER"}


def test_driver_filter_does_not_leak_laps_from_other_races(client: TestClient):
    laps = client.get(LAPS_URL, params={"driver": "VER"}).json()["laps"]

    assert 99999 not in [lap["lap_time_ms"] for lap in laps]  # a volta do VER em 2023/1


def test_driver_who_did_not_start_gets_empty_list_not_404(client: TestClient):
    """ALB participou da corrida (consta nas inscrições) mas não completou nenhuma volta.
    Isso é um fato válido (200, lista vazia), diferente de um código que não existe na corrida."""
    response = client.get(LAPS_URL, params={"driver": "ALB"})

    assert response.status_code == 200
    assert response.json()["laps"] == []
    assert response.json()["driver"] == "ALB"


def test_driver_not_in_race_returns_404_with_clear_message(client: TestClient):
    # HAM existe no banco (correu em outras corridas), mas não nesta.
    response = client.get(LAPS_URL, params={"driver": "HAM"})

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "HAM" in detail
    assert "2024/21" in detail
    assert "ALB, NOR, VER" in detail  # ajuda quem digitou errado a achar o código certo


def test_completely_unknown_driver_code_returns_404(client: TestClient):
    response = client.get(LAPS_URL, params={"driver": "XYZ"})

    assert response.status_code == 404
    assert "XYZ" in response.json()["detail"]


@pytest.mark.parametrize("bad_code", ["VERSTAPPEN", "VE", "V1R", ""])
def test_malformed_driver_code_is_rejected_by_validation(client: TestClient, bad_code: str):
    # Não é "piloto inexistente": o formato nem é de um código de 3 letras -> 422.
    response = client.get(LAPS_URL, params={"driver": bad_code})

    assert response.status_code == 422


def test_unknown_race_returns_404_with_ingest_hint(empty_client: TestClient):
    response = empty_client.get(LAPS_URL)

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "2024/21" in detail
    assert "python -m ingestion --year 2024 --round 21" in detail


def test_race_not_ingested_is_404_even_if_other_races_exist(client: TestClient):
    assert client.get("/races/2024/22/laps").status_code == 404


@pytest.mark.parametrize("url", ["/races/1900/1/laps", "/races/2024/0/laps", "/races/abc/1/laps"])
def test_invalid_path_parameters_are_rejected_by_validation(client: TestClient, url: str):
    assert client.get(url).status_code == 422
