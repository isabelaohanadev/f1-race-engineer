"""O docker-compose.yml: serviço com várias cópias não pode disputar porta do host nem nome de
container. É uma conferência da CONFIGURAÇÃO (o YAML lido e interpretado): não há Docker neste
ambiente, então o `docker compose up` em si nunca foi executado (ver docs/estrategia-ao-vivo.md)."""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]

# Os serviços que sobem em várias cópias: `strategy` por `deploy.replicas`, `worker` por
# `docker compose up --scale worker=N` (está no README).
SCALABLE = ["strategy", "worker"]


def default_of(text: str) -> str:
    """`${APP_PORT:-8000}` -> `8000` (o valor que vale quando a variável não está definida)."""
    return re.sub(r"\$\{[^}:]+:-([^}]*)\}", r"\1", text)


def split_port(entry: str | int) -> tuple[str, str, str]:
    """(ip, porta do host, porta do container) de uma entrada de `ports`; vazio = não definido.
    Formatos: `9109`, `8000:8000`, `127.0.0.1:6379:6379`, `127.0.0.1::9109` (porta sorteada)."""
    parts = default_of(str(entry)).split(":")
    if len(parts) == 1:
        return "", "", parts[0]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    return parts[0], parts[1], parts[2]


def host_ports(service: dict) -> list[tuple[str, str, str]]:
    return [split_port(p) for p in service.get("ports", [])]


def test_the_strategy_service_runs_two_copies_and_the_worker_is_scalable():
    assert SERVICES["strategy"]["deploy"]["replicas"] == 2
    assert "worker" in SERVICES  # `--scale worker=N` está documentado no README


@pytest.mark.parametrize("name", SCALABLE)
def test_a_scalable_service_does_not_pin_a_host_port_so_copies_never_collide(name: str):
    ports = host_ports(SERVICES[name])

    assert ports, f"{name} publica a porta do /metrics?"
    for ip, host_port, container_port in ports:
        # `127.0.0.1::9109`: o Docker sorteia a porta do host, uma por cópia. Uma porta fixa
        # (`9109:9109`) faria a segunda cópia falhar ao subir.
        assert host_port == "", f"{name}: porta fixa {host_port} no host colide com 2 cópias"
        assert ip == "127.0.0.1", f"{name}: a porta de métricas só em loopback"
        assert container_port.isdigit()


@pytest.mark.parametrize("name", SCALABLE)
def test_a_scalable_service_has_no_container_name(name: str):
    assert "container_name" not in SERVICES[name]  # um nome fixo só admite UMA cópia


def test_the_metrics_port_published_is_the_one_the_process_listens_on():
    expected = {
        "strategy": ("STRATEGY_METRICS_PORT", "9109"),
        "worker": ("WORKER_METRICS_PORT", "9108"),
    }
    for name, (variable, port) in expected.items():
        environment = SERVICES[name]["environment"]
        assert environment[variable] == port
        assert host_ports(SERVICES[name])[0][2] == port


def test_no_two_services_publish_the_same_fixed_host_port():
    fixed = [
        (name, host_port)
        for name, service in SERVICES.items()
        for _, host_port, _ in host_ports(service)
        if host_port
    ]

    ports = [port for _, port in fixed]
    assert len(ports) == len(set(ports)), fixed


def test_the_two_strategy_copies_listen_on_their_own_network_namespace_port_without_a_conflict():
    """Dentro do Compose cada cópia é um container com a SUA rede: as duas escutam a 9109 sem
    conflito, e `strategy:9109` (DNS do serviço) devolve as duas para o Prometheus."""
    strategy = SERVICES["strategy"]

    assert "network_mode" not in strategy  # `host` faria as duas cópias dividirem as portas
    assert strategy["environment"]["STRATEGY_METRICS_PORT"] == "9109"
    assert strategy["command"] == ["python", "-m", "strategy.service"]
    assert strategy["depends_on"]["redis"]["condition"] == "service_healthy"
    assert strategy["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
