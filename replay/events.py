"""Contrato dos eventos que trafegam no Redis Stream de uma corrida.

Cada entrada do stream é um mapa plano de strings (é o que o Redis Streams armazena). Escolhemos
campos planos em vez de um único campo `data` com JSON porque:
* o stream fica legível no `redis-cli XRANGE` (debug sem ferramenta extra);
* o Redis guarda os NOMES dos campos uma vez por bloco quando as entradas têm o mesmo formato, então
  o custo de memória por evento é baixo;
* campos novos são aditivos: consumidor antigo ignora o que não conhece.

O tempo `t` de todo evento é o instante DA CORRIDA em milissegundos desde a época Unix (UTC),
inteiro: sem ambiguidade de fuso, comparação trivial e é o que o motor usa para dar o ritmo.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar, Literal

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MS = timedelta(milliseconds=1)

STATE_STARTED = "started"  # primeira entrada de todo replay: o consumidor deve zerar o estado
STATE_RUNNING = "running"
STATE_FINISHED = "finished"  # última entrada: acabaram os dados da corrida

RaceState = Literal["started", "running", "finished"]


def stream_key(year: int, round_number: int) -> str:
    """Nome do stream de uma corrida: `race:2024:21`."""
    return f"race:{year}:{round_number}"


def to_epoch_ms(moment: datetime) -> int:
    """Instante -> milissegundos Unix, em aritmética inteira (sem erro de float).

    O Postgres sempre devolve `timestamptz` (com fuso). Datetimes sem fuso só aparecem no SQLite dos
    testes unitários; por convenção do projeto todo horário sem fuso é UTC (ver db/models.py).
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - _EPOCH) // _ONE_MS


@dataclass(frozen=True, slots=True)
class PositionEvent:
    """Onde um carro estava (x, y em décimos de metro, unidade do FastF1)."""

    driver: str
    x: int
    y: int
    t: int
    type: ClassVar[str] = "position"

    def to_fields(self) -> dict[str, str]:
        return {
            "type": self.type,
            "driver": self.driver,
            "x": str(self.x),
            "y": str(self.y),
            "t": str(self.t),
        }


@dataclass(frozen=True, slots=True)
class LapCompletedEvent:
    """Um piloto terminou uma volta. `position` é a posição dele na corrida ao fim da volta; é o
    que permite ao consumidor montar a classificação sem consultar o banco."""

    driver: str
    lap: int
    t: int
    lap_time_ms: int | None = None  # nulo em volta sem tempo válido (largada, box, bandeira)
    compound: str | None = None
    position: int | None = None
    type: ClassVar[str] = "lap_completed"

    def to_fields(self) -> dict[str, str]:
        fields = {
            "type": self.type,
            "driver": self.driver,
            "lap": str(self.lap),
            "t": str(self.t),
        }
        # Campo ausente = valor nulo (em vez de "" ou "None": não há como confundir com dado real).
        if self.lap_time_ms is not None:
            fields["lap_time_ms"] = str(self.lap_time_ms)
        if self.compound is not None:
            fields["compound"] = self.compound
        if self.position is not None:
            fields["position"] = str(self.position)
        return fields


@dataclass(frozen=True, slots=True)
class RaceStatusEvent:
    """Estado da corrida: em que volta o LÍDER está. `started` abre o replay; `finished` o fecha."""

    state: RaceState
    lap: int
    t: int
    total_laps: int | None = None
    type: ClassVar[str] = "race_status"

    def to_fields(self) -> dict[str, str]:
        fields = {"type": self.type, "state": self.state, "lap": str(self.lap), "t": str(self.t)}
        if self.total_laps is not None:
            fields["total_laps"] = str(self.total_laps)
        return fields


ReplayEvent = PositionEvent | LapCompletedEvent | RaceStatusEvent


class MalformedEvent(ValueError):
    """Entrada do stream que não é um evento válido (tipo desconhecido ou campo faltando)."""


def _opt_int(fields: Mapping[str, str], name: str) -> int | None:
    value = fields.get(name)
    return None if value is None else int(value)


def parse_event(fields: Mapping[str, str]) -> ReplayEvent:
    """Inverso de `to_fields()`: é o que o worker usa para ler uma entrada do stream."""
    try:
        match fields.get("type"):
            case "position":
                return PositionEvent(
                    driver=fields["driver"],
                    x=int(fields["x"]),
                    y=int(fields["y"]),
                    t=int(fields["t"]),
                )
            case "lap_completed":
                return LapCompletedEvent(
                    driver=fields["driver"],
                    lap=int(fields["lap"]),
                    t=int(fields["t"]),
                    lap_time_ms=_opt_int(fields, "lap_time_ms"),
                    compound=fields.get("compound"),
                    position=_opt_int(fields, "position"),
                )
            case "race_status":
                return RaceStatusEvent(
                    state=fields["state"],  # type: ignore[arg-type]
                    lap=int(fields["lap"]),
                    t=int(fields["t"]),
                    total_laps=_opt_int(fields, "total_laps"),
                )
    except (KeyError, ValueError) as exc:
        raise MalformedEvent(f"evento inválido {dict(fields)!r}: {exc!r}") from exc
    raise MalformedEvent(f"tipo de evento desconhecido: {fields.get('type')!r}")
