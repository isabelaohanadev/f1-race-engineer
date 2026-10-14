"""Plano de controle do replay ENTRE PROCESSOS: status visível a todos e comandos para o dono.

O PROBLEMA
O motor roda no processo que atendeu o `start`. Com 2 instâncias da API, um `POST /pause` pode cair
na OUTRA, que não tem o motor na memória. O controle e o status precisam passar pelo Redis, o ponto
em comum.

O DESENHO (duas peças, cada uma no tipo de estrutura que combina com ela)
* STATUS = hash `replay:status:{y}:{r}`, escrito pelo DONO a cada ~1 s e a cada comando, com TTL de
  10 s. Qualquer instância lê (GET /replay/...). O TTL é a detecção de falha: se o dono morre, o
  hash expira sozinho e o replay deixa de aparecer como ativo (o lock expira junto, ver lock.py).
  Estados terminais (finished/stopped/failed) ficam 1 h, para consultar o desfecho.
* COMANDOS = stream `replay:ctl:{y}:{r}` (XADD pela instância que recebeu o POST; o dono lê com
  XREAD BLOCK). Um stream, e não Pub/Sub, pelo mesmo motivo do stream de eventos: se o dono estiver
  ocupado ou reconectando no instante do comando, o comando espera no log em vez de se perder. O
  dono começa a ler do "0-0" de um stream recém-apagado, então não há janela entre o `start`
  devolver e o dono passar a escutar.
* RESPOSTA = lista `replay:reply:{y}:{r}:{id}`: o dono faz LPUSH do resultado e quem mandou faz
  BLPOP (bloqueia até chegar, ou 2 s). É um RPC sobre o Redis: o POST devolve o status DEPOIS de o
  comando ter surtido efeito, em vez de um "aceito" que o cliente teria de conferir com polling.

Os comandos são idempotentes por construção (pause de quem já está pausado não muda nada), então o
pior caso de uma entrega duplicada é inofensivo.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from redis.asyncio import Redis

from replay.engine import ACTIVE_STATES, ReplayState, ReplayStatus
from replay.errors import ReplayNotRunning, ReplayTimeout

logger = logging.getLogger(__name__)

# O status expira se o dono parar de renová-lo (a renovação é a cada ~1 s: 10 s tolera 9 falhas).
STATUS_TTL_MS = 10_000
FINISHED_STATUS_TTL_MS = 3_600_000
REPLY_TTL_S = 30
COMMAND_TIMEOUT_S = 2.0  # quanto o POST espera o dono confirmar
CONTROL_MAXLEN = 1_000  # comandos são raros; só limita o pior caso (alguém martelando a API)

_TERMINAL = frozenset(ReplayState) - ACTIVE_STATES


def status_key(year: int, round_number: int) -> str:
    return f"replay:status:{year}:{round_number}"


def control_key(year: int, round_number: int) -> str:
    return f"replay:ctl:{year}:{round_number}"


def reply_key(year: int, round_number: int, command_id: str) -> str:
    return f"replay:reply:{year}:{round_number}:{command_id}"


@dataclass(frozen=True, slots=True)
class Command:
    id: str
    name: str  # pause | resume | speed | stop (o dono rejeita o que não conhece)
    arg: float | None = None


@dataclass(frozen=True, slots=True)
class ReplayInfo:
    """O status como as OUTRAS instâncias o enxergam: o status do motor + quem é o dono."""

    status: ReplayStatus
    epoch: int
    owner: str
    updated_at_ms: int  # relógio do DONO quando publicou (só informativo; a validade é o TTL)

    @property
    def active(self) -> bool:
        return self.status.state in ACTIVE_STATES


@dataclass(frozen=True, slots=True)
class CommandResult:
    ok: bool
    info: ReplayInfo | None = None
    error: str | None = None  # código curto: "invalid_speed", "unknown_command"
    detail: str | None = None


# ---------------------------------------------------------------- (de)serialização


def info_to_fields(info: ReplayInfo) -> dict[str, str]:
    s = info.status
    fields = {
        "state": s.state.value,
        "speed": str(s.speed),
        "events_published": str(s.events_published),
        "epoch": str(info.epoch),
        "owner": info.owner,
        "updated_at_ms": str(info.updated_at_ms),
    }
    for name in ("race_time_ms", "lap", "total_laps", "error"):
        value = getattr(s, name)
        if value is not None:  # ausente = nulo
            fields[name] = str(value)
    return fields


def _opt_int(fields: dict[str, str], name: str) -> int | None:
    return int(fields[name]) if name in fields else None


def info_from_fields(fields: dict[str, str]) -> ReplayInfo:
    return ReplayInfo(
        status=ReplayStatus(
            state=ReplayState(fields["state"]),
            speed=float(fields["speed"]),
            race_time_ms=_opt_int(fields, "race_time_ms"),
            lap=_opt_int(fields, "lap"),
            total_laps=_opt_int(fields, "total_laps"),
            events_published=int(fields["events_published"]),
            error=fields.get("error"),
        ),
        epoch=int(fields["epoch"]),
        owner=fields["owner"],
        updated_at_ms=int(fields["updated_at_ms"]),
    )


def result_to_json(result: CommandResult) -> str:
    return json.dumps(
        {
            "ok": result.ok,
            "error": result.error,
            "detail": result.detail,
            "info": info_to_fields(result.info) if result.info else None,
        }
    )


def result_from_json(payload: str) -> CommandResult:
    data: dict[str, Any] = json.loads(payload)
    info = info_from_fields(data["info"]) if data.get("info") else None
    return CommandResult(data["ok"], info, data.get("error"), data.get("detail"))


# ---------------------------------------------------------------- o plano de controle


class ControlPlane(Protocol):
    """O que o DONO do replay (o gerenciador) usa. A outra metade, `get_info` e `send_command`, é
    de quem consulta/comanda e só existe na implementação real."""

    async def reset(self, year: int, round_number: int) -> None: ...

    async def publish_status(
        self, year: int, round_number: int, status: ReplayStatus, *, epoch: int, owner: str
    ) -> None: ...

    async def read_commands(
        self, year: int, round_number: int, after_id: str, block_ms: int
    ) -> tuple[list[Command], str]: ...

    async def reply(
        self, year: int, round_number: int, command_id: str, result: CommandResult
    ) -> None: ...


class RedisControlPlane:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------ lado do DONO

    async def reset(self, year: int, round_number: int) -> None:
        """Começa com a fila de comandos vazia: os de um replay anterior não valem para este."""
        await self._redis.delete(control_key(year, round_number))

    async def publish_status(
        self, year: int, round_number: int, status: ReplayStatus, *, epoch: int, owner: str
    ) -> None:
        info = ReplayInfo(status, epoch, owner, int(time.time() * 1000))
        ttl = FINISHED_STATUS_TTL_MS if status.state in _TERMINAL else STATUS_TTL_MS
        key = status_key(year, round_number)
        # DEL + HSET + PEXPIRE numa transação: SUBSTITUI o status (um campo que sumiu, como o
        # `error`, não pode sobrar do anterior) e ninguém o enxerga pela metade.
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            pipe.hset(key, mapping=info_to_fields(info))
            pipe.pexpire(key, ttl)
            await pipe.execute()

    async def read_commands(
        self, year: int, round_number: int, after_id: str, block_ms: int
    ) -> tuple[list[Command], str]:
        """Comandos novos desde `after_id` (bloqueia até `block_ms`). Devolve também o último ID
        lido, mesmo que algum comando venha ilegível, para o dono nunca reler o mesmo."""
        reply = await self._redis.xread(
            {control_key(year, round_number): after_id}, count=10, block=block_ms
        )
        commands: list[Command] = []
        last_id = after_id
        for _stream, entries in reply or []:
            for entry_id, fields in entries:
                last_id = entry_id
                try:
                    arg = float(fields["arg"]) if "arg" in fields else None
                except ValueError:
                    arg = None
                commands.append(Command(entry_id, fields.get("cmd", ""), arg))
        return commands, last_id

    async def reply(
        self, year: int, round_number: int, command_id: str, result: CommandResult
    ) -> None:
        key = reply_key(year, round_number, command_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.lpush(key, result_to_json(result))
            pipe.expire(key, REPLY_TTL_S)  # se ninguém buscar a resposta, ela some sozinha
            await pipe.execute()

    # ------------------------------------------------------------ lado de QUEM CONSULTA/COMANDA

    async def get_info(self, year: int, round_number: int) -> ReplayInfo | None:
        fields = await self._redis.hgetall(status_key(year, round_number))
        return info_from_fields(fields) if fields else None

    async def send_command(
        self,
        year: int,
        round_number: int,
        name: str,
        arg: float | None = None,
        *,
        timeout_s: float = COMMAND_TIMEOUT_S,
    ) -> CommandResult:
        """Manda o comando ao dono (esteja em qual processo estiver) e espera a confirmação.

        ReplayNotRunning: não há replay ativo (nunca houve, terminou, ou o dono morreu e o status
        expirou). ReplayTimeout: há status ativo, mas o dono não respondeu em `timeout_s`.
        """
        info = await self.get_info(year, round_number)
        if info is None or not info.active:
            raise ReplayNotRunning(f"não há replay ativo para a corrida {year}/{round_number}")

        fields = {"cmd": name}
        if arg is not None:
            fields["arg"] = str(arg)
        command_id = await self._redis.xadd(
            control_key(year, round_number), fields, maxlen=CONTROL_MAXLEN, approximate=True
        )
        # BLPOP: bloqueia até o dono fazer LPUSH da resposta. `timeout_s` < socket_timeout do
        # cliente (app/redis_client.py), como todo BLOCK.
        reply = await self._redis.blpop(
            [reply_key(year, round_number, command_id)], timeout=timeout_s
        )
        if reply is None:
            info = await self.get_info(year, round_number)
            if info is None or not info.active:  # terminou (ou morreu) enquanto esperávamos
                raise ReplayNotRunning(
                    f"o replay {year}/{round_number} terminou antes de responder"
                )
            raise ReplayTimeout(f"o dono do replay {year}/{round_number} não confirmou o comando")
        return result_from_json(reply[1])
