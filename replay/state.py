"""O estado ATUAL da corrida, guardado no Redis e compartilhado por workers e instâncias da API.

POR QUE O ESTADO FICA NO REDIS (e não na memória do worker)
O WebSocket precisa de um snapshot que QUALQUER instância da API consiga ler, e vários workers
dividem os eventos (consumer group compartilhado): se cada um guardasse o estado na memória, cada um
teria só uma fatia dos eventos. O estado, portanto, mora num lugar só.

UM HASH POR CORRIDA (`state:{year}:{round}`), campos "CÓDIGO:campo" (VER:x, VER:lap, ...) e
"race:campo" (race:lap, ...). Um hash só, e não um por piloto, porque:
* o snapshot é UM HGETALL, e um comando do Redis é atômico: a foto é consistente entre os 20
  pilotos (com 20 hashes, uma foto poderia pegar o VER "agora" e o NOR "2 s atrás");
* o script Lua toca uma única chave (também é o que o Redis Cluster exige de um script).

TRAVA 1: ORDEM ENTRE WORKERS (last-write-wins por versão, em Lua)
Dois workers tiram eventos do mesmo piloto do stream e os aplicam em qualquer ordem. Se o worker A
aplicar o evento t=5 e depois o B aplicar o t=3 (que pegou antes, mas demorou), um `HSET` simples
deixaria o estado "voltar no tempo". O script `_APPLY` só grava se o tempo do evento for MAIOR que o
já gravado, e faz LER-COMPARAR-GRAVAR numa operação atômica (Redis roda um script inteiro sem
intercalar outro comando). Resultado: o estado final é o mesmo qualquer que seja a ordem em que os
eventos foram processados. Por que Lua e não WATCH/MULTI: o WATCH exige repetir quando há conflito
(e com 20 pilotos x 5 eventos/s há conflito o tempo todo); o script é um único ida-e-volta, sem laço
de repetição. O custo é um script a manter, pequeno (~20 linhas).

Por piloto há DUAS versões independentes: `pos_t` (da posição x,y) e `lap_t` (da última volta). Um
único tempo por piloto descartaria uma posição legítima só porque uma volta mais "nova" já chegou.

A mesma regra vale para REPETIÇÕES: o consumer group entrega "pelo menos uma vez" (uma mensagem
reivindicada por XAUTOCLAIM pode ser aplicada duas vezes). Evento repetido tem versão IGUAL, que não
é maior: o script o ignora. Aplicar duas vezes = aplicar uma, o que torna seguro reentregar.

EPOCH: UM NÚMERO POR REPLAY
Reproduzir a MESMA corrida de novo gera eventos com os MESMOS tempos. Sem proteção, o estado
gravado pelo replay anterior (que chegou ao fim da corrida) "ganharia" de todos os eventos do replay
novo (tempos menores), e um evento atrasado do replay velho ainda em voo travaria o novo. Por isso
todo evento carrega o `epoch` do replay (INCR no início). O script compara o epoch ANTES da versão:
* epoch menor que o do hash  -> evento de replay antigo: ignorado;
* epoch maior que o do hash  -> replay novo: o hash é apagado e recomeça (a própria primeira
  escrita do replay novo faz o reset, sem ninguém precisar coordenar workers).
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.exceptions import ResponseError

from replay.events import (
    LapCompletedEvent,
    PositionEvent,
    RaceStatusEvent,
    StreamEntry,
    id_before,
    stream_key,
)

logger = logging.getLogger(__name__)

# O consumer group compartilhado por TODOS os workers: o Redis reparte as entradas entre os
# consumidores do mesmo grupo (cada entrada vai para UM deles) e lembra até onde o grupo leu.
WORKER_GROUP = "state-workers"

# Conjunto dos streams de corrida que já tiveram replay: é como o worker DESCOBRE o que consumir
# sem ser reconfigurado a cada corrida iniciada pela API.
STREAMS_REGISTRY_KEY = "replay:streams"


def state_key(year: int, round_number: int) -> str:
    return f"state:{year}:{round_number}"


def epoch_key(year: int, round_number: int) -> str:
    return f"replay:epoch:{year}:{round_number}"


# KEYS[1] = hash da corrida | ARGV[1] = epoch do evento | ARGV[2] = prefixo dos campos ("VER:" ou
# "race:") | ARGV[3] = nome do campo de versão | ARGV[4] = versão do evento | ARGV[5..] = pares
# campo, valor ("" remove o campo). Devolve 1 se gravou, 0 se ignorou (evento velho ou repetido).
_APPLY = """
local epoch = tonumber(ARGV[1])
local current_epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch')) or 0
if epoch < current_epoch then
    return 0
end
if epoch > current_epoch then
    redis.call('DEL', KEYS[1])
    redis.call('HSET', KEYS[1], 'epoch', epoch)
end
local version_field = ARGV[2] .. ARGV[3]
local current = tonumber(redis.call('HGET', KEYS[1], version_field))
local version = tonumber(ARGV[4])
if current ~= nil and current >= version then
    return 0
end
redis.call('HSET', KEYS[1], version_field, ARGV[4])
for i = 5, #ARGV, 2 do
    local name = ARGV[2] .. ARGV[i]
    if ARGV[i + 1] == '' then
        redis.call('HDEL', KEYS[1], name)
    else
        redis.call('HSET', KEYS[1], name, ARGV[i + 1])
    end
end
return 1
"""

# `finished` no MESMO milissegundo de um `running` (a última volta termina no último instante) não
# pode ser descartado por empate de versão: a versão do status é t*4 + posição no ciclo de vida.
_STATUS_RANK = {"started": 0, "running": 1, "finished": 2}

RACE_PREFIX = "race:"


@dataclass(frozen=True, slots=True)
class Update:
    """O que um evento escreve no hash: de qual entidade (prefixo), versão e campos."""

    prefix: str
    version_field: str
    version: int
    fields: tuple[tuple[str, str], ...]


def to_update(event: PositionEvent | LapCompletedEvent | RaceStatusEvent) -> Update:
    """Função PURA: traduz um evento no que ele grava. Fica separada do Redis para ser testada
    sem servidor."""
    match event:
        case PositionEvent():
            return Update(
                f"{event.driver}:", "pos_t", event.t, (("x", str(event.x)), ("y", str(event.y)))
            )
        case LapCompletedEvent():
            # Valor vazio = "sem dado": o script REMOVE o campo, para um tempo de volta nulo não
            # deixar o tempo da volta anterior no lugar dele.
            return Update(
                f"{event.driver}:",
                "lap_t",
                event.t,
                (
                    ("lap", str(event.lap)),
                    ("lap_time_ms", "" if event.lap_time_ms is None else str(event.lap_time_ms)),
                    ("compound", event.compound or ""),
                    ("position", "" if event.position is None else str(event.position)),
                ),
            )
        case RaceStatusEvent():
            return Update(
                RACE_PREFIX,
                "status_v",
                event.t * 4 + _STATUS_RANK[event.state],
                (
                    ("state", event.state),
                    ("lap", str(event.lap)),
                    ("total_laps", "" if event.total_laps is None else str(event.total_laps)),
                ),
            )


# ---------------------------------------------------------------- snapshot


_INT_FIELDS = {"x", "y", "pos_t", "lap", "lap_t", "lap_time_ms", "position", "total_laps"}


def _coerce(name: str, value: str) -> int | str:
    return int(value) if name in _INT_FIELDS else value


def classify(cars: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Classificação: mais voltas completas primeiro; empatando em voltas, quem cruzou a linha
    antes (`lap_t` menor). Quem ainda não completou nenhuma volta vai por último, por código."""
    with_laps = [(code, c) for code, c in cars.items() if "lap" in c]
    without = sorted(code for code, c in cars.items() if "lap" not in c)
    with_laps.sort(key=lambda item: (-item[1]["lap"], item[1].get("lap_t", 0), item[0]))
    return [code for code, _ in with_laps] + without


@dataclass(frozen=True, slots=True)
class RaceSnapshot:
    epoch: int  # 0 = nenhum replay ainda
    race: dict[str, int | str] = field(default_factory=dict)  # state, lap, total_laps
    cars: dict[str, dict[str, int | str]] = field(default_factory=dict)
    classification: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "race": self.race,
            "cars": self.cars,
            "classification": self.classification,
        }


def build_snapshot(epoch: int, raw: Mapping[str, str]) -> RaceSnapshot:
    race: dict[str, int | str] = {}
    cars: dict[str, dict[str, int | str]] = {}
    for key, value in raw.items():
        if key == "epoch":
            continue
        prefix, _, name = key.partition(":")
        if prefix == "race":
            if name != "status_v":  # a versão é detalhe interno
                race[name] = _coerce(name, value)
        else:
            cars.setdefault(prefix, {})[name] = _coerce(name, value)
    return RaceSnapshot(epoch, race, cars, classify(cars))


class RaceStateStore:
    """Escrita (workers) e leitura (API) do estado da corrida."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._apply = redis.register_script(_APPLY)

    # ------------------------------------------------------------ ciclo de vida do replay

    async def new_epoch(self, year: int, round_number: int) -> int:
        """Número do replay que começa. INCR é atômico: dois starts nunca recebem o mesmo."""
        return int(await self._redis.incr(epoch_key(year, round_number)))

    async def register_stream(self, year: int, round_number: int) -> None:
        await self._redis.sadd(STREAMS_REGISTRY_KEY, stream_key(year, round_number))  # type: ignore[misc]

    # ------------------------------------------------------------ escrita

    @staticmethod
    def _args(entry: StreamEntry) -> list[str]:
        update = to_update(entry.event)
        args = [str(entry.epoch), update.prefix, update.version_field, str(update.version)]
        for name, value in update.fields:
            args += [name, value]
        return args

    async def queue_apply(
        self, pipe: Pipeline, year: int, round_number: int, entry: StreamEntry
    ) -> None:
        """Enfileira a atualização num pipeline (o worker aplica um lote em um ida-e-volta)."""
        await self._apply(keys=[state_key(year, round_number)], args=self._args(entry), client=pipe)

    async def apply(self, year: int, round_number: int, entry: StreamEntry) -> bool:
        """Aplica um evento. True = gravou; False = ignorado (velho, repetido ou replay antigo)."""
        result = await self._apply(keys=[state_key(year, round_number)], args=self._args(entry))
        return result == 1

    # ------------------------------------------------------------ leitura

    async def snapshot(self, year: int, round_number: int) -> RaceSnapshot:
        """O estado do replay ATUAL. Se o hash é de um replay anterior (o novo ainda não gravou
        nada), o snapshot sai vazio em vez de mostrar a corrida velha como se fosse a de agora."""
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.get(epoch_key(year, round_number))
            pipe.hgetall(state_key(year, round_number))
            current, raw = await pipe.execute()
        current_epoch = int(current or 0)
        state_epoch = int(raw.get("epoch", 0))
        if not raw or state_epoch < current_epoch:
            return RaceSnapshot(epoch=current_epoch)
        return build_snapshot(state_epoch, raw)

    async def applied_watermark(self, year: int, round_number: int) -> str:
        """ID do stream até o qual TODOS os eventos já foram aplicados ao estado (e portanto estão
        no snapshot). É o ponto de partida seguro para quem lê o snapshot e depois segue o stream.

        Por que não "o último ID do stream": o worker aplica DEPOIS da publicação. Se o cliente
        anotasse o fim do stream e lesse o estado, os eventos publicados mas ainda não aplicados
        ficariam num buraco (não estão no snapshot e o XREAD só traz o que vem depois do ID).
        Um evento está aplicado se foi entregue ao grupo e confirmado (XACK é DEPOIS do apply):

        * há pendentes (entregues, sem XACK): marca = logo antes do MENOR pendente (tudo abaixo
          dele foi entregue e confirmado);
        * não há pendentes: marca = o último ID entregue ao grupo.

        A ordem das leituras importa: primeiro `last-delivered-id`, depois os pendentes. O que for
        entregue entre as duas leituras só pode ENTRAR no PEL (e então aparece no mínimo) ou já ter
        sido confirmado; nunca deixa a marca adiante de um evento não aplicado.
        Sem grupo (nenhum worker rodou ainda): "0-0", ou seja, nada foi aplicado.
        """
        stream = stream_key(year, round_number)
        try:
            groups = await self._redis.xinfo_groups(stream)
        except ResponseError:  # o stream não existe
            return "0-0"
        group = next((g for g in groups if g["name"] == WORKER_GROUP), None)
        if group is None:
            return "0-0"
        last_delivered = group["last-delivered-id"]
        pending = await self._redis.xpending(stream, WORKER_GROUP)
        if pending["pending"] > 0:
            return id_before(pending["min"])
        return last_delivered
