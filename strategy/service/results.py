"""Onde os resultados de estratégia moram no Redis: um hash por corrida e um stream de eventos.

DUAS ESTRUTURAS, CADA UMA PARA UMA PERGUNTA
* HASH `strategy:result:{y}:{r}`: campos `epoch`, `latest` e `lap:{N}` (o JSON da volta N). É o
  que o `GET /races/{y}/{r}/strategy?lap=N` LÊ (só lê: nunca dispara cálculo) e o que o WebSocket
  manda a quem entra no meio. Uma volta qualquer da corrida atual, ou a mais recente.
* STREAM `strategy:events:{y}:{r}`: uma entrada por resultado novo. É o que o hub lê (ao lado do
  stream da corrida) para empurrar o evento `strategy` aos clientes. Aparado (MAXLEN): é um buffer
  de entrega, o histórico é o hash.

UMA ESCRITA, DOIS EFEITOS, ATOMICAMENTE (`_PUT`, em Lua). O hash e o stream mudam juntos: nenhum
cliente vê um evento cujo resultado ainda não está no hash, e o GET nunca mostra uma volta que o
WebSocket não anunciou. O mesmo script faz as duas travas do consumidor:

* TRAVA DE EPOCH (a mesma ideia de replay/state.py): o resultado carrega o epoch do replay de que
  veio. Epoch menor que o do hash = resultado de um replay antigo (ou de um consumidor zumbi que
  perdeu o lock e ainda tinha uma simulação em voo): recusado. Epoch maior = replay novo: o hash é
  apagado e recomeça.
* PRIMEIRO A CHEGAR VENCE por volta: se `lap:{N}` já existe neste epoch, nada é escrito nem
  publicado. Com o consumidor em standby assumindo no meio de uma volta, ou um zumbi terminando um
  cálculo que o novo líder já fez, o cliente não recebe a mesma volta duas vezes. O cálculo é
  determinístico (a seed é (ano, etapa, volta)), então "o primeiro" e "o segundo" teriam os mesmos
  números de qualquer forma.

O GET ESCONDE O QUE É DE REPLAY ANTIGO: `get` compara o epoch do hash com o do replay atual
(`replay:epoch:{y}:{r}`). Um replay que acabou de começar e ainda não calculou nada não mostra a
estratégia da corrida passada como se fosse a de agora.
"""

from dataclasses import dataclass
from enum import StrEnum

from redis.asyncio import Redis

from replay.state import epoch_key

RESULTS_TTL_S = 24 * 3600  # um dia depois da última volta calculada
EVENTS_MAXLEN = 2_000  # ~35 corridas de 57 voltas: um buffer de entrega, não o histórico


def result_key(year: int, round_number: int) -> str:
    return f"strategy:result:{year}:{round_number}"


def events_key(year: int, round_number: int) -> str:
    return f"strategy:events:{year}:{round_number}"


# KEYS[1] = hash | KEYS[2] = stream | ARGV: epoch, volta, JSON, maxlen, ttl (s), versão do formato.
# Devolve 1 (gravou e publicou), 2 (a volta já existia neste epoch) ou 0 (epoch antigo).
_PUT = """
local epoch = tonumber(ARGV[1])
local current = tonumber(redis.call('HGET', KEYS[1], 'epoch')) or 0
if epoch < current then
    return 0
end
if epoch > current then
    redis.call('DEL', KEYS[1])
    redis.call('HSET', KEYS[1], 'epoch', epoch)
end
local field = 'lap:' .. ARGV[2]
if redis.call('HEXISTS', KEYS[1], field) == 1 then
    return 2
end
redis.call('HSET', KEYS[1], field, ARGV[3])
local latest = tonumber(redis.call('HGET', KEYS[1], 'latest')) or 0
if tonumber(ARGV[2]) > latest then
    redis.call('HSET', KEYS[1], 'latest', ARGV[2])
end
redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[4], '*',
           'v', ARGV[6], 'epoch', ARGV[1], 'lap', ARGV[2], 'data', ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[5])
return 1
"""


class PutOutcome(StrEnum):
    WRITTEN = "written"
    EXISTS = "exists"  # a volta já tinha resultado neste epoch: o primeiro vence
    FENCED = "fenced"  # epoch antigo: de um replay que já passou (ou de um consumidor zumbi)


@dataclass(frozen=True, slots=True)
class StoredResult:
    epoch: int
    lap: int
    payload: str  # o JSON como o consumidor o gravou
    latest: int  # a volta mais recente com resultado neste epoch


class ResultStore:
    def __init__(
        self,
        redis: Redis,
        *,
        ttl_s: int = RESULTS_TTL_S,
        events_maxlen: int = EVENTS_MAXLEN,
    ) -> None:
        self._redis = redis
        self._ttl_s = ttl_s
        self._events_maxlen = events_maxlen
        self._put = redis.register_script(_PUT)

    async def put(
        self, year: int, round_number: int, *, epoch: int, lap: int, payload: str, version: int
    ) -> PutOutcome:
        outcome = await self._put(
            keys=[result_key(year, round_number), events_key(year, round_number)],
            args=[epoch, lap, payload, self._events_maxlen, self._ttl_s, version],
        )
        return {1: PutOutcome.WRITTEN, 2: PutOutcome.EXISTS}.get(int(outcome), PutOutcome.FENCED)

    async def _current(self, year: int, round_number: int) -> tuple[int, int, int | None]:
        """(epoch do replay atual, epoch do hash, volta mais recente), numa leitura consistente."""
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.get(epoch_key(year, round_number))
            pipe.hmget(result_key(year, round_number), "epoch", "latest")
            replay_epoch, (stored_epoch, latest) = await pipe.execute()
        return (
            int(replay_epoch or 0),
            int(stored_epoch or 0),
            None if latest is None else int(latest),
        )

    async def get(
        self, year: int, round_number: int, lap: int | None = None
    ) -> StoredResult | None:
        """O resultado da volta `lap` (ou o mais recente) do replay ATUAL; None se não há."""
        replay_epoch, stored_epoch, latest = await self._current(year, round_number)
        if latest is None or stored_epoch < replay_epoch:
            return None  # nada ainda, ou só o da corrida passada
        wanted = latest if lap is None else lap
        payload = await self._redis.hget(result_key(year, round_number), f"lap:{wanted}")
        return None if payload is None else StoredResult(stored_epoch, wanted, payload, latest)

    async def has(self, year: int, round_number: int, *, epoch: int, lap: int) -> bool:
        """A volta já tem resultado neste epoch (o consumidor que assume não a refaz)."""
        key = result_key(year, round_number)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hget(key, "epoch")
            pipe.hexists(key, f"lap:{lap}")
            stored_epoch, exists = await pipe.execute()
        return int(stored_epoch or 0) == epoch and bool(exists)
