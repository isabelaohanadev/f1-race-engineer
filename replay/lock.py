"""Lock distribuído: um único replay por corrida, mesmo com vários processos da API.

Dois replays da mesma corrida escreveriam no MESMO stream: eventos duplicados e intercalados, com o
tempo "voltando". Um dicionário em memória só protege dentro de UM processo; com 2 réplicas da API,
dois `POST /start` simultâneos poderiam cair em processos diferentes. O Redis é o ponto em comum.

Padrão (o "lock simples" da documentação do Redis, sem Redlock: temos UM Redis, não um cluster):
* adquirir:  SET chave <token> NX PX <ttl>   (NX = só se não existir; token único = "quem sou eu")
* renovar:   só se o token ainda for o MEU (script Lua: GET + PEXPIRE atômicos)
* liberar:   só se o token ainda for o MEU (Lua: GET + DEL atômicos)

O TTL é a rede de segurança: se o processo morrer sem liberar, o lock expira sozinho e o replay pode
ser iniciado de novo. Por isso quem segura o lock precisa RENOVAR (heartbeat, ver manager.py). O
token evita o bug clássico de apagar o lock de OUTRO (o meu expirou e outro o adquiriu).
Limitação conhecida: se o processo ficar CONGELADO por mais que o TTL (pausa de GC, VM suspensa),
ele pode publicar mais alguns eventos antes de perceber que perdeu o lock; um "fencing token"
resolveria, e não vale a complexidade para um replay.
"""

import uuid

from redis.asyncio import Redis

_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

DEFAULT_TTL_MS = 10_000


def lock_key(year: int, round_number: int) -> str:
    return f"replay:lock:{year}:{round_number}"


class ReplayLock:
    def __init__(self, redis: Redis, key: str, ttl_ms: int = DEFAULT_TTL_MS) -> None:
        self._redis = redis
        self._key = key
        self._ttl_ms = ttl_ms
        self._token = uuid.uuid4().hex
        self._renew = redis.register_script(_RENEW)
        self._release = redis.register_script(_RELEASE)

    @property
    def ttl_ms(self) -> int:
        return self._ttl_ms

    async def acquire(self) -> bool:
        return bool(await self._redis.set(self._key, self._token, nx=True, px=self._ttl_ms))

    async def renew(self) -> bool:
        """`False` = o lock não é mais meu (expirou e/ou outro o pegou): quem o usava deve parar."""
        return bool(await self._renew(keys=[self._key], args=[self._token, self._ttl_ms]))

    async def release(self) -> None:
        await self._release(keys=[self._key], args=[self._token])
