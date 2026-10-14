"""A régua de latência compartilhada pela API (entrega ao cliente) e pelo worker (estado aplicado).

O instante de PUBLICAÇÃO de um evento é a parte de tempo do ID da entrada do stream, atribuída pelo
Redis no XADD: não precisamos gravar relógio nenhum no evento. Consequência: a latência medida
mistura o relógio do Redis com o de quem mede. Na mesma máquina (o Compose) é exata; com Redis em
outra máquina inclui o desvio entre os relógios (por isso nunca é negativa: ver `seconds_since`).
"""

import time

# De 1 ms a 1 s, em passos de ~2,5x: a latência normal (poucos ms) cai nos primeiros buckets, e o
# 1 s final distingue "lento" de "muito lento". Acima de 1 s só entra no +Inf (o cliente o põe).
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def seconds_since(published_ms: int) -> float:
    """Segundos desde a publicação, nunca negativo (desvio entre relógios)."""
    return max(time.time() - published_ms / 1000, 0.0)
