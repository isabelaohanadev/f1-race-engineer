"""Worker consumidor: lê o stream da corrida por consumer group e mantém o estado no Redis.

Roda como processo separado (`python -m worker`), em quantas cópias se quiser: todas no MESMO grupo
(`state-workers`), então o Redis reparte as entradas entre elas.
"""
