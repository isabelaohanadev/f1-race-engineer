"""O consumidor de estratégia: a parte da Fase 3 que fala com Redis e roda como processo.

Fica separado do resto de `strategy/` (que é cálculo puro) de propósito. Aqui estão o armazém dos
resultados (`results.py`), as constantes lidas do banco (`constants.py`) e o laço do consumidor com
o lock e o standby (`consumer.py`, `python -m strategy.service`). A lógica que decide O QUE calcular
e QUANDO (dedupe, janela, voltas atrasadas) está em `strategy/live.py`, pura e testada sem Redis.
"""
