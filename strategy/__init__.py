"""Estratégia de corrida (Fase 3): degradação de pneu, undercut e simulação Monte Carlo.

Tudo aqui é cálculo PURO sobre voltas já carregadas (numpy, sem banco, sem Redis, sem relógio), de
propósito: é o que torna cada peça testável com dados sintéticos de resposta conhecida, e o que
permite ao backtest e ao consumidor em tempo real chamarem exatamente o mesmo código.

A única parte que fala com Redis e roda como processo está em `strategy/service/` (o consumidor de
estratégia): a fronteira do pacote é a fronteira entre o que é puro e o que tem efeitos.
"""
