"""Estratégia de corrida (Fase 3): degradação de pneu, undercut e simulação Monte Carlo.

Tudo aqui é cálculo PURO sobre voltas já carregadas (numpy, sem banco, sem Redis, sem relógio), de
propósito: é o que torna cada peça testável com dados sintéticos de resposta conhecida, e o que
permite ao backtest e ao consumidor em tempo real chamarem exatamente o mesmo código.
"""
