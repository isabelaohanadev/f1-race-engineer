# Limpeza das voltas: o funil de exclusão

Gerado por `python -m strategy.cleaning --markdown docs/limpeza-das-voltas.md` em 2026-10-14, sobre 24 corridas. A regra está em `strategy/cleaning.py`.

## Resumo

- Ao todo: 26604 voltas, 23014 úteis (87%).
- Tirado por cada regra (na ordem, sem contar duas vezes): desconhecido 0, sem_tempo 223, volta_1 465, box 1483, sem_pneu 0, pista_nao_verde 861, so_amarela 558.
- A amarela, sozinha, custa 558 voltas (2% do total).
- Corridas que perdem mais de 40%: nenhuma.

## Como ler

Cada volta cai na **primeira** regra que a pega, na ordem das colunas, então a linha soma
(`voltas` = soma das regras + `úteis`) e nada é contado duas vezes.

- **desconhecido**: `pit_in`, `pit_out` ou `track_status` nulos (volta gravada antes de existirem).
- **sem_tempo**: sem tempo de volta.
- **volta_1**: a largada.
- **box**: volta de entrada ou de saída dos boxes.
- **sem_pneu**: sem composto, idade do pneu ou stint.
- **pista_nao_verde**: status diferente de `1` por algo além da amarela (SC, VSC, vermelha...).
- **so_amarela**: status só de verde e amarela (`12`, `21`, `2`). Fica por último para a coluna
  ser o custo **marginal** da amarela: o que ela tira além de todas as outras regras.
- **perda se a amarela ficasse**: a perda que a corrida teria se a amarela fosse mantida.
- **alerta**: `perde N%` se a corrida perde mais de 40% das voltas; `sem pista seca` se menos da
  metade das voltas úteis é de SOFT, MEDIUM ou HARD (não há desgaste de pneu seco a modelar);
  `pista mista` se há voltas úteis de pista molhada junto com as de seco (a pista seca durante o
  stint, e o tempo de volta cai por isso, não por falta de desgaste).

A regra de exclusão do status é uma só: fica quem tem o status exatamente `1`. A divisão entre
`pista_nao_verde` e `so_amarela` existe só para este relatório.

## Funil por corrida

| corrida | voltas | desconhecido | sem_tempo | volta_1 | box | sem_pneu | pista_nao_verde | so_amarela | úteis | perda | perda se a amarela ficasse | alerta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | 1129 | 0 | 2 | 20 | 83 | 0 | 0 | 18 | 1006 | 11% | 9% | - |
| 2024 R02 Saudi Arabian | 901 | 0 | 28 | 20 | 24 | 0 | 13 | 14 | 802 | 11% | 9% | - |
| 2024 R03 Australian | 998 | 0 | 3 | 19 | 72 | 0 | 47 | 5 | 852 | 15% | 14% | - |
| 2024 R04 Japanese | 907 | 0 | 31 | 18 | 72 | 0 | 0 | 21 | 765 | 16% | 13% | - |
| 2024 R05 Chinese | 1032 | 0 | 25 | 20 | 72 | 0 | 153 | 18 | 744 | 28% | 26% | - |
| 2024 R06 Miami | 1111 | 0 | 5 | 20 | 56 | 0 | 96 | 0 | 934 | 16% | 16% | - |
| 2024 R07 Emilia Romagna | 1238 | 0 | 1 | 20 | 53 | 0 | 0 | 0 | 1164 | 6% | 6% | - |
| 2024 R08 Monaco | 1237 | 0 | 11 | 16 | 23 | 0 | 0 | 16 | 1171 | 5% | 4% | - |
| 2024 R09 Canadian | 1272 | 0 | 5 | 20 | 84 | 0 | 141 | 108 | 914 | 28% | 20% | sem pista seca |
| 2024 R10 Spanish | 1310 | 0 | 0 | 20 | 84 | 0 | 0 | 0 | 1206 | 8% | 8% | - |
| 2024 R11 Austrian | 1405 | 0 | 0 | 20 | 89 | 0 | 27 | 19 | 1250 | 11% | 10% | - |
| 2024 R12 British | 960 | 0 | 0 | 19 | 91 | 0 | 0 | 0 | 850 | 11% | 11% | pista mista (22% molhado) |
| 2024 R13 Hungarian | 1355 | 0 | 0 | 20 | 81 | 0 | 0 | 21 | 1233 | 9% | 7% | - |
| 2024 R14 Belgian | 841 | 0 | 1 | 20 | 69 | 0 | 0 | 17 | 734 | 13% | 11% | - |
| 2024 R15 Dutch | 1426 | 0 | 0 | 20 | 52 | 0 | 0 | 0 | 1354 | 5% | 5% | - |
| 2024 R16 Italian | 1008 | 0 | 0 | 20 | 61 | 0 | 0 | 0 | 927 | 8% | 8% | - |
| 2024 R17 Azerbaijan | 973 | 0 | 21 | 20 | 41 | 0 | 10 | 28 | 853 | 12% | 9% | - |
| 2024 R18 Singapore | 1177 | 0 | 1 | 20 | 47 | 0 | 0 | 0 | 1109 | 6% | 6% | - |
| 2024 R19 United States | 1059 | 0 | 28 | 20 | 45 | 0 | 29 | 54 | 883 | 17% | 12% | - |
| 2024 R20 Mexico City | 1215 | 0 | 2 | 18 | 43 | 0 | 90 | 0 | 1062 | 13% | 13% | - |
| 2024 R21 São Paulo | 1134 | 0 | 37 | 18 | 31 | 0 | 105 | 143 | 800 | 29% | 17% | sem pista seca |
| 2024 R22 Las Vegas | 938 | 0 | 0 | 20 | 79 | 0 | 0 | 0 | 839 | 11% | 11% | - |
| 2024 R23 Qatar | 943 | 0 | 20 | 18 | 75 | 0 | 113 | 38 | 679 | 28% | 24% | - |
| 2024 R24 Abu Dhabi | 1035 | 0 | 2 | 19 | 56 | 0 | 37 | 38 | 883 | 15% | 11% | - |
