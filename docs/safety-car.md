# Safety Car, abandonos e trocas de composto

Gerado por `python -m strategy.history --markdown docs/safety-car.md` em 2026-10-14.

## Resumo

- **Safety Car:** 8 períodos em 7 de 21 corridas secas (0.38 por corrida; probabilidade de ao menos um na corrida 33%). Voltas de início: [1, 1, 1, 3, 7, 23, 28, 35].
- **Só 3 dos 8 começam depois da volta 10**; os demais são incidentes de largada. A chance de SC no meio da corrida vem desses poucos eventos e é muito incerta.
- **Duração** (voltas): [2, 3, 3, 4, 5, 6, 8, 9]; mediana 4.5.
- **VSC:** 8 períodos (fora do simulador). **Vermelha:** 2 (fora do simulador).
- **Abandonos:** 41 em 419 largadas (9.8% por piloto-corrida).
- **Perda da parada contra o campo:** verde 22.6 s (n=530); sob SC 13.2 s (n=15), razão 0.58.
- **Ritmo sob SC:** 1.49x a volta verde (por corrida de 1.23 a 1.62, 4 corridas). **Espaçamento entre carros:** 1.91 s (de 1.37 a 2.35).

## Como ler

- **SC como evento da corrida.** O número de SCs é Poisson(lambda), o início sai das frações
  históricas e a duração das durações históricas; a incerteza de lambda (Gamma posteriori) entra em
  cada simulação. Detalhes em `strategy/history.py`.
- **A perda da parada sob SC é medida contra o CAMPO**, e não contra o ritmo próprio do piloto (que
  dá mais sob SC, porque todo o campo anda devagar). Em bandeira verde o método dá o mesmo que
  `docs/perda-no-pit.md`, o que o valida.
- **Poucos dados.** Todas as razões e probabilidades têm o `n` ao lado; o das paradas sob SC é
  pequeno e muito disperso, e a razão de custo tem um desvio grande.
- **Não está aqui:** o desgaste do pneu sob SC (não se mede num tempo de volta sob SC; é uma
  heurística declarada no simulador), VSC e bandeira vermelha.

## Por corrida

| corrida | voltas | SC (início-fim) | VSC | vermelha | largaram | abandonos | ritmo no SC | espaçamento (s) | paradas verdes | paradas sob SC |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | 57 | - | - | - | 20 | 0 | - | - | 35 | 0 |
| 2024 R02 Saudi Arabian | 50 | 7-9 | - | - | 20 | 2 | - | - | 5 | 0 |
| 2024 R03 Australian | 58 | - | 17-17, 57-58 | - | 19 | 3 | - | - | 31 | 0 |
| 2024 R04 Japanese | 53 | - | - | 1-1 | 20 | 3 | - | - | 35 | 0 |
| 2024 R05 Chinese | 56 | 23-31 | 21-23 | - | 20 | 3 | 1.39 | 1.37 | 22 | 2 |
| 2024 R06 Miami | 57 | 28-32 | 22-23 | - | 20 | 1 | 1.23 | 2.35 | 11 | 10 |
| 2024 R07 Emilia Romagna | 63 | - | - | - | 20 | 1 | - | - | 27 | 0 |
| 2024 R08 Monaco | 78 | 1-2 | - | 1-1 | 20 | 4 | - | - | 5 | 0 |
| 2024 R10 Spanish | 66 | - | - | - | 20 | 0 | - | - | 42 | 0 |
| 2024 R11 Austrian | 71 | - | 65-65 | - | 20 | 0 | - | - | 42 | 0 |
| 2024 R13 Hungarian | 70 | - | - | - | 20 | 1 | - | - | 39 | 0 |
| 2024 R14 Belgian | 44 | - | - | - | 20 | 1 | - | - | 32 | 0 |
| 2024 R15 Dutch | 72 | - | - | - | 20 | 0 | - | - | 26 | 0 |
| 2024 R16 Italian | 53 | - | - | - | 20 | 1 | - | - | 30 | 0 |
| 2024 R17 Azerbaijan | 51 | - | 50-51 | - | 20 | 4 | - | - | 15 | 0 |
| 2024 R18 Singapore | 62 | - | - | - | 20 | 2 | - | - | 23 | 0 |
| 2024 R19 United States | 56 | 3-5 | - | - | 20 | 1 | - | - | 21 | 1 |
| 2024 R20 Mexico City | 71 | 1-6 | - | - | 20 | 3 | 1.62 | 1.86 | 21 | 0 |
| 2024 R22 Las Vegas | 50 | - | - | - | 20 | 2 | - | - | 39 | 0 |
| 2024 R23 Qatar | 57 | 1-4, 35-42 | 40-40 | - | 20 | 5 | 1.59 | 1.95 | 8 | 2 |
| 2024 R24 Abu Dhabi | 58 | - | 2-3 | - | 20 | 4 | - | - | 21 | 0 |
