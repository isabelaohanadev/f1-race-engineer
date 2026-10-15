# Diagnóstico pós-resultado do Monte Carlo

Gerado por `python -m strategy.diagnostics` em 2026-10-15 sobre 21 corridas secas, **depois** do backtest. Não é parte do backtest.

## O que é isto

**Depois de ver o resultado do backtest, e fora dele.** O backtest congelado (`docs/backtest.md`)
mostrou que o Monte Carlo não supera "o líder atual vence" e é pior nas últimas faixas. Esta análise
pergunta **onde** está a hesitação. Nenhum valor do protocolo mudou e nada do que foi medido foi
refeito: cada linha abaixo é o mesmo simulador, nos mesmos números aleatórios, com UM componente
desligado.

- "p do favorito" é a probabilidade que o Monte Carlo dá ao piloto que ele considera mais provável
  de vencer; "favorito ganhou" é a frequência com que ele ganhou. Se a primeira é menor que a
  segunda, o Monte Carlo é **sub-confiante** (hesita mais do que deveria).
- "líder atual ganhou" é a frequência com que quem lidera no instante da previsão vence. Se o
  favorito do Monte Carlo ganha menos que o líder atual, a escolha dele é pior que a posição na
  pista.
- As ablações desligam componentes que **existem na realidade**: o resultado mostra quanto cada um
  custa em hesitação, não que deva ser desligado.

## Resultado

| configuração | andamento | previsões | Brier da vitória | p do favorito | favorito ganhou | líder atual ganhou |
|---|---|---|---|---|---|---|
| congelado (o backtest) | 0% a 40% | 87 | 0.757 [0.651, 0.863] | 0.33 | 0.41 | 0.68 |
| congelado (o backtest) | 40% a 80% | 102 | 0.500 [0.371, 0.647] | 0.54 | 0.68 | 0.81 |
| congelado (o backtest) | 80% a 100% | 49 | 0.267 [0.118, 0.451] | 0.84 | 0.82 | 0.94 |
| sem incerteza do modelo | 0% a 40% | 87 | 0.899 [0.646, 1.168] | 0.76 | 0.44 | 0.68 |
| sem incerteza do modelo | 40% a 80% | 102 | 0.553 [0.317, 0.829] | 0.83 | 0.68 | 0.81 |
| sem incerteza do modelo | 80% a 100% | 49 | 0.314 [0.120, 0.551] | 0.93 | 0.82 | 0.94 |
| sem ruído de volta | 0% a 40% | 87 | 0.749 [0.639, 0.860] | 0.34 | 0.43 | 0.68 |
| sem ruído de volta | 40% a 80% | 102 | 0.476 [0.349, 0.615] | 0.54 | 0.70 | 0.81 |
| sem ruído de volta | 80% a 100% | 49 | 0.234 [0.092, 0.419] | 0.85 | 0.86 | 0.94 |
| sem SC e sem abandono | 0% a 40% | 87 | 0.755 [0.637, 0.874] | 0.36 | 0.43 | 0.68 |
| sem SC e sem abandono | 40% a 80% | 102 | 0.494 [0.360, 0.646] | 0.55 | 0.68 | 0.81 |
| sem SC e sem abandono | 80% a 100% | 49 | 0.269 [0.117, 0.458] | 0.85 | 0.82 | 0.94 |
| sem folga na decisão de parar | 0% a 40% | 87 | 0.756 [0.649, 0.863] | 0.34 | 0.41 | 0.68 |
| sem folga na decisão de parar | 40% a 80% | 102 | 0.500 [0.370, 0.648] | 0.54 | 0.68 | 0.81 |
| sem folga na decisão de parar | 80% a 100% | 49 | 0.267 [0.117, 0.454] | 0.84 | 0.82 | 0.94 |
| tudo desligado (quase determinístico) | 0% a 40% | 87 | 0.862 [0.585, 1.178] | 0.89 | 0.57 | 0.68 |
| tudo desligado (quase determinístico) | 40% a 80% | 102 | 0.568 [0.305, 0.869] | 0.92 | 0.71 | 0.81 |
| tudo desligado (quase determinístico) | 80% a 100% | 49 | 0.254 [0.048, 0.524] | 1.00 | 0.88 | 0.94 |

## Leitura (faixa de 40% a 80% da corrida)

- No meio da corrida o Monte Carlo congelado dá ao favorito 0.54 e o favorito ganha 0.68: sub-confiante. O líder atual ganha 0.81.
- **sem incerteza do modelo:** p do favorito 0.83, ganhou 0.68 (super-confiante); o favorito ganha menos que o líder atual (0.81).
- **sem ruído de volta:** p do favorito 0.54, ganhou 0.70 (sub-confiante); o favorito ganha menos que o líder atual (0.81).
- **sem SC e sem abandono:** p do favorito 0.55, ganhou 0.68 (sub-confiante); o favorito ganha menos que o líder atual (0.81).
- **sem folga na decisão de parar:** p do favorito 0.54, ganhou 0.68 (sub-confiante); o favorito ganha menos que o líder atual (0.81).
- **tudo desligado (quase determinístico):** p do favorito 0.92, ganhou 0.71 (super-confiante); o favorito ganha menos que o líder atual (0.81).
