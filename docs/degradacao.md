# Modelo de degradação do pneu

Gerado por `python -m strategy.degradation --markdown docs/degradacao.md` em 2026-10-14. A regra está em `strategy/degradation.py`.

## Resumo

- 21 corridas do backtest principal, modelo ajustado com a corrida inteira e a variante `encolhimento` do `k`.
- Ruído de volta (σ), mediana entre corridas: 0.320 s. É a dispersão de UMA volta em torno da reta; não cai com mais dados.
- Variação da inclinação entre pilotos/stints (τ), mediana: 0.022 s/volta de idade (21 de 21 corridas com dados para estimá-la; as demais usam o padrão 0.022).
- Stints futuros (`k` do modelo, n=251): o intervalo de 95% só com o erro-padrão cobre a inclinação real em 66%; com a variação τ, em 91%. O alvo é 95%.
- Stints futuros (`k` da corrida inteira, n=251): o intervalo de 95% só com o erro-padrão cobre a inclinação real em 67%; com a variação τ, em 87%. O alvo é 95%.

## Método

- **Dados**: voltas limpas (status exatamente `1`, sem box, sem volta 1) e de pneu de pista seca,
  só até a volta N. O tempo é corrigido do combustível com a variante do `k` escolhida.
- **Por (piloto, composto)**: reta do tempo corrigido contra a idade, com um intercepto por stint e
  Huber (`fit_within`). Precisa de pelo menos 6 voltas, em stints de pelo menos 3.
- **Variação entre pilotos/stints τ**: DerSimonian-Laird sobre as inclinações dessas unidades, com
  médias por composto. Se sobram menos de 3 graus de liberdade, usa o padrão (0,03).
- **Encolhimento**: a inclinação do piloto vai para a média do composto na corrida com peso
  `B = τ² / (τ² + se²)`. Sem dado do piloto naquele composto, vale a média do composto; sem dado do
  composto, a de todos os compostos juntos; sem nada, não há modelo.
- **Três números de incerteza, separados**: `se` (parâmetro; cai com dados), `σ` (ruído de uma
  volta; não cai) e `τ` (variação entre pilotos/stints). `predictive_sd = sqrt(se² + τ²)` é a de
  um stint FUTURO, a que o undercut deve usar.
- **Calibração**: para os stints que começam depois de 50% da corrida, o intervalo previsto com o
  modelo cortado em 50% cobre a inclinação que o stint teve de fato? Compara o intervalo só com o
  erro-padrão e o intervalo com `τ`. Soma o erro da própria medida do stint.

## Por corrida, com a corrida inteira

Inclinação do composto na corrida (s por volta de idade, `+` = o pneu fica mais lento), ± erro-padrão, e entre parênteses quantos (piloto, composto) entraram.

| corrida | k (s/volta) | peso do k estimado | ruído de volta σ | variação entre pilotos/stints τ | HARD (± erro-padrão, unidades) | MEDIUM (± erro-padrão, unidades) | SOFT (± erro-padrão, unidades) |
|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | 0.071 | 0.99 | 0.237 | 0.022 | +0.112 ± 0.006 (20) | - | +0.131 ± 0.008 (20) |
| 2024 R02 Saudi Arabian | 0.111 | 0.91 | 0.380 | 0.018 | +0.044 ± 0.005 (15) | +0.045 ± 0.010 (4) | +0.119 ± 0.015 (4) |
| 2024 R03 Australian | 0.053 | 0.99 | 0.313 | 0.018 | +0.046 ± 0.005 (18) | +0.060 ± 0.010 (13) | - |
| 2024 R04 Japanese | 0.096 | 0.99 | 0.383 | 0.073 | +0.125 ± 0.019 (17) | +0.174 ± 0.021 (13) | +0.206 ± 0.051 (3) |
| 2024 R05 Chinese | 0.085 | 0.94 | 0.265 | 0.027 | +0.125 ± 0.007 (18) | +0.163 ± 0.010 (20) | +0.173 ± 0.022 (6) |
| 2024 R06 Miami | 0.054 | 0.99 | 0.278 | 0.024 | +0.032 ± 0.006 (19) | +0.022 ± 0.006 (20) | +0.020 ± 0.025 (2) |
| 2024 R07 Emilia Romagna | 0.035 | 0.98 | 0.310 | 0.019 | +0.047 ± 0.005 (19) | +0.076 ± 0.005 (20) | -0.100 ± 0.091 (1) |
| 2024 R08 Monaco | 0.084 | 0.98 | 0.657 | 0.017 | +0.035 ± 0.005 (12) | +0.019 ± 0.007 (8) | +0.119 ± 0.030 (1) |
| 2024 R10 Spanish | 0.061 | 0.99 | 0.350 | 0.028 | +0.081 ± 0.009 (13) | +0.088 ± 0.007 (19) | +0.094 ± 0.008 (20) |
| 2024 R11 Austrian | 0.028 | 1.00 | 0.257 | 0.015 | +0.065 ± 0.004 (20) | +0.090 ± 0.005 (20) | - |
| 2024 R13 Hungarian | 0.048 | 0.99 | 0.422 | 0.025 | +0.081 ± 0.006 (20) | +0.067 ± 0.008 (13) | -0.126 ± 0.038 (1) |
| 2024 R14 Belgian | 0.093 | 0.99 | 0.328 | 0.039 | +0.077 ± 0.011 (15) | +0.096 ± 0.012 (19) | +0.050 ± 0.029 (2) |
| 2024 R15 Dutch | 0.040 | 0.98 | 0.293 | 0.013 | +0.040 ± 0.003 (20) | +0.041 ± 0.004 (19) | +0.061 ± 0.010 (5) |
| 2024 R16 Italian | 0.063 | 1.00 | 0.361 | 0.024 | +0.049 ± 0.006 (19) | +0.022 ± 0.011 (18) | - |
| 2024 R17 Azerbaijan | 0.077 | 0.92 | 0.425 | 0.025 | +0.039 ± 0.006 (19) | +0.034 ± 0.012 (16) | - |
| 2024 R18 Singapore | 0.047 | 0.99 | 0.395 | 0.020 | +0.061 ± 0.006 (16) | +0.065 ± 0.006 (19) | +0.027 ± 0.017 (5) |
| 2024 R19 United States | 0.072 | 0.99 | 0.320 | 0.028 | +0.034 ± 0.007 (19) | +0.043 ± 0.008 (19) | - |
| 2024 R20 Mexico City | 0.044 | 0.99 | 0.300 | 0.017 | +0.029 ± 0.005 (17) | +0.079 ± 0.005 (18) | - |
| 2024 R22 Las Vegas | 0.083 | 0.98 | 0.542 | 0.044 | +0.063 ± 0.012 (19) | +0.135 ± 0.022 (19) | - |
| 2024 R23 Qatar | 0.065 | 0.58 | 0.267 | 0.010 | -0.017 ± 0.007 (12) | -0.021 ± 0.003 (16) | +0.125 ± 0.036 (3) |
| 2024 R24 Abu Dhabi | 0.058 | 0.99 | 0.253 | 0.019 | +0.057 ± 0.005 (19) | +0.093 ± 0.006 (16) | +0.158 ± 0.030 (1) |

## Calibração: o intervalo cobre o stint que ainda não aconteceu?

| stints futuros | n | cobertura só com o erro-padrão | cobertura com a variação τ |
|---|---|---|---|
| `k` do modelo (isola a variação entre stints) | 251 | 66% | 91% |
| `k` da corrida inteira (inclui o erro do `k`) | 251 | 67% | 87% |
|   só previsões por piloto (k do modelo) | 134 | 66% | 93% |
|   só previsões por composto (k do modelo) | 98 | 66% | 94% |
|   só previsões por todos os compostos (k do modelo) | 19 | 58% | 68% |

## O `k` muda a degradação: as três variantes

Mediana, entre as corridas, da inclinação do composto. Cada +0,01 s/volta no `k` soma +0,01 s/volta à inclinação (ver `docs/combustivel.md`); o backtest decide a variante.

| variante do k | HARD: mediana entre corridas | MEDIUM: mediana entre corridas | SOFT: mediana entre corridas |
|---|---|---|---|
| fisico | +0.050 | +0.068 | +0.076 |
| estimado | +0.049 | +0.066 | +0.086 |
| encolhimento | +0.049 | +0.066 | +0.107 |

## Reportadas à parte (fora da nota principal)

Pista mista ou molhada: o modelo ajusta só as voltas de pneu de seco, e nelas a pista está secando ou molhada, então os números abaixo **não medem a qualidade do modelo**. Estão aqui para serem vistos, não para entrar em nenhuma média.

Corridas: 2024 R09 Canadian: pista molhada (33% seca); 2024 R12 British: pista mista (76% seca); 2024 R21 São Paulo: pista molhada (0% seca).

| corrida | k (s/volta) | peso do k estimado | ruído de volta σ | variação entre pilotos/stints τ | HARD (± erro-padrão, unidades) | MEDIUM (± erro-padrão, unidades) | SOFT (± erro-padrão, unidades) |
|---|---|---|---|---|---|---|---|
| 2024 R09 Canadian | 0.052 | 0.10 | 0.866 | 0.162 | -0.223 ± 0.087 (4) | -0.307 ± 0.048 (14) | - |
| 2024 R12 British | 0.065 | 0.99 | 0.730 | 0.039 | +0.037 ± 0.028 (3) | +0.133 ± 0.014 (18) | +0.134 ± 0.022 (11) |
| 2024 R21 São Paulo | 0.048 | 0.00 | - | 0.022 (padrão) | - | - | - |
