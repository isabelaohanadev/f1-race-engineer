# Combustível: estimado x físico e sensibilidade da degradação

Gerado por `python -m strategy.fuel --markdown docs/combustivel.md` em 2026-10-14. A regra está em `strategy/fuel.py`.

## Resumo

- 21 corridas secas analisadas (reportadas à parte, fora da nota principal: 2024 R09 Canadian: pista molhada (33% seca); 2024 R12 British: pista mista (76% seca); 2024 R21 São Paulo: pista molhada (0% seca)).
- Físico: de 0.042 a 0.075 s/volta, conforme o número de voltas.
- Com a corrida inteira, o coeficiente é identificável em 21 de 21 corridas; o critério escolhe o estimado em 20 e o físico em 1.
- Razão estimado/físico entre as identificáveis: mediana 1.20, de 0.60 a 2.25.
- Erro-padrão mediano do estimado: 0.003 s/volta (físico mediano: 0.058).
- Por que o físico, quando foi o escolhido: diverge do físico em 1.
- Sensibilidade: somar `k * volta` ao tempo soma `k` à inclinação da idade. Desvio máximo entre a variação medida da degradação e a variação de `k`, em todas as linhas: 1.0e-11 s/volta (a identidade vale; a incerteza do combustível passa 1 para 1).
- Degradação mediana (linhas com 30+ voltas): sem correção +0.007; `k` a 0,5x +0.033; `k` físico +0.060; `k` a 1,5x +0.085 s por volta de idade.
- Linhas com degradação NEGATIVA (o pneu 'melhora' com a idade, o que não é desgaste): sem correção 25 de 53; `k` físico 7 de 53; `k` escolhido 5 de 53.
- Com dados só até 25% da corrida: estimado em 2 de 21.
- Com dados só até 50% da corrida: estimado em 17 de 21.
- Com dados só até 75% da corrida: estimado em 19 de 21.

## Método

- **Correção**: `tempo_corrigido = tempo + k * (volta - 1)`, o tempo que a volta teria com o
  combustível da volta 1. `k` = segundos que o carro ganha por volta ao queimar combustível.
- **Físico**: `k = 0,03 s/kg * 110 kg / voltas`. A constante de 0,03 s/kg é uma **suposição** de
  engenharia (0,025 a 0,035 conforme o circuito), não medida com estes dados.
- **Estimado**: `tempo = piloto + composto + degradação_do_composto * idade - k * volta`, com
  mínimos quadrados robustos (Huber) e erro-padrão por cluster de piloto, nas voltas limpas e secas.
  O intercepto é **por piloto e não por stint**: com intercepto por stint, `k` não se separa da
  degradação (dentro do stint `volta = idade + constante`). O `k` estimado contém tudo o que melhora
  o ritmo com a volta da corrida, não só o combustível: a pista que borracha, a temperatura.
- **Critério** (fixado antes de olhar os resultados): o estimado só é usado se for identificável,
  preciso (erro-padrão de no máximo metade do físico) e parecido (razão de 0,5 a 2 em relação ao
  físico). Senão usa o físico e a coluna `motivo` diz por quê.
- **Sensibilidade**: a degradação de cada composto (inclinação do tempo corrigido contra a idade,
  intercepto por stint) para `k` de 0 a 2 vezes o físico. Dentro do stint
  `volta = idade + constante`, então **cada +0,01 s/volta de `k` soma exatamente +0,01 s/volta à
  degradação estimada, em todos os compostos**: a incerteza do combustível passa 1 para 1 para a da
  degradação, e a ordem entre compostos não muda.

## Estimado x físico, com a corrida inteira

| corrida | voltas | voltas úteis | físico (s/volta) | estimado ± erro-padrão | razão | usa | motivo |
|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | 57 | 1006 | 0.058 | 0.071 ± 0.003 | 1.23 | estimado | parecido e preciso |
| 2024 R02 Saudi Arabian | 50 | 802 | 0.066 | 0.116 ± 0.004 | 1.76 | estimado | parecido e preciso |
| 2024 R03 Australian | 58 | 852 | 0.057 | 0.053 ± 0.002 | 0.93 | estimado | parecido e preciso |
| 2024 R04 Japanese | 53 | 765 | 0.062 | 0.096 ± 0.002 | 1.55 | estimado | parecido e preciso |
| 2024 R05 Chinese | 56 | 744 | 0.059 | 0.086 ± 0.005 | 1.46 | estimado | parecido e preciso |
| 2024 R06 Miami | 57 | 934 | 0.058 | 0.054 ± 0.003 | 0.94 | estimado | parecido e preciso |
| 2024 R07 Emilia Romagna | 63 | 1164 | 0.052 | 0.035 ± 0.003 | 0.67 | estimado | parecido e preciso |
| 2024 R08 Monaco | 78 | 1171 | 0.042 | 0.095 ± 0.002 | 2.25 | físico | diverge do físico (razão 2.25, fora de 0,5 a 2) |
| 2024 R10 Spanish | 66 | 1206 | 0.050 | 0.062 ± 0.002 | 1.23 | estimado | parecido e preciso |
| 2024 R11 Austrian | 71 | 1250 | 0.046 | 0.028 ± 0.001 | 0.60 | estimado | parecido e preciso |
| 2024 R13 Hungarian | 70 | 1233 | 0.047 | 0.048 ± 0.003 | 1.02 | estimado | parecido e preciso |
| 2024 R14 Belgian | 44 | 734 | 0.075 | 0.093 ± 0.003 | 1.24 | estimado | parecido e preciso |
| 2024 R15 Dutch | 72 | 1354 | 0.046 | 0.040 ± 0.003 | 0.87 | estimado | parecido e preciso |
| 2024 R16 Italian | 53 | 927 | 0.062 | 0.063 ± 0.002 | 1.00 | estimado | parecido e preciso |
| 2024 R17 Azerbaijan | 51 | 853 | 0.065 | 0.078 ± 0.008 | 1.20 | estimado | parecido e preciso |
| 2024 R18 Singapore | 62 | 1109 | 0.053 | 0.047 ± 0.002 | 0.89 | estimado | parecido e preciso |
| 2024 R19 United States | 56 | 883 | 0.059 | 0.073 ± 0.003 | 1.23 | estimado | parecido e preciso |
| 2024 R20 Mexico City | 71 | 1062 | 0.046 | 0.044 ± 0.002 | 0.95 | estimado | parecido e preciso |
| 2024 R22 Las Vegas | 50 | 839 | 0.066 | 0.083 ± 0.004 | 1.26 | estimado | parecido e preciso |
| 2024 R23 Qatar | 57 | 679 | 0.058 | 0.070 ± 0.021 | 1.21 | estimado | parecido e preciso |
| 2024 R24 Abu Dhabi | 58 | 883 | 0.057 | 0.058 ± 0.002 | 1.03 | estimado | parecido e preciso |

## Decisão com dados só até cada fração da corrida (sem o futuro)

| corrida | até 25% da corrida | até 50% da corrida | até 75% da corrida |
|---|---|---|---|
| 2024 R01 Bahrain | físico (-1.33x) | estimado (0.70x) | estimado (1.28x) |
| 2024 R02 Saudi Arabian | físico (19.69x) | físico (4.56x) | estimado (1.84x) |
| 2024 R03 Australian | físico (0.66x) | estimado (0.96x) | estimado (0.79x) |
| 2024 R04 Japanese | físico (-5.49x) | estimado (1.40x) | estimado (1.47x) |
| 2024 R05 Chinese | estimado (1.10x) | estimado (1.80x) | estimado (1.61x) |
| 2024 R06 Miami | físico (7.79x) | estimado (0.93x) | estimado (1.00x) |
| 2024 R07 Emilia Romagna | físico (1.11x) | estimado (0.51x) | estimado (0.63x) |
| 2024 R08 Monaco | físico (-) | físico (-) | físico (2.57x) |
| 2024 R10 Spanish | físico (2.19x) | estimado (1.36x) | estimado (1.21x) |
| 2024 R11 Austrian | físico (-0.13x) | estimado (0.56x) | estimado (0.57x) |
| 2024 R13 Hungarian | físico (1.19x) | estimado (1.23x) | estimado (1.14x) |
| 2024 R14 Belgian | estimado (1.65x) | estimado (1.36x) | estimado (1.46x) |
| 2024 R15 Dutch | físico (2.51x) | estimado (0.81x) | estimado (0.87x) |
| 2024 R16 Italian | físico (0.14x) | estimado (1.11x) | estimado (0.96x) |
| 2024 R17 Azerbaijan | físico (9.62x) | estimado (0.96x) | estimado (1.03x) |
| 2024 R18 Singapore | físico (7.49x) | estimado (0.50x) | estimado (0.89x) |
| 2024 R19 United States | físico (-) | estimado (1.13x) | estimado (1.35x) |
| 2024 R20 Mexico City | físico (-) | físico (0.44x) | estimado (0.89x) |
| 2024 R22 Las Vegas | físico (5.66x) | estimado (1.62x) | estimado (1.15x) |
| 2024 R23 Qatar | físico (-) | físico (-) | físico (-0.25x) |
| 2024 R24 Abu Dhabi | físico (-0.96x) | estimado (0.66x) | estimado (0.97x) |

## Sensibilidade: degradação (s por volta de idade) para `k` de 0 a 2x o físico

| corrida | composto | voltas | 0x (sem correção) | 0.5x | 0.75x | 1x (físico) | 1.25x | 1.5x | 2x |
|---|---|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | HARD | 729 | +0.040 | +0.069 | +0.083 | +0.098 | +0.112 | +0.127 | +0.156 |
| 2024 R01 Bahrain | SOFT | 277 | +0.066 | +0.095 | +0.109 | +0.124 | +0.138 | +0.153 | +0.182 |
| 2024 R02 Saudi Arabian | HARD | 570 | -0.068 | -0.035 | -0.019 | -0.002 | +0.014 | +0.031 | +0.064 |
| 2024 R02 Saudi Arabian | MEDIUM | 179 | -0.067 | -0.034 | -0.017 | -0.001 | +0.016 | +0.032 | +0.065 |
| 2024 R02 Saudi Arabian | SOFT | 53 | +0.016 | +0.049 | +0.066 | +0.082 | +0.099 | +0.115 | +0.148 |
| 2024 R03 Australian | HARD | 692 | -0.010 | +0.018 | +0.033 | +0.047 | +0.061 | +0.075 | +0.104 |
| 2024 R03 Australian | MEDIUM | 148 | +0.007 | +0.035 | +0.049 | +0.064 | +0.078 | +0.092 | +0.121 |
| 2024 R03 Australian | SOFT | 12 | - | - | - | - | - | - | - |
| 2024 R04 Japanese | HARD | 483 | +0.016 | +0.047 | +0.063 | +0.078 | +0.094 | +0.110 | +0.141 |
| 2024 R04 Japanese | MEDIUM | 227 | +0.021 | +0.052 | +0.068 | +0.083 | +0.099 | +0.114 | +0.145 |
| 2024 R04 Japanese | SOFT | 55 | +0.047 | +0.078 | +0.093 | +0.109 | +0.125 | +0.140 | +0.171 |
| 2024 R05 Chinese | HARD | 414 | +0.033 | +0.063 | +0.077 | +0.092 | +0.107 | +0.122 | +0.151 |
| 2024 R05 Chinese | MEDIUM | 274 | +0.062 | +0.091 | +0.106 | +0.121 | +0.135 | +0.150 | +0.179 |
| 2024 R05 Chinese | SOFT | 56 | +0.126 | +0.155 | +0.170 | +0.185 | +0.199 | +0.214 | +0.244 |
| 2024 R06 Miami | HARD | 413 | -0.028 | +0.001 | +0.015 | +0.030 | +0.044 | +0.059 | +0.088 |
| 2024 R06 Miami | MEDIUM | 484 | -0.035 | -0.006 | +0.009 | +0.023 | +0.038 | +0.052 | +0.081 |
| 2024 R06 Miami | SOFT | 37 | -0.037 | -0.008 | +0.006 | +0.021 | +0.035 | +0.050 | +0.078 |
| 2024 R07 Emilia Romagna | HARD | 684 | +0.007 | +0.033 | +0.046 | +0.060 | +0.073 | +0.086 | +0.112 |
| 2024 R07 Emilia Romagna | MEDIUM | 467 | +0.034 | +0.060 | +0.073 | +0.086 | +0.099 | +0.113 | +0.139 |
| 2024 R07 Emilia Romagna | SOFT | 13 | - | - | - | - | - | - | - |
| 2024 R08 Monaco | HARD | 753 | -0.056 | -0.035 | -0.024 | -0.013 | -0.003 | +0.008 | +0.029 |
| 2024 R08 Monaco | MEDIUM | 387 | -0.070 | -0.048 | -0.038 | -0.027 | -0.017 | -0.006 | +0.015 |
| 2024 R08 Monaco | SOFT | 31 | +0.033 | +0.054 | +0.065 | +0.075 | +0.086 | +0.096 | +0.117 |
| 2024 R10 Spanish | HARD | 346 | +0.019 | +0.044 | +0.056 | +0.069 | +0.081 | +0.094 | +0.119 |
| 2024 R10 Spanish | MEDIUM | 401 | +0.021 | +0.046 | +0.059 | +0.071 | +0.084 | +0.096 | +0.121 |
| 2024 R10 Spanish | SOFT | 459 | +0.036 | +0.061 | +0.074 | +0.086 | +0.099 | +0.111 | +0.136 |
| 2024 R11 Austrian | HARD | 660 | +0.032 | +0.055 | +0.067 | +0.079 | +0.090 | +0.102 | +0.125 |
| 2024 R11 Austrian | MEDIUM | 584 | +0.058 | +0.082 | +0.093 | +0.105 | +0.117 | +0.128 | +0.151 |
| 2024 R11 Austrian | SOFT | 6 | - | - | - | - | - | - | - |
| 2024 R13 Hungarian | HARD | 806 | +0.030 | +0.054 | +0.066 | +0.077 | +0.089 | +0.101 | +0.125 |
| 2024 R13 Hungarian | MEDIUM | 393 | +0.019 | +0.042 | +0.054 | +0.066 | +0.078 | +0.089 | +0.113 |
| 2024 R13 Hungarian | SOFT | 34 | -0.137 | -0.113 | -0.101 | -0.089 | -0.078 | -0.066 | -0.042 |
| 2024 R14 Belgian | HARD | 367 | -0.029 | +0.009 | +0.027 | +0.046 | +0.065 | +0.084 | +0.121 |
| 2024 R14 Belgian | MEDIUM | 310 | -0.029 | +0.009 | +0.028 | +0.046 | +0.065 | +0.084 | +0.121 |
| 2024 R14 Belgian | SOFT | 57 | -0.045 | -0.008 | +0.011 | +0.030 | +0.048 | +0.067 | +0.105 |
| 2024 R15 Dutch | HARD | 780 | -0.001 | +0.022 | +0.033 | +0.045 | +0.056 | +0.068 | +0.090 |
| 2024 R15 Dutch | MEDIUM | 470 | +0.002 | +0.025 | +0.036 | +0.048 | +0.059 | +0.071 | +0.094 |
| 2024 R15 Dutch | SOFT | 104 | +0.016 | +0.039 | +0.051 | +0.062 | +0.074 | +0.085 | +0.108 |
| 2024 R16 Italian | HARD | 686 | -0.013 | +0.018 | +0.034 | +0.049 | +0.065 | +0.080 | +0.112 |
| 2024 R16 Italian | MEDIUM | 240 | -0.026 | +0.005 | +0.020 | +0.036 | +0.052 | +0.067 | +0.098 |
| 2024 R16 Italian | SOFT | 1 | - | - | - | - | - | - | - |
| 2024 R17 Azerbaijan | HARD | 670 | -0.043 | -0.011 | +0.005 | +0.021 | +0.037 | +0.054 | +0.086 |
| 2024 R17 Azerbaijan | MEDIUM | 183 | -0.035 | -0.002 | +0.014 | +0.030 | +0.046 | +0.063 | +0.095 |
| 2024 R18 Singapore | HARD | 512 | +0.009 | +0.036 | +0.049 | +0.062 | +0.076 | +0.089 | +0.116 |
| 2024 R18 Singapore | MEDIUM | 506 | +0.017 | +0.044 | +0.057 | +0.070 | +0.084 | +0.097 | +0.124 |
| 2024 R18 Singapore | SOFT | 91 | -0.017 | +0.010 | +0.023 | +0.037 | +0.050 | +0.063 | +0.090 |
| 2024 R19 United States | HARD | 513 | -0.041 | -0.012 | +0.003 | +0.017 | +0.032 | +0.047 | +0.076 |
| 2024 R19 United States | MEDIUM | 367 | -0.034 | -0.005 | +0.010 | +0.025 | +0.040 | +0.054 | +0.084 |
| 2024 R19 United States | SOFT | 3 | - | - | - | - | - | - | - |
| 2024 R20 Mexico City | HARD | 638 | -0.018 | +0.005 | +0.017 | +0.029 | +0.040 | +0.052 | +0.075 |
| 2024 R20 Mexico City | MEDIUM | 418 | +0.027 | +0.050 | +0.062 | +0.073 | +0.085 | +0.096 | +0.120 |
| 2024 R20 Mexico City | SOFT | 6 | - | - | - | - | - | - | - |
| 2024 R22 Las Vegas | HARD | 652 | -0.019 | +0.014 | +0.030 | +0.047 | +0.063 | +0.080 | +0.113 |
| 2024 R22 Las Vegas | MEDIUM | 181 | +0.087 | +0.120 | +0.136 | +0.153 | +0.169 | +0.186 | +0.219 |
| 2024 R22 Las Vegas | SOFT | 6 | - | - | - | - | - | - | - |
| 2024 R23 Qatar | HARD | 192 | -0.081 | -0.052 | -0.038 | -0.023 | -0.009 | +0.006 | +0.035 |
| 2024 R23 Qatar | MEDIUM | 445 | -0.086 | -0.058 | -0.043 | -0.029 | -0.014 | +0.000 | +0.029 |
| 2024 R23 Qatar | SOFT | 42 | +0.089 | +0.118 | +0.133 | +0.147 | +0.161 | +0.176 | +0.205 |
| 2024 R24 Abu Dhabi | HARD | 616 | -0.006 | +0.022 | +0.036 | +0.051 | +0.065 | +0.079 | +0.108 |
| 2024 R24 Abu Dhabi | MEDIUM | 243 | +0.019 | +0.047 | +0.061 | +0.076 | +0.090 | +0.104 | +0.133 |
| 2024 R24 Abu Dhabi | SOFT | 24 | - | - | - | - | - | - | - |
