# Backtest da Fase 3: previsão do tempo de volta, Monte Carlo e alerta de undercut

Gerado por `python -m strategy.backtest` em 2026-10-15 sobre o commit `62d1510` (21 corridas secas com leave-one-race-out e 3 fora delas; 1497 s de CPU). O protocolo foi **congelado antes** de ver o resultado (`docs/backtest-protocolo.md`).

## Resumo

**Tempo de volta.** Contra a persistência (a mediana das 3 últimas voltas limpas do piloto, sem modelo): o modelo supera em 0 dos 5 horizontes, não se distingue em 0 e é pior em 5. No horizonte de 1 volta a diferença de erro absoluto (modelo menos persistência) é 0.044 [0.024, 0.065] s.

**Probabilidade de vitória.** Contra "o líder atual vence", em 10 faixas de andamento, o Monte Carlo supera em 0, não se distingue em 8 e é pior em 2. No geral o Brier é 0.541 [0.428, 0.659] contra 0.425 [0.223, 0.664] (diferença pareada 0.116 [-0.048, 0.265]: não supera). Contra "todos iguais" (0.945 [0.943, 0.947]) o veredito é "supera".

**Variantes de k.** Nenhuma variante se distingue do encolhimento (sem diferença detectável).

**Taxa de SC** (Brier contra a taxa de 1x, pareado por corrida): taxa 0.5x: sem diferença detectável; taxa 2x: sem diferença detectável.

**Undercut ao vivo** (74 eventos primários de 115). 18 alertas no limiar de 0.6, 15 deram certo (precisão 83%, Wilson 61% a 94%); Brier 0.130 contra 0.235 de quem prevê sempre a taxa-base; cobertura do intervalo de 90% da margem: 0.81. **Precisão otimista** (só as paradas que aconteceram) e revocação não mensurável.

**Vazamento do estado por contagem de voltas.** O Brier da vitória com o estado antigo menos o do estado no instante t: -0.030 [-0.048, -0.011] (o estado antigo é mais otimista).

## Valores congelados

Congelado no commit `3be1004` (`docs/backtest-protocolo.md`), sobre o código do commit `0169cde`, **antes** de qualquer execução na população. Valores de `strategy/protocol.py`:

| parâmetro | valor |
|---|---|
| `alert_threshold` | 0.6 |
| `alert_release` | 0.5 |
| `undercut_sd_inflation` | 1.3 |
| `undercut_k_min` | 3 |
| `undercut_k_max` | 10 |
| `undercut_min_laps_left` | 5 |
| `pass_advantage_s` | 0.7 |
| `follow_gap_s` | 0.7 |
| `sc_wear` | 0.3 |
| `pit_decision_sd` | 3.0 |
| `max_stops` | 3 |
| `force_laps_left` | 3 |
| `min_stint_laps` | 5 |
| `max_sc` | 3 |
| `pseudo_events` | 4.0 |
| `start_jitter_laps` | 3 |
| `n_sims` | 5000 |
| `first_lap` | 5 |
| `every` | 5 |
| `sc_multipliers` | (0.5, 1.0, 2.0) |
| `horizons` | (1, 2, 3, 4, 5) |
| `interval_coverage` | 0.9 |
| `persistence_baseline_laps` | 3 |
| `max_silence_laps` | 2.0 |
| `bootstrap_resamples` | 2000 |
| `bootstrap_seed` | 0 |
| `confidence` | 0.95 |
| `headline_variant` | encolhimento |
| `variants` | ('fisico', 'estimado', 'encolhimento') |

## 1. Erro de previsão do tempo de volta

Voltas limpas, de seco e do mesmo stint, no instante em que o líder completa a volta N, para os horizontes de 1 a 5 voltas à frente de cada piloto. Cada célula é a média por corrida com o intervalo de 95% por bootstrap sobre corridas. "Modelo menos persistência" é pareado por corrida (negativo = o modelo erra menos).

| variante de k | horizonte | viés (s) | erro absoluto (s) | persistência: erro abs. (s) | modelo menos persistência | cobertura do intervalo de 90% |
|---|---|---|---|---|---|---|
| fisico | 1 | -0.137 [-0.204, -0.083] | 0.441 [0.388, 0.506] | 0.397 [0.355, 0.448] | 0.044 [0.024, 0.066] | 86% [84%, 87%] |
| fisico | 2 | -0.154 [-0.234, -0.087] | 0.473 [0.415, 0.538] | 0.432 [0.391, 0.478] | 0.041 [0.016, 0.070] | 86% [84%, 88%] |
| fisico | 3 | -0.160 [-0.245, -0.089] | 0.522 [0.455, 0.597] | 0.476 [0.427, 0.528] | 0.047 [0.017, 0.078] | 82% [80%, 85%] |
| fisico | 4 | -0.146 [-0.237, -0.066] | 0.565 [0.496, 0.641] | 0.497 [0.451, 0.546] | 0.069 [0.037, 0.101] | 83% [80%, 85%] |
| fisico | 5 | -0.137 [-0.256, -0.027] | 0.592 [0.509, 0.684] | 0.514 [0.458, 0.580] | 0.078 [0.036, 0.120] | 82% [79%, 86%] |
| estimado | 1 | -0.140 [-0.206, -0.088] | 0.441 [0.388, 0.505] | 0.397 [0.355, 0.448] | 0.044 [0.024, 0.066] | 86% [84%, 87%] |
| estimado | 2 | -0.157 [-0.237, -0.091] | 0.473 [0.415, 0.537] | 0.432 [0.391, 0.478] | 0.041 [0.016, 0.069] | 86% [84%, 88%] |
| estimado | 3 | -0.164 [-0.247, -0.093] | 0.522 [0.455, 0.596] | 0.476 [0.427, 0.528] | 0.046 [0.017, 0.078] | 83% [80%, 85%] |
| estimado | 4 | -0.149 [-0.239, -0.070] | 0.564 [0.496, 0.640] | 0.497 [0.451, 0.546] | 0.068 [0.037, 0.099] | 83% [80%, 85%] |
| estimado | 5 | -0.140 [-0.259, -0.030] | 0.591 [0.509, 0.684] | 0.514 [0.458, 0.580] | 0.078 [0.037, 0.118] | 82% [79%, 85%] |
| encolhimento | 1 | -0.140 [-0.206, -0.087] | 0.441 [0.388, 0.505] | 0.397 [0.355, 0.448] | 0.044 [0.024, 0.065] | 86% [84%, 87%] |
| encolhimento | 2 | -0.157 [-0.237, -0.090] | 0.473 [0.415, 0.537] | 0.432 [0.391, 0.478] | 0.041 [0.016, 0.070] | 86% [84%, 88%] |
| encolhimento | 3 | -0.163 [-0.247, -0.093] | 0.522 [0.455, 0.596] | 0.476 [0.427, 0.528] | 0.046 [0.017, 0.077] | 83% [80%, 85%] |
| encolhimento | 4 | -0.149 [-0.239, -0.070] | 0.565 [0.496, 0.640] | 0.497 [0.451, 0.546] | 0.068 [0.037, 0.100] | 83% [80%, 85%] |
| encolhimento | 5 | -0.140 [-0.258, -0.030] | 0.592 [0.509, 0.684] | 0.514 [0.458, 0.580] | 0.078 [0.037, 0.118] | 82% [79%, 85%] |

**Na variante principal (encolhimento) o modelo NÃO supera a persistência em nenhum horizonte** (o intervalo da diferença contém o zero ou fica acima).

## 2. Probabilidade de vitória: Brier por andamento da corrida

O Brier da vitória é a soma, sobre os pilotos, de (p - y)^2: 0 é perfeito e 2, o pior. Brier por faixa de andamento da corrida; a unidade do intervalo é a **corrida**, não a volta.

| andamento | corridas | Monte Carlo | o líder vence | todos iguais | MC − líder | contra o líder | MC − iguais | contra todos iguais | pódio (MC) |
|---|---|---|---|---|---|---|---|---|---|
| 0% a 10% | 16 | 0.741 [0.553, 0.915] | 0.625 [0.125, 1.125] | 0.936 [0.910, 0.949] | 0.116 [-0.250, 0.484] | não supera | -0.195 [-0.386, -0.019] | supera | 1.281 [1.037, 1.573] |
| 10% a 20% | 21 | 0.778 [0.636, 0.916] | 0.571 [0.190, 0.952] | 0.948 [0.946, 0.949] | 0.207 [-0.166, 0.545] | não supera | -0.169 [-0.312, -0.032] | supera | 1.845 [1.602, 2.070] |
| 20% a 30% | 21 | 0.781 [0.654, 0.907] | 0.714 [0.381, 1.095] | 0.947 [0.945, 0.949] | 0.067 [-0.295, 0.435] | não supera | -0.166 [-0.293, -0.040] | supera | 1.735 [1.479, 2.014] |
| 30% a 40% | 19 | 0.698 [0.566, 0.837] | 0.632 [0.263, 1.053] | 0.947 [0.945, 0.948] | 0.066 [-0.265, 0.353] | não supera | -0.249 [-0.381, -0.109] | supera | 1.489 [1.240, 1.786] |
| 40% a 50% | 21 | 0.629 [0.500, 0.749] | 0.429 [0.095, 0.763] | 0.946 [0.945, 0.948] | 0.200 [-0.105, 0.463] | não supera | -0.318 [-0.447, -0.197] | supera | 1.310 [1.068, 1.578] |
| 50% a 60% | 21 | 0.590 [0.436, 0.742] | 0.381 [0.095, 0.762] | 0.946 [0.944, 0.948] | 0.209 [-0.129, 0.482] | não supera | -0.356 [-0.509, -0.205] | supera | 1.151 [0.856, 1.491] |
| 60% a 70% | 21 | 0.442 [0.293, 0.614] | 0.429 [0.143, 0.762] | 0.946 [0.944, 0.947] | 0.014 [-0.293, 0.287] | não supera | -0.503 [-0.651, -0.331] | supera | 1.019 [0.683, 1.389] |
| 70% a 80% | 19 | 0.367 [0.196, 0.578] | 0.211 [0.000, 0.526] | 0.945 [0.943, 0.947] | 0.156 [-0.077, 0.345] | não supera | -0.578 [-0.748, -0.368] | supera | 0.800 [0.458, 1.231] |
| 80% a 90% | 20 | 0.351 [0.158, 0.583] | 0.200 [0.000, 0.500] | 0.945 [0.943, 0.947] | 0.151 [0.002, 0.314] | é pior | -0.594 [-0.786, -0.364] | supera | 0.536 [0.195, 0.976] |
| 90% a 100% | 21 | 0.180 [0.034, 0.393] | 0.095 [0.000, 0.286] | 0.945 [0.943, 0.947] | 0.085 [0.014, 0.175] | é pior | -0.765 [-0.910, -0.552] | supera | 0.197 [0.045, 0.432] |

Contra **o líder atual vence**, o Monte Carlo NÃO SUPERA em 8 de 10 faixas (0%-10%, 10%-20%, 20%-30%, 30%-40%, 40%-50%, 50%-60%, 60%-70%, 70%-80%); É PIOR em 2 de 10 faixas (80%-90%, 90%-100%). SUPERA = o intervalo de 95% da diferença pareada de Brier fica inteiro abaixo de zero; NÃO SUPERA = o intervalo contém o zero (não se distingue do comparador nesta amostra); É PIOR = fica inteiro acima.

Contra **todos iguais**, o Monte Carlo SUPERA em 10 de 10 faixas (0%-10%, 10%-20%, 20%-30%, 30%-40%, 40%-50%, 50%-60%, 60%-70%, 70%-80%, 80%-90%, 90%-100%). SUPERA = o intervalo de 95% da diferença pareada de Brier fica inteiro abaixo de zero; NÃO SUPERA = o intervalo contém o zero (não se distingue do comparador nesta amostra); É PIOR = fica inteiro acima.

|  | Brier da vitória por corrida (todas as voltas avaliadas) |
|---|---|
| Monte Carlo | 0.541 [0.428, 0.659] |
| o líder atual vence | 0.425 [0.223, 0.664] |
| todos iguais | 0.945 [0.943, 0.947] |
| Monte Carlo menos o líder (pareado) | 0.116 [-0.048, 0.265] |
| Monte Carlo menos todos iguais (pareado) | -0.404 [-0.516, -0.287] |

O favorito do Monte Carlo ganhou em 0.61 [0.47, 0.73] das previsões (média por corrida).

## 3. Curva de calibração

Probabilidade prevista contra a frequência observada, **com o n de cada faixa**. Os pares de uma mesma corrida e volta não são independentes (as probabilidades somam 1) nem as voltas entre si; por isso a coluna "corridas" é o tamanho efetivo e o intervalo da frequência ressorteia corridas. Uma faixa com poucas corridas não diz nada sobre calibração, qualquer que seja a frequência observada.

**Vitória**

| faixa de probabilidade | pares (piloto, volta) | corridas | prevista (média) | observada [IC 95%] |
|---|---|---|---|---|
| 0.00 a 0.02 | 2948 | 21 | 0.003 | 0% |
| 0.02 a 0.05 | 511 | 21 | 0.033 | 2% [1%, 3%] |
| 0.05 a 0.10 | 389 | 21 | 0.072 | 6% [4%, 11%] |
| 0.10 a 0.20 | 285 | 21 | 0.139 | 15% [11%, 19%] |
| 0.20 a 0.35 | 117 | 20 | 0.261 | 30% [23%, 37%] |
| 0.35 a 0.50 | 68 | 20 | 0.412 | 50% [36%, 64%] |
| 0.50 a 0.70 | 43 | 19 | 0.588 | 58% [38%, 79%] |
| 0.70 a 0.90 | 34 | 15 | 0.802 | 82% [57%, 100%] |
| 0.90 a 1.00 | 36 | 18 | 0.970 | 97% [90%, 100%] |

**Pódio**

| faixa de probabilidade | pares (piloto, volta) | corridas | prevista (média) | observada [IC 95%] |
|---|---|---|---|---|
| 0.00 a 0.02 | 2107 | 21 | 0.003 | 0% |
| 0.02 a 0.05 | 388 | 21 | 0.033 | 2% [0%, 3%] |
| 0.05 a 0.10 | 400 | 21 | 0.073 | 2% [0%, 3%] |
| 0.10 a 0.20 | 449 | 21 | 0.146 | 11% [7%, 17%] |
| 0.20 a 0.35 | 355 | 21 | 0.264 | 33% [29%, 38%] |
| 0.35 a 0.50 | 210 | 21 | 0.419 | 50% [44%, 57%] |
| 0.50 a 0.70 | 169 | 21 | 0.600 | 69% [62%, 78%] |
| 0.70 a 0.90 | 145 | 21 | 0.804 | 81% [71%, 92%] |
| 0.90 a 1.00 | 208 | 21 | 0.966 | 91% [78%, 100%] |

## 4. As variantes de k

A regra do protocolo: uma variante só é declarada melhor se o intervalo de 95% da diferença pareada por corrida exclui o zero, **no mesmo sentido**, no erro de tempo de volta **e** no Brier da vitória; do contrário, "sem diferença detectável" e o principal permanece. Diferença negativa = a variante erra menos que o principal.

| variante | Brier da vitória | erro de tempo de volta (s) | Brier − principal | erro − principal | conclusão |
|---|---|---|---|---|---|
| fisico | 0.542 [0.427, 0.664] | 0.514 [0.451, 0.585] | 0.001 [-0.006, 0.007] | 0.000 [-0.001, 0.001] | sem diferença detectável |
| estimado | 0.542 [0.431, 0.661] | 0.514 [0.452, 0.584] | 0.001 [-0.000, 0.003] | -0.000 | sem diferença detectável |
| encolhimento (principal) | 0.541 [0.428, 0.659] | 0.514 [0.452, 0.584] | - | - | - |

- `fisico`: sem diferença detectável.
- `estimado`: sem diferença detectável.

## 5. Sensibilidade à taxa de Safety Car

A mesma simulação (os **mesmos números aleatórios**) com a taxa de SC multiplicada. O Brier por faixa de andamento (5 faixas) e o geral:

| andamento | taxa de SC 0.5x | taxa de SC 1x | taxa de SC 2x |
|---|---|---|---|
| 0% a 20% | 0.779 [0.640, 0.923] | 0.777 [0.643, 0.915] | 0.776 [0.649, 0.905] |
| 20% a 40% | 0.743 [0.622, 0.861] | 0.748 [0.630, 0.862] | 0.756 [0.644, 0.863] |
| 40% a 60% | 0.605 [0.467, 0.741] | 0.609 [0.473, 0.743] | 0.616 [0.481, 0.746] |
| 60% a 80% | 0.395 [0.251, 0.571] | 0.396 [0.253, 0.572] | 0.397 [0.254, 0.572] |
| 80% a 100% | 0.267 [0.118, 0.451] | 0.267 [0.118, 0.451] | 0.265 [0.117, 0.448] |
| todas as voltas | 0.539 [0.425, 0.659] | 0.541 [0.428, 0.659] | 0.543 [0.434, 0.658] |

- taxa 0.5x menos 1x (Brier, pareado): -0.002 [-0.003, 0.000]
- taxa 2x menos 1x (Brier, pareado): 0.002 [-0.001, 0.006]

Se os intervalos se sobrepõem e a diferença pareada contém o zero, o resultado é **insensível** à taxa de SC nesta amostra (com 8 períodos de SC em 21 corridas, é o que se espera: a maioria das corridas não tem SC depois da volta 10).

## 6. O alerta de undercut nas paradas reais

**A ressalva do viés, antes dos números.** Só se observam as paradas que **aconteceram**, e as equipes tendem a parar quando o undercut funciona. Por isso (1) a taxa de sucesso entre as paradas observadas é maior que a de uma situação qualquer de atacante atrás de rival, (2) a **precisão é otimista** e a **revocação não é mensurável** (o denominador, as situações em que funcionaria e ninguém parou, não existe) e (3) a calibração é **condicional** a "a equipe decidiu parar". O resultado real também depende do que os dois lados fizeram depois.

Eventos descartados na busca (21 corridas): horizonte fora do intervalo: 159, rival em outro stint: 151, volta inicial: 97, rival não parou: 78, paradas na mesma volta: 65, sem rival à frente: 50, parada extra na janela: 6, sem relógio no fim: 1.

### Ao vivo (horizonte e composto previstos; o resultado principal)

Eventos: 115; primários (sem regra dura de não alertar e sem pista não verde): 74. Alertas no limiar de 0.6: 18 alertas, 15 deram certo (precisão 83%, intervalo de Wilson 61% a 94%).

| conjunto | eventos | primários | taxa de sucesso | prob. média | Brier | Brier (taxa-base) | alertas | precisão | viés da margem (s) | erro abs. (s) | cobertura 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fator congelado (1,3) | 115 | 74 | 0.38 | 0.37 | 0.130 | 0.235 | 18 | 0.83 | -1.03 | 6.39 | 0.81 |
| diagnóstico: fator calibrado nas outras corridas | 115 | 74 | 0.38 | 0.39 | 0.140 | 0.235 | 12 | 0.92 | -1.03 | 6.39 | 0.88 |

Calibração (fator congelado), com o n de cada faixa (eventos, não corridas):

| faixa de probabilidade | eventos | prevista (média) | observada |
|---|---|---|---|
| 0.0 a 0.2 | 25 | 0.06 | 0.00 |
| 0.2 a 0.4 | 14 | 0.31 | 0.36 |
| 0.4 a 0.5 | 10 | 0.43 | 0.30 |
| 0.5 a 0.6 | 7 | 0.57 | 0.71 |
| 0.6 a 0.8 | 13 | 0.68 | 0.77 |
| 0.8 a 1.0 | 5 | 0.87 | 1.00 |

O fator de inflação que os dados pedem para 90% de cobertura da margem: 1.87 (n=74); a cobertura do intervalo com o fator congelado: 0.81.

### Oráculo (horizonte e compostos reais; mede a física do modelo)

Eventos: 115; primários (sem regra dura de não alertar e sem pista não verde): 76. Alertas no limiar de 0.6: 24 alertas, 16 deram certo (precisão 67%, intervalo de Wilson 47% a 82%).

| conjunto | eventos | primários | taxa de sucesso | prob. média | Brier | Brier (taxa-base) | alertas | precisão | viés da margem (s) | erro abs. (s) | cobertura 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fator congelado (1,3) | 115 | 76 | 0.37 | 0.43 | 0.160 | 0.233 | 24 | 0.67 | 0.57 | 6.75 | 0.89 |
| diagnóstico: fator calibrado nas outras corridas | 115 | 76 | 0.37 | 0.43 | 0.160 | 0.233 | 23 | 0.70 | 0.57 | 6.75 | 0.89 |

Calibração (fator congelado), com o n de cada faixa (eventos, não corridas):

| faixa de probabilidade | eventos | prevista (média) | observada |
|---|---|---|---|
| 0.0 a 0.2 | 21 | 0.09 | 0.05 |
| 0.2 a 0.4 | 13 | 0.30 | 0.15 |
| 0.4 a 0.5 | 9 | 0.45 | 0.44 |
| 0.5 a 0.6 | 9 | 0.53 | 0.56 |
| 0.6 a 0.8 | 16 | 0.69 | 0.50 |
| 0.8 a 1.0 | 8 | 0.90 | 1.00 |

O fator de inflação que os dados pedem para 90% de cobertura da margem: 1.31 (n=76); a cobertura do intervalo com o fator congelado: 0.89.


## 7. Diagnóstico do vazamento: o estado por instante contra o por contagem de voltas

O estado antigo dava a cada piloto a volta N DELE, o que dá ao retardatário informação de depois do instante em que o líder completa a volta. Aqui as duas versões, sobre as mesmas corridas e voltas, mostram o quanto esse vazamento flatearia o backtest. **Diferença negativa = o estado antigo é mais otimista.** O resultado do relatório é o da coluna do instante t.

| medida | estado no instante t (o resultado) | estado por contagem de voltas (diagnóstico) | diferença pareada (contagem − instante) |
|---|---|---|---|
| Brier da vitória | 0.541 [0.428, 0.659] | 0.511 [0.410, 0.620] | -0.030 [-0.048, -0.011] |
| erro do tempo de volta (s) | 0.514 [0.452, 0.584] | 0.506 [0.441, 0.579] | -0.008 [-0.019, 0.003] |
| undercut: Brier (eventos primários) | 0.130 | 0.145 | - |
| undercut: alertas / precisão | 18 / 0.83 | 14 / 0.93 | - |

## 8. A britânica, o Canadá e o Brasil, à parte

A corrida britânica (pista secando), o Canadá e o Brasil (molhadas) **não entram nas médias das secas**. O modelo só conhece compostos de seco: pilotos de intermediário ou de chuva ficam sem modelo e, se vencerem, contam como previsão com probabilidade 0. **Uma corrida cada: sem intervalo de confiança, só os pontos**, que descrevem o que acontece fora do domínio do modelo, não o desempenho dele.

| corrida | tipo | previsões | Brier MC | líder vence | todos iguais | tempo de volta h=1 (s) | persistência h=1 (s) | eventos de undercut |
|---|---|---|---|---|---|---|---|---|
| 2024 R09 Canadian | molhada | 5 | 1.158 | 0.200 | 1.016 | 1.490 | 1.623 | 2 (0 primários) |
| 2024 R12 British | mista | 8 | 0.647 | 1.250 | 0.942 | 1.354 | 1.118 | 0 (0 primários) |
| 2024 R21 São Paulo | molhada | 0 | - | - | - | - | - | 0 (0 primários) |

## Limitações

- **O SC depois da volta 10 vem de 3 eventos** (8 períodos de SC em 7 das 21 corridas secas, 5 deles
  nas voltas 1 a 7). As paradas sob SC são 15 e a razão de custo (0,58) tem desvio de ~0,4.
- **As regras de parada, de ultrapassagem e o desgaste sob SC são heurísticas**, não medidas:
  equipes decidem por rivais, tráfego e estratégia que o ganho de tempo não vê.
- **VSC (8 períodos) e bandeira vermelha (2) ficam fora do simulador**, e abandono não causa SC.
- **Vinte e uma corridas** é uma amostra pequena: os intervalos de bootstrap são largos, e o poder
  para distinguir variantes ou comparadores é baixo.
- **Uma só temporada** (2024) e um só conjunto de regras: nada garante que o resultado vale para
  2025.
- **O viés do undercut** (só as paradas que aconteceram): a precisão é otimista e a revocação não é
  mensurável.
- **As constantes (perda no pit, SC, abandonos, trocas)** saem das outras 20 corridas, não só das
  anteriores no calendário, o que ao vivo não se saberia.
- **O estado por instante** exclui quem não cruza a linha há mais de duas voltas típicas
  (heurística).

## Ajustes pós-resultado

Nenhum. Nenhum valor do protocolo foi alterado depois de ver o resultado.

**Análises e correções feitas depois do resultado, que NÃO alteram o protocolo nem o que foi medido:**

- O cache do backtest era gravado com as classes como `__main__.RaceOutcome` e não podia ser relido por outro processo (defeito do ponto de entrada, corrigido). Os números não mudam: o cache foi convertido e o backtest não foi rodado de novo.
- `docs/backtest-diagnostico.md`: onde está a hesitação do Monte Carlo, desligando um componente de incerteza por vez nos mesmos números aleatórios. Desligar a incerteza do modelo o deixa super-confiante sem melhorar o acerto do favorito, e a projeção quase determinística acerta o vencedor menos que o líder atual: a fraqueza está no ponto do modelo, não na dispersão. É um diagnóstico: nenhuma correção foi aplicada.
