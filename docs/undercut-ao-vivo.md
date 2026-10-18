# O alerta de undercut na forma ao vivo

Gerado por `python -m strategy.undercut_live` em 2026-10-15 sobre o commit `91bc335`. O protocolo está no docstring de `strategy/undercut_live.py` e foi commitado **antes** de rodar na população.

## O que é isto

O backtest (`docs/backtest.md`) mediu o alerta de undercut no instante em que o **atacante** completa a volta: a **forma antiga**. O consumidor de estratégia o calcula no instante em que o **líder** completa a volta, com um contexto só por volta: a **forma ao vivo**. São formas diferentes, e **os números do backtest não valem para a ao vivo**. Este relatório mede a ao vivo, offline, nas mesmas paradas reais.

Duas diferenças estruturais, conhecidas antes de medir: (1) quem está atrás do líder aparece com a volta anterior; (2) o líder nunca é o carro à frente de ninguém (`car_ahead` só compara carros na mesma volta, e no instante em que o líder completa a volta N os outros estão na N-1).

## Resumo

- **78 de 115 eventos** viraram uma decisão ao vivo no par do evento; 24 ficaram fora por "par diferente do evento", 13 ficaram fora por "rival é o líder (já completou a volta; o atacante ainda não)". **Nenhum undercut contra o líder é avaliado nem alertado ao vivo.**
- Dos **43** eventos primários, os alertas no limiar acertaram 10 de 12 (83%, Wilson 55% a 95%); Brier 0.145 contra 0.249 de quem prevê sempre a taxa-base. **A precisão é otimista** (só as paradas que aconteceram) e a amostra é pequena.
- A margem: viés de -0.42 s, erro absoluto de 6.25 s e o intervalo de 90% cobre 0.79 (o congelado é 1,3; os dados pedem 1.95: **não aplicado**).
- **Contra a forma antiga, nos mesmos 42 eventos:** Brier da ao vivo menos o da antiga +0.024 [+0.009, +0.043] (13 corridas): a forma ao vivo erra mais que a antiga.

## Como cada parada real virou uma decisão ao vivo

Eventos do backtest (a mesma busca e a mesma definição): **115** nas 21 corridas secas. Cada um virou uma decisão ao vivo de duas formas.

**Primária** (o resultado em vigor quando o atacante cruza a linha da volta N):

| destino do evento | eventos |
|---|---|
| no par do evento (avaliado) | 78 |
| par diferente do evento | 24 |
| rival é o líder (já completou a volta; o atacante ainda não) | 13 |

**Sensibilidade** (o resultado seguinte, antes de o atacante entrar nos boxes):

| destino do evento | eventos |
|---|---|
| no par do evento (avaliado) | 101 |
| rival é o líder (já completou a volta; o atacante ainda não) | 14 |

Quantas voltas o atacante estava atrás da volta da decisão no estado ao vivo (primária: 1 volta(s): 78; sensibilidade: 0 volta(s): 101).

## Resultado

| conjunto | eventos | primários | taxa de sucesso | prob. média | Brier | Brier (taxa-base) | alertas no limiar: acertos (Wilson) | viés da margem (s) | erro abs. (s) | cobertura 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| ao vivo, decisão no cruzamento (primária) | 78 | 43 | 0.47 | 0.43 | 0.145 | 0.249 | 10 de 12 (83%, Wilson 55% a 95%) | -0.42 | 6.25 | 0.79 |
| ao vivo, decisão mais tarde na volta (sensibilidade) | 101 | 66 | 0.42 | 0.39 | 0.154 | 0.244 | 12 de 14 (86%, Wilson 60% a 96%) | -1.33 | 6.88 | 0.79 |

Calibração da decisão primária, com o `n` de cada faixa (eventos, não corridas):

| faixa de probabilidade | eventos (n) | prevista (média) | observada |
|---|---|---|---|
| 0.0 a 0.2 | 11 | 0.09 | 0.00 |
| 0.2 a 0.4 | 8 | 0.32 | 0.38 |
| 0.4 a 0.5 | 7 | 0.44 | 0.57 |
| 0.5 a 0.6 | 5 | 0.54 | 0.60 |
| 0.6 a 0.8 | 6 | 0.68 | 0.67 |
| 0.8 a 1.0 | 6 | 0.85 | 1.00 |

Fator de inflação que os dados pedem para 90% de cobertura da margem: 1.95 (n=43); em uso, o congelado (1.30). **Diagnóstico, não aplicado.**

## Com a histerese do consumidor

O que o consumidor de fato publicaria é o alerta do `AlertTracker` (limiar 0.6 para ligar, 0,5 para desligar, passando por todas as voltas do líder). Nos eventos primários da decisão primária, o par estava **ligado** em **13**; deram certo **10** (77%, Wilson 50% a 92%).

## Contra a forma antiga, nos mesmos eventos

Os **42** eventos que são primários nas duas formas (a mesma parada, o mesmo rival, o mesmo resultado real):

| forma | eventos | Brier | alertas no limiar: acertos (Wilson) |
|---|---|---|---|
| antiga (instante do atacante) | 42 | 0.121 | 11 de 11 (100%, Wilson 74% a 100%) |
| ao vivo (instante do líder) | 42 | 0.142 | 10 de 12 (83%, Wilson 55% a 95%) |

Alertas no limiar: nas duas formas **10**, só na antiga **1**, só na ao vivo **2**. Diferença absoluta média entre as duas probabilidades: 0.048.

Brier da forma ao vivo menos o da antiga, com bootstrap sobre as corridas: **+0.024 [+0.009, +0.043] (13 corridas)**. Pela regra do backtest: **a forma ao vivo erra mais que a antiga**.

## Por corrida

| corrida | voltas do líder avaliadas | eventos | avaliados | primários | taxa de sucesso |
|---|---|---|---|---|---|
| 2024 R01 Bahrain | 52 | 12 | 8 | 2 | 0.50 |
| 2024 R02 Saudi Arabian | 45 | 0 | 0 | 0 | - |
| 2024 R03 Australian | 53 | 8 | 4 | 4 | 0.25 |
| 2024 R04 Japanese | 48 | 5 | 4 | 2 | 0.50 |
| 2024 R05 Chinese | 51 | 5 | 3 | 2 | 0.50 |
| 2024 R06 Miami | 52 | 2 | 2 | 0 | - |
| 2024 R07 Emilia Romagna | 58 | 4 | 3 | 3 | 0.67 |
| 2024 R08 Monaco | 73 | 0 | 0 | 0 | - |
| 2024 R10 Spanish | 61 | 13 | 11 | 5 | 0.40 |
| 2024 R11 Austrian | 66 | 11 | 7 | 7 | 0.29 |
| 2024 R13 Hungarian | 65 | 10 | 9 | 6 | 0.50 |
| 2024 R14 Belgian | 39 | 7 | 5 | 2 | 0.50 |
| 2024 R15 Dutch | 67 | 7 | 5 | 4 | 0.50 |
| 2024 R16 Italian | 48 | 7 | 3 | 0 | - |
| 2024 R17 Azerbaijan | 46 | 2 | 2 | 0 | - |
| 2024 R18 Singapore | 57 | 4 | 2 | 0 | - |
| 2024 R19 United States | 51 | 3 | 2 | 1 | 1.00 |
| 2024 R20 Mexico City | 66 | 2 | 2 | 2 | 0.00 |
| 2024 R22 Las Vegas | 45 | 10 | 4 | 3 | 1.00 |
| 2024 R23 Qatar | 52 | 0 | 0 | 0 | - |
| 2024 R24 Abu Dhabi | 53 | 3 | 2 | 0 | - |

## Ressalvas

- **Só paradas que aconteceram, e as equipes tendem a parar quando funciona.** A taxa de sucesso
  daqui é mais alta que a de uma situação qualquer de atacante atrás de rival: a **precisão é
  otimista**, a **revocação não é mensurável** e a calibração é **condicional** a "a equipe decidiu
  parar".
- **Uma seleção a mais.** Só entram os eventos em que o par ao vivo é o do evento. Os que ficam fora
  (principalmente o líder como rival) não são sorteados: o líder, por exemplo, nunca é o "carro à
  frente" de ninguém na forma ao vivo, então **nenhum undercut contra o líder é avaliado nem
  alertado**.
- **Amostra pequena e dependente.** São dezenas de eventos em 21 corridas, e os eventos da mesma
  corrida não são independentes (o intervalo de Wilson os trata como se fossem). Faixas com poucos
  eventos não dizem nada sobre calibração: o `n` de cada uma está na tabela.
- **O resultado real depende do que os dois lados fizeram depois.** Eventos com parada extra na
  janela são descartados; os de pista não verde ficam fora dos primários.
- **Nada foi ajustado.** O limiar (0,6 liga, 0,5 desliga) e o fator de inflação (1,3) são os
  congelados antes do backtest. O fator que os dados pedem sai só como diagnóstico.

## Ajustes pós-resultado

Nenhum. O limiar e o fator de inflação são os congelados antes do backtest.

**Análises e acréscimos feitos depois do resultado, que NÃO alteram o protocolo nem o que foi medido:**

- Depois da primeira execução acrescentei ao relatório o resumo gerado dos números e o intervalo, por bootstrap sobre as corridas, da diferença de Brier nos mesmos eventos (a primeira versão só tinha as duas colunas). Nenhum limiar, fator de inflação ou regra mudou, e nada foi rodado de novo: o relatório sai do mesmo cache.
- A razão de acertos da tabela de resultado (10 de 12) é, por coincidência, a mesma da forma antiga no backtest (15 de 18). São n e intervalos diferentes e não são a mesma medida: a comparação que vale é a da seção "Contra a forma antiga, nos mesmos eventos".
