# Qualidade dos dados de voltas

Gerado por `python -m strategy.quality --markdown docs/qualidade-dos-dados.md` em 2026-10-14, sobre 24 corridas. É uma inspeção dos dados **brutos**: não decide o que excluir (isso é da limpeza das voltas, `strategy/cleaning.py`).

## Como ler

**Composição e cobertura**

- **posições**: `sim` = ingestão completa (tem a trajetória); `não` = ingerida com `--laps-only`.
- **seco**: % das voltas em SOFT, MEDIUM ou HARD. Corrida com pouco "seco" não serve para modelar o
  desgaste do pneu de pista seca.
- **sem tempo**: % das voltas com tempo de volta nulo.
- **box entrada / saída (com tempo)**: quantas voltas terminam / começam nos boxes e, entre
  parênteses, quantas delas **têm** tempo de volta. É a armadilha: um modelo que só descartasse
  "voltas sem tempo" manteria essas voltas lentas na regressão.
- **SC, VSC, vermelha, amarela**: voltas em que o código de pista ocorreu, mesmo que por parte da
  volta (uma volta pode contar em mais de uma coluna).
- **pista livre**: % das voltas com status exatamente `1`.

**Inconsistências estruturais** (esperado: 0)

- **buracos_de_volta**: voltas que faltam no meio da sequência de um piloto (abandono não conta).
- **idade_do_pneu_pula**: a idade do pneu não subiu 1 entre voltas seguidas do mesmo stint.
- **composto_muda_no_stint**: o composto mudou sem o stint mudar.
- **relogio_nao_cresce**: o relógio da sessão no fim da volta não cresceu de uma volta para outra.
- **relogio_vs_tempo_de_volta**: o tempo de volta não bate (tolerância de 2 ms) com a diferença
  entre os relógios de duas voltas seguidas.
- **posicao_vs_chegada**: voltas em que a ordem por `position` discorda da ordem de chegada à linha.
- **voltas_lentas_acima_de_125pct**: voltas acima de 125% da mediana da corrida (medida de sujeira
  por tráfego, SC ou erro, não regra de exclusão).

**Resultado oficial** (a "verdade" do backtest)

- **sem_posicao_final / posicao_final_repetida / posicao_final_com_buraco**: a ordem de chegada
  tem de ser 1..N sem faltar, repetir nem pular.
- **sem_status**: pilotos sem `status`. O resultado oficial vem de uma API externa (Ergast/Jolpica);
  quando ela não responde, o FastF1 **calcula** a ordem a partir das voltas, que sai completa mas
  sem status. Acima de 0 = o resultado gravado é o calculado, e reingerir a corrida o corrige.
- **final_vs_ultima_volta**: o resultado oficial discorda da posição na última volta. É esperado
  para quem abandonou e para penalidades ou desqualificações pós-corrida (medida, não erro).

**Desconhecidos**: `pit_in`, `pit_out`, `track_status` e `session_time_ms` nulos são voltas gravadas
antes de existirem essas colunas, que a limpeza trata como "não sei" (e descarta).

## Composição e cobertura

| corrida | voltas | pilotos | posições | seco | sem tempo | box entrada (com tempo) | box saída (com tempo) | SC | VSC | vermelha | amarela | pista livre |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | 1129 | 20 | não | 100% | 0% | 43 (42) | 43 (42) | 0 | 0 | 0 | 42 | 96% |
| 2024 R02 Saudi Arabian | 901 | 20 | não | 100% | 3% | 20 (20) | 19 (5) | 55 | 0 | 0 | 33 | 92% |
| 2024 R03 Australian | 998 | 19 | não | 100% | 0% | 37 (36) | 37 (37) | 0 | 52 | 0 | 40 | 94% |
| 2024 R04 Japanese | 907 | 20 | não | 100% | 3% | 55 (55) | 54 (36) | 0 | 0 | 20 | 42 | 95% |
| 2024 R05 Chinese | 1032 | 20 | não | 100% | 2% | 41 (39) | 41 (34) | 167 | 56 | 0 | 57 | 78% |
| 2024 R06 Miami | 1111 | 20 | sim | 100% | 0% | 28 (28) | 28 (28) | 96 | 33 | 0 | 20 | 88% |
| 2024 R07 Emilia Romagna | 1238 | 20 | não | 100% | 0% | 28 (27) | 28 (27) | 0 | 0 | 0 | 0 | 100% |
| 2024 R08 Monaco | 1237 | 20 | não | 100% | 1% | 23 (23) | 23 (16) | 36 | 0 | 20 | 38 | 96% |
| 2024 R09 Canadian | 1272 | 20 | não | 33% | 0% | 45 (42) | 44 (44) | 186 | 0 | 0 | 187 | 75% |
| 2024 R10 Spanish | 1310 | 20 | sim | 100% | 0% | 42 (42) | 43 (43) | 0 | 0 | 0 | 0 | 100% |
| 2024 R11 Austrian | 1405 | 20 | não | 100% | 0% | 46 (46) | 46 (46) | 0 | 23 | 0 | 21 | 97% |
| 2024 R12 British | 960 | 19 | não | 76% | 0% | 46 (46) | 46 (46) | 0 | 0 | 0 | 0 | 100% |
| 2024 R13 Hungarian | 1355 | 20 | sim | 100% | 0% | 41 (41) | 41 (41) | 0 | 0 | 0 | 21 | 98% |
| 2024 R14 Belgian | 841 | 20 | não | 100% | 0% | 35 (35) | 34 (34) | 0 | 0 | 0 | 19 | 98% |
| 2024 R15 Dutch | 1426 | 20 | não | 100% | 0% | 26 (26) | 27 (27) | 0 | 0 | 0 | 0 | 100% |
| 2024 R16 Italian | 1008 | 20 | sim | 100% | 0% | 31 (31) | 30 (30) | 0 | 0 | 0 | 0 | 100% |
| 2024 R17 Azerbaijan | 973 | 20 | não | 100% | 2% | 24 (24) | 24 (20) | 0 | 33 | 0 | 48 | 94% |
| 2024 R18 Singapore | 1177 | 20 | não | 100% | 0% | 25 (24) | 23 (23) | 0 | 0 | 0 | 0 | 100% |
| 2024 R19 United States | 1059 | 20 | não | 100% | 3% | 23 (23) | 24 (23) | 58 | 0 | 0 | 95 | 87% |
| 2024 R20 Mexico City | 1215 | 20 | não | 100% | 0% | 22 (22) | 22 (22) | 110 | 0 | 0 | 20 | 91% |
| 2024 R21 São Paulo | 1134 | 18 | sim | 0% | 3% | 36 (17) | 35 (16) | 114 | 33 | 18 | 213 | 71% |
| 2024 R22 Las Vegas | 938 | 20 | não | 100% | 0% | 41 (41) | 40 (40) | 0 | 0 | 0 | 0 | 100% |
| 2024 R23 Qatar | 943 | 20 | não | 100% | 2% | 61 (48) | 60 (47) | 208 | 16 | 0 | 74 | 74% |
| 2024 R24 Abu Dhabi | 1035 | 20 | não | 100% | 0% | 30 (30) | 28 (28) | 0 | 38 | 0 | 83 | 90% |

## Inconsistências, resultado oficial e valores desconhecidos

| corrida | buracos_de_volta | idade_do_pneu_pula | composto_muda_no_stint | relogio_nao_cresce | relogio_vs_tempo_de_volta | posicao_vs_chegada | voltas_lentas_acima_de_125pct | sem_posicao_final | sem_status | posicao_final_repetida | posicao_final_com_buraco | final_vs_ultima_volta | pit_in nulo | pit_out nulo | track_status nulo | session_time_ms nulo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R02 Saudi Arabian | 0 | 0 | 0 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| 2024 R03 Australian | 0 | 0 | 0 | 0 | 0 | 0 | 34 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| 2024 R04 Japanese | 0 | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R05 Chinese | 0 | 0 | 0 | 0 | 0 | 0 | 134 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 2024 R06 Miami | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |
| 2024 R07 Emilia Romagna | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R08 Monaco | 0 | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R09 Canadian | 0 | 0 | 0 | 0 | 0 | 0 | 135 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 2024 R10 Spanish | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R11 Austrian | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| 2024 R12 British | 0 | 0 | 0 | 0 | 0 | 0 | 38 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 2024 R13 Hungarian | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 2024 R14 Belgian | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 |
| 2024 R15 Dutch | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 2024 R16 Italian | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| 2024 R17 Azerbaijan | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R18 Singapore | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R19 United States | 0 | 0 | 0 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 2024 R20 Mexico City | 0 | 0 | 0 | 0 | 0 | 0 | 112 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 2024 R21 São Paulo | 0 | 0 | 0 | 0 | 0 | 0 | 105 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| 2024 R22 Las Vegas | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 2024 R23 Qatar | 0 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2024 R24 Abu Dhabi | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
