# A verdade do backtest: ordem na pista, com o oficial ao lado

Gerado por `python -m strategy.results --markdown docs/verdade-do-backtest.md` em 2026-10-14, sobre 24 corridas. A regra está em `strategy/results.py`.

## Resumo

- O vencedor é o mesmo em 23 de 24 corridas (difere em: 2024 R14 Belgian).
- O pódio (três pilotos, na ordem) é o mesmo em 22 de 24 (difere em: 2024 R14 Belgian, 2024 R19 United States).

## A decisão

O backtest mede a previsão de vitória e de pódio contra a **ordem na pista**. O **resultado
oficial é reportado ao lado**, nunca no lugar.

- **Ordem na pista**: mais voltas completas primeiro; empatando, quem cruzou a linha antes (a mesma
  regra da classificação do replay). Quem não largou não tem posição na pista.
- **Por que**: o modelo simula a corrida na pista (ritmo, pneu, paradas, safety car).
  Desqualificações e acréscimos de tempo aplicados depois da bandeira são decididos fora da pista e
  nada que se saiba na volta N os prevê: medir contra o oficial puniria o modelo por algo que ele
  não tem como saber. Penalidades cumpridas durante a corrida já estão nos tempos de volta.
- **O caso que expôs o problema**: Bélgica 2024. O Russell venceu na pista e consta como
  `Disqualified`; com o oficial como verdade, prever a vitória dele (certa na pista) contaria como
  erro, e todas as posições dos outros pilotos se deslocam.
- **Como ler a tabela**: `posições que diferem` compara só quem aparece nas duas ordens (quem não
  largou não tem posição na pista). `observações` lista os status que não são de rotina, onde o
  oficial pode divergir da pista.

## Corrida a corrida

| corrida | vencedor na pista | vencedor oficial | pódio igual | posições que diferem | observações (status incomuns) |
|---|---|---|---|---|---|
| 2024 R01 Bahrain | VER | VER | sim | 0 de 20 | - |
| 2024 R02 Saudi Arabian | VER | VER | sim | 4 de 20 | - |
| 2024 R03 Australian | SAI | SAI | sim | 9 de 19 | - |
| 2024 R04 Japanese | VER | VER | sim | 2 de 20 | - |
| 2024 R05 Chinese | VER | VER | sim | 2 de 20 | - |
| 2024 R06 Miami | NOR | NOR | sim | 9 de 20 | - |
| 2024 R07 Emilia Romagna | VER | VER | sim | 0 de 20 | - |
| 2024 R08 Monaco | LEC | LEC | sim | 4 de 20 | - |
| 2024 R09 Canadian | VER | VER | sim | 0 de 20 | - |
| 2024 R10 Spanish | VER | VER | sim | 0 de 20 | - |
| 2024 R11 Austrian | RUS | RUS | sim | 2 de 20 | - |
| 2024 R12 British | HAM | HAM | sim | 0 de 19 | GAS: Did not start |
| 2024 R13 Hungarian | PIA | PIA | sim | 0 de 20 | - |
| 2024 R14 Belgian | RUS | HAM | NÃO | 20 de 20 | RUS: Disqualified |
| 2024 R15 Dutch | NOR | NOR | sim | 2 de 20 | - |
| 2024 R16 Italian | LEC | LEC | sim | 4 de 20 | - |
| 2024 R17 Azerbaijan | PIA | PIA | sim | 4 de 20 | - |
| 2024 R18 Singapore | NOR | NOR | sim | 0 de 20 | - |
| 2024 R19 United States | LEC | LEC | NÃO | 2 de 20 | - |
| 2024 R20 Mexico City | SAI | SAI | sim | 0 de 20 | - |
| 2024 R21 São Paulo | VER | VER | sim | 2 de 18 | ALB: Did not start; HUL: Disqualified; STR: Did not start |
| 2024 R22 Las Vegas | RUS | RUS | sim | 0 de 20 | - |
| 2024 R23 Qatar | VER | VER | sim | 0 de 20 | - |
| 2024 R24 Abu Dhabi | NOR | NOR | sim | 3 de 20 | - |
