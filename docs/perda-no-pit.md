# Perda no pit e constantes por corrida

Gerado por `python -m strategy.race_stats --markdown docs/perda-no-pit.md` em 2026-10-14. As regras estão em `strategy/pitloss.py` e `strategy/priors.py`.

## Resumo

- 21 corridas; 511 paradas em bandeira verde.
- Perda no pit (mediana por corrida): mediana 22.5 s, de 18.8 a 29.8; a variação ENTRE corridas (desvio robusto) é 2.1 s e a dispersão entre paradas dentro de uma corrida, em mediana, 1.4 s.
- Paradas FORA da bandeira verde (safety car, VSC, vermelha, amarela): 24, mediana 30.2 s contra 22.7 s das verdes. Não saíram mais baratas; ver a nota abaixo sobre por que isso não responde à pergunta.
- Com menos de 10 paradas verdes: 2024 R02 Saudi Arabian, 2024 R08 Monaco, 2024 R23 Qatar. É nelas que o prior pesa.
- τ mediano 0.022; diferença entre compostos mediana 0.017.
- Valores de `DEFAULT_PRIORS` (todas as corridas dentro): perda 22.5 s, variação entre corridas 2.1 s, dispersão entre paradas 1.4 s, τ 0.022, piso entre compostos 0.017.

## Como ler

- **Paradas verdes**: a volta de entrada e a de saída têm status exatamente `1`. A perda é
  `(entrada + saída) - 2 x ritmo normal`, com o ritmo normal vindo das voltas limpas antes e depois.
- **Reserva SEM esta corrida**: a mediana das perdas das **outras** corridas (leave-one-race-out).
  Com `earlier_only` só entram as anteriores no calendário, o que se saberia ao vivo.
- **Edições anteriores do circuito**: com o banco só de 2024 não há nenhuma, então o caminho "mesma
  pista em edições anteriores" não tem o que devolver e vale a mediana das outras corridas.
- **Paradas fora da bandeira verde** (safety car, VSC, vermelha, amarela): ficam fora da estimativa.
  Medidas contra o ritmo normal do próprio piloto elas **não** saem mais baratas (ver o resumo), mas
  a amostra é pequena e enviesada, porque só entram paradas com voltas verdes ao redor para servir
  de referência; e "custa menos" sob safety car é uma afirmação **relativa ao pelotão** (todos andam
  devagar), que esta medida não captura.

## Por corrida

| corrida | circuito | paradas verdes | perda no pit (mediana, s) | dispersão entre paradas (s) | τ | diferença entre compostos | reserva da perda SEM esta corrida (s) | edições anteriores do circuito |
|---|---|---|---|---|---|---|---|---|
| 2024 R01 Bahrain | Sakhir | 37 | 24.2 | 0.8 | 0.022 | 0.013 | 22.4 | 0 |
| 2024 R02 Saudi Arabian | Jeddah | 5 | 22.0 | 2.1 | 0.018 | 0.043 | 22.8 | 0 |
| 2024 R03 Australian | Melbourne | 29 | 20.7 | 1.4 | 0.018 | 0.010 | 22.8 | 0 |
| 2024 R04 Japanese | Suzuka | 32 | 22.5 | 1.0 | 0.073 | 0.041 | 22.7 | 0 |
| 2024 R05 Chinese | Shanghai | 20 | 23.9 | 0.8 | 0.027 | 0.025 | 22.4 | 0 |
| 2024 R06 Miami | Miami | 10 | 20.5 | 2.0 | 0.024 | 0.007 | 22.8 | 0 |
| 2024 R07 Emilia Romagna | Imola | 25 | 28.3 | 1.4 | 0.019 | 0.020 | 22.4 | 0 |
| 2024 R08 Monaco | Monaco | 5 | 21.1 | 1.6 | 0.017 | 0.011 | 22.8 | 0 |
| 2024 R10 Spanish | Barcelona | 42 | 23.1 | 1.4 | 0.028 | 0.007 | 22.4 | 0 |
| 2024 R11 Austrian | Spielberg | 41 | 20.8 | 1.2 | 0.015 | 0.017 | 22.8 | 0 |
| 2024 R13 Hungarian | Budapest | 38 | 20.4 | 1.4 | 0.025 | 0.010 | 22.8 | 0 |
| 2024 R14 Belgian | Spa-Francorchamps | 31 | 18.8 | 0.9 | 0.039 | 0.013 | 22.8 | 0 |
| 2024 R15 Dutch | Zandvoort | 26 | 23.1 | 1.5 | 0.013 | 0.012 | 22.4 | 0 |
| 2024 R16 Italian | Monza | 29 | 25.6 | 1.0 | 0.024 | 0.019 | 22.4 | 0 |
| 2024 R17 Azerbaijan | Baku | 15 | 21.6 | 1.4 | 0.025 | 0.004 | 22.8 | 0 |
| 2024 R18 Singapore | Marina Bay | 22 | 29.8 | 2.0 | 0.020 | 0.021 | 22.4 | 0 |
| 2024 R19 United States | Austin | 21 | 21.4 | 1.9 | 0.028 | 0.007 | 22.8 | 0 |
| 2024 R20 Mexico City | Mexico City | 19 | 23.5 | 2.2 | 0.017 | 0.035 | 22.4 | 0 |
| 2024 R22 Las Vegas | Las Vegas | 38 | 22.3 | 1.6 | 0.044 | 0.051 | 22.8 | 0 |
| 2024 R23 Qatar | Lusail | 5 | 28.2 | 4.5 | 0.010 | 0.083 | 22.4 | 0 |
| 2024 R24 Abu Dhabi | Yas Island | 21 | 23.0 | 2.3 | 0.019 | 0.025 | 22.4 | 0 |
