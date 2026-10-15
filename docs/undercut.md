# Alerta de undercut: a mecânica e o que ela NÃO prova

Código: `strategy/undercut.py`. Testes: `tests/strategy/test_undercut.py` (com corridas sintéticas
de ganho e perda conhecidos). A **avaliação contra paradas reais** (`strategy/undercut_eval.py`) rodou
no backtest e o resultado está em `docs/backtest.md` (seção 6): precisão de 83% em 18 alertas ao vivo
(otimista pelo viés descrito abaixo), cobertura do intervalo de 90% da margem de 81%. Este documento
descreve o que o código faz, não o quanto acerta.

## A pergunta

O atacante `A` está atrás do rival `R`. Se `A` parar agora (entra na volta N+1, sai na N+2), ele sai
na frente de `R` depois que `R` também parar?

## A conta

Horizonte de `K` voltas (`K` = quantas voltas faltam para o rival parar, pelos stints já concluídos
do composto dele, entre 3 e 10):

| | volta N+1 | volta N+2 | voltas N+3 .. N+K | N+K+1 | N+K+2 |
|---|---|---|---|---|---|
| **A** | entrada (pneu gasto) | saída: pneu novo e frio, **+ perda** | pneu novo | pneu novo | pneu novo |
| **R** | pneu gasto | pneu gasto | pneu gasto (entra na N+K) | saída, **+ perda** | pneu novo |

A **margem** é o relógio de `R` menos o de `A` ao fim da volta N+K+2 (`K + 2` voltas cada), com os
relógios de partida vindos de `session_time_ms`; positiva = `A` termina à frente. A perda no pit (~21 s)
**já inclui o pneu frio da volta de saída** (é medida entrada + saída contra o ritmo normal), e como os
dois pagam uma parada a média cancela: a margem não depende dela (testado: perda 8 s maior, mesma
margem), só o ponto de retorno depende.

### O ponto de retorno e o tráfego

Pelos relógios de todos os carros dá para saber atrás de quem `A` volta à pista. Cada carro à frente
dele é projetado no pneu atual (os outros ficam fora durante o horizonte); `A` **não termina uma volta à
frente de nenhum carro que não consiga passar** (vantagem de ritmo menor que `pass_advantage_s` = 0,7
s/volta), e a distância que mantém atrás dele é `follow_gap_s` = 0,7 s. `margin_free` é a margem sem
tráfego; `margin` é a que vale.

Um erro meu, pego por teste: a primeira versão só olhava o carro mais próximo à frente. Quando o rival
parava e passava a ser esse carro, a volta lenta dele o fazia "passável" e o carro de trás (o que de
fato segurava `A`) era ignorado. A margem saía −0,70 s contra −1,25 s correto.

Uma consequência da regra de ultrapassagem: um déficit **menor que 0,7 s** no fim vira −0,7 s (o carro
não passa quem tem o mesmo ritmo). A descontinuidade é de até 0,1 s e está documentada num teste.

## A confiança

`P(margem > 0) = Φ(margem / desvio)`. O desvio é a propagação **linear exata** das incertezas pela soma
dos tempos de volta, não uma soma em quadratura de parcelas "independentes":

- **Nível e inclinação do mesmo carro não são independentes.** O erro da inclinação mexe nas voltas
  de pneu gasto e, pela extrapolação à idade zero (o nível do pneu novo sai do nível do atual), nas
  `K+1` de pneu novo, com sinal oposto. O coeficiente da inclinação é `(Σ idades_velhas − idade_média) −
  n_novas × idade_média`. Conferido por simulação: o desvio da fórmula bate com o de sortear os
  parâmetros e somar os tempos volta a volta, a 2%.
- **A diferença de composto é um erro compartilhado** quando os dois trocam o mesmo par (por exemplo
  MEDIUM para HARD): só conta o saldo de voltas no pneu novo (`K+1` contra 2), e só o `τ` de cada piloto
  é individual. Com pares diferentes, os termos somam em quadratura.
- **A inclinação do pneu novo** usa `predictive_sd` (stint futuro), nunca o erro-padrão.
- **Ruído de volta:** `sqrt((K+2)(σ_A² + σ_R²))`. **Dispersão entre as duas paradas:** `√2 × sd` da perda.
- **Erro do `k`:** refaz o modelo inteiro (degradação, nível, diferenças) com `k ± sd` e mede o quanto a
  margem muda. O `sd` do `k` mistura a incerteza a priori da constante física (50%) e a do estimado,
  pelo peso. Com `k` físico, o erro do `k` é grande por construção.
- **Fator de inflação** (`sd_inflation`, provisório 1,3), porque a cobertura medida ainda é 87–91%.
  Vem da calibração da **inclinação** (o fator que cobriria 95% dos 251 stints futuros com o `k` do
  modelo é 1,29), não da margem do undercut: a etapa de backtest o recalibra com
  `calibrate_inflation` e o substitui.

## Quando não alertar

Cada motivo vem em `reasons`; a margem e a probabilidade ainda são devolvidas, para exibir.

| Motivo | Regra |
|---|---|
| sem rival | o líder, ou piloto sem relógio na volta N |
| sem perda no pit | nem paradas verdes nem prior |
| o rival (ou o atacante) já parou | stint diferente do do outro |
| carro nos boxes | entrada ou saída na volta N |
| pista não verde | status diferente de `"1"` nas voltas N−1 e N dos dois, **ou** metade do grid fora da verde na volta N |
| origem "todos os compostos" | previsão com 68% de cobertura, não sustenta alerta |
| poucas voltas | faltam menos voltas que `max(voltas para recuperar a perda, 5, K+2)`; sem recuperar até o fim, também |

**Histerese** (`AlertTracker`): liga em `threshold` = 0,6 e só desliga abaixo de `release` = 0,5. Uma regra
dura desliga na hora e apaga o estado. A chave é o par (atacante, rival): outro rival é outra situação.

## Sem vazamento do futuro

`build_context` corta as voltas depois de N como primeiro passo. O teste de corrupção troca **todas** as
voltas depois de N por lixo (tempos, compostos, status, relógio, paradas que não existiram) e exige
avaliação idêntica de todos os pares, valor por valor; e um teste de contraprova mostra que, sem o corte,
o resultado muda (o teste morde).

## Como validei o código (e o que isso vale)

- **À mão:** num mundo plano (MEDIUM degrada 0,10 s/volta, HARD não, sem diferença de ritmo, `k` físico),
  a margem é `0,10 × (K·N + K(K+1)/2 − (N+1)) − gap`; para N=30, K=5, gap=5 s: **8,4 s**. O código dá
  8,3999995.
- **Previsto contra realizado:** um mundo contrafactual em que o undercut acontece de fato no gerador
  (A para na N+1, R na N+K, mesma história até N). O erro absoluto fica abaixo de 1,6 s e abaixo do
  desvio declarado, em 8 combinações de semente e horizonte (nos 48 casos medidos, |z| máximo 0,33).
- **Mutações:** 19 defeitos introduzidos de propósito (sinal da margem, piso do bloqueio, perda do
  rival, corte do futuro, coeficiente líquido da inclinação, histerese, cada regra dura, inflação, erro
  do `k`, número de voltas do ruído...). Todos derrubam pelo menos um teste. Na primeira rodada um
  sobreviveu: "só o carro mais próximo bloqueia" (o erro descrito acima), porque o teste só conferia o
  sinal da margem; passou a conferir o valor contra a margem realizada.

## O que NÃO foi validado

- **Até o backtest, nenhuma avaliação contra a realidade.** Tudo acima é dado sintético. **O desvio
  previsto ficou bem acima do erro real no sintético (|z| < 0,35) porque o mundo sintético não tem
  espalhamento entre corridas.** O backtest (`docs/backtest.md`, seção 6) mediu: com o fator de inflação
  congelado (1,3) o intervalo de 90% da margem cobre 81%, e os dados pedem 1,87 (no oráculo, 1,31). O
  fator **não foi mudado** depois do resultado.
- **Um teste de execução em dado real**, numa corrida (Miami, voltas 20, 28 e 40): roda em 0,1–0,2 s por
  volta, as regras duras aparecem (pista não verde na 28, carros nos boxes, poucas voltas). Isso mostra que **não
  quebra**; não mostra que acerta.
- **Os outros carros não param no horizonte** e ficam no pneu atual; só o conjunto de carros à frente
  prende o atacante. Uma parada de um terceiro carro muda o tráfego e não é modelada.
- **O composto seguinte é o mais provável** pelos stints já feitos; a equipe escolhe, e uma troca para um
  composto que ninguém correu faz a avaliação ser descartada ("sem diferença de ritmo").
- **A regra de ultrapassagem é uma heurística** (0,7 s/volta de vantagem, 0,7 s de distância), não
  medida. Ficam como configuração.
- **Sem Safety Car.** O risco de SC dentro do horizonte vem com o Monte Carlo.
- **A diferença de ritmo entre compostos é fraca nestes dados** (ver README). O undercut tende a alertar
  menos por causa disso; se o backtest apontar essa incerteza como gargalo, o caminho é um modelo de
  ritmo com efeito fixo de volta, que não fiz.
- **O fator de inflação 1,3 foi congelado antes do backtest**, que mostrou ser baixo para o modo ao
  vivo (cobertura 81%, os dados pedem 1,87). Não foi ajustado, para a avaliação continuar limpa.

## A avaliação nas paradas reais (rodou no backtest)

Código: `strategy/undercut_eval.py` (`python -m strategy.undercut_eval`). Testes:
`tests/strategy/test_undercut_eval.py` (45 testes, corridas sintéticas em que o undercut acontece no
gerador; 20 mutações derrubam todas um teste, e uma sobrevivente da 1ª rodada, o limiar inclusivo
`>=`, ganhou o teste do valor exatamente no limiar).

**Protocolo.** Cada parada de um piloto na volta `s` é um evento: decide-se na volta N = s − 1, o rival é
o carro à frente (pelo relógio), e o resultado real é quem está à frente ao fim da volta `s_r + 2`, em que
`s_r` é a volta em que o rival de fato parou (a mesma N+K+2 da margem prevista). A margem realizada é a
diferença dos relógios; "sucesso" = o atacante terminou à frente. Medidas: **Brier** (e o do "sempre a
taxa-base", o piso a bater), **curva de calibração** por faixa de probabilidade (com os cortes 0,5 e 0,6
do alerta), **precisão no limiar**, erro da margem (viés, erro absoluto) e **cobertura do intervalo de
90%** da margem realizada.

**Dois modos.** `oráculo` (padrão) usa o horizonte e os compostos que de fato aconteceram: mede a física
do modelo (ritmo, degradação, perda, tráfego). `--live` usa o que o sistema saberia na volta N (horizonte
esperado, composto mais provável): é o que decide o alerta e erra mais; a diferença é o custo de não saber
o que o rival fará.

**Sem vazamento.** Cada evento usa `build_context` na volta N (corta o futuro; testado: a avaliação é
igual à de uma corrida que parou em N) e as constantes **sem a corrida avaliada** (`leave_one_out`,
testado com um espião: a corrida diferente não entra na reserva dela). O **fator de inflação é calibrado
só nas outras corridas** (`loo_inflation`): a margem e o desvio cru ficam guardados em cada resultado, e
refazer a probabilidade com outro fator não reavalia nada. Piso de 1,0: é um fator de inflação, e uma
amostra pequena não deve desinflar; o que os dados pedem vai ao relatório.

**O que entra na média (eventos primários).** Sem regra dura de não alertar e **sem pista não verde** entre
N−1 e a volta final (uma parada sob SC não é comparável). Os outros aparecem em linha à parte, com o
motivo. São descartados na busca: rival que não parou, parou na mesma volta, parou num horizonte fora de
[3, 10] voltas, estava em outro stint, parada extra de qualquer um na janela, corrida que acaba antes.

### O viés que o relatório não finge tirar

Só se observam as paradas que **aconteceram**, e as equipes tendem a parar quando o undercut funciona:

1. a taxa de sucesso entre as paradas observadas é **maior** que a de todas as situações em que um
   atacante estava atrás de um rival (as em que não funcionaria, ninguém parou, e não aparecem);
2. a **precisão** medida é otimista, e a **revocação** (dos undercuts que dariam certo, quantos o modelo
   alertou) **não é mensurável**: o denominador não existe;
3. a calibração é **condicional** a "a equipe decidiu parar": diz se, nas paradas feitas, a probabilidade
   bate com a frequência, e não se vale para uma situação qualquer;
4. o resultado real também depende do que os dois lados fizeram depois; por isso as paradas extras na
   janela são descartadas.

### Um aviso de poder estatístico, já visível

Uma execução do pipeline completo numa corrida real (Miami, **só para ver que roda**, sem interpretar
números) achou **2 eventos**, **ambos sob pista não verde** (0 primários); descartou 26 paradas por
motivos estruturais (7 com horizonte fora de [3, 10], 7 sem o rival parar depois, 4 paradas na mesma
volta, 4 com o rival em outro stint, 3 sem rival à frente, 1 com parada extra). Isto vale como alerta: com
poucos eventos limpos por corrida, a curva de calibração por faixa vai ter faixas com pouquíssimos casos.
O relatório mostra `n` por faixa justamente para ninguém ler uma frequência de 1 em 2 como calibração.
Não ampliei os critérios (K fora de [3, 10], eventos sob SC) para inflar a amostra: isso mudaria a pergunta.
