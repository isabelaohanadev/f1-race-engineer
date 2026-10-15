# Backtest da Fase 3: o que foi congelado ANTES de ver o resultado

Este documento foi escrito e commitado **antes de qualquer execução do backtest na população de
corridas**. Os valores vivem em `strategy/protocol.py` (o backtest os lê de lá, não dos padrões dos
módulos), e `tests/strategy/test_protocol.py` prende cada um aos padrões do código. O código em que
foram fixados é o commit `0169cde`.

## A regra

Qualquer mudança nos valores abaixo, **depois de ver o resultado do backtest**, é um **ajuste
pós-resultado**: aparece no relatório (`docs/backtest.md`, seção "Ajustes pós-resultado") com o valor
antes, o valor depois e o efeito medido. Sem isso, o backtest deixa de ser uma avaliação limpa: um limiar
escolhido olhando o Brier é um limiar que tem o Brier como parâmetro.

Corrigir um **defeito de código** (um resultado absurdo causado por um erro, não por uma escolha) é
permitido, e também vira ajuste pós-resultado, com o antes e o depois.

## O que foi congelado

| Decisão | Valor | De onde vem |
|---|---|---|
| Limiar para ligar o alerta de undercut | **0,6** | pedido da usuária ("comece em 0,6") |
| Limiar para desligar (histerese) | **0,5** | idem |
| Fator de inflação do desvio da margem do undercut | **1,3** | o quantil 95% de \|z\| da inclinação dos 251 stints futuros com o `k` do modelo (1,29); **não** vem da margem do undercut |
| Horizonte do undercut | K entre 3 e 10 voltas; mínimo de 5 voltas restantes | `UndercutConfig` |
| Ultrapassagem | o carro só passa com **0,7 s/volta** de vantagem de ritmo; senão fica a **0,7 s** atrás | heurística, não medida |
| Parada dos outros carros | ganho de trocar agora até o fim > perda, com folga sorteada de **3,0 s**; até **3** paradas; quem não usou dois compostos para até **3** voltas do fim; sem parada voluntária com pneu de até **5** voltas | heurística, não medida |
| Desgaste do pneu sob SC | **0,3** da volta verde | heurística, não medida |
| SC | até 3 novos; mistura do histórico com o uniforme com peso de **4** eventos; início espalhado em ±**3** voltas | `SimConfig` |
| Simulações por previsão | **5.000** | pedido |
| Voltas avaliadas | volta **5**, **10**, **15**, ... até a penúltima (uma em cada 5) | custo: 5.000 simulações por volta de 21 corridas |
| Sensibilidade à taxa de SC | **0,5x, 1x, 2x** | pedido |
| Variantes de `k` | **físico, estimado, encolhimento**; o resultado principal usa o **encolhimento** | o padrão do código desde a etapa 3 |
| Horizontes do erro de tempo de volta | **1 a 5** voltas à frente; cobertura nominal do intervalo **90%** | |
| Estado no instante t | um piloto que não cruza a linha há mais de **2,0** voltas típicas dele saiu da corrida | heurística |
| Bootstrap | **2.000** reamostragens de corridas inteiras, semente **0**, intervalo de **95%** por percentis | |

## O plano de análise

**População.** Leave-one-race-out nas **21 corridas secas de 2024** (≥ 99% das voltas em pneu de seco).
A corrida britânica (pista secando), o Canadá e o Brasil (molhadas) são **reportados à parte**, nunca em
média com as secas: o modelo só conhece compostos de seco e a degradação negativa de pista molhada mede o
clima. Para essas três, as constantes vêm das 21 secas.

**Sem a corrida avaliada.** Para cada corrida, as constantes (`Priors` do ritmo e da perda no pit,
e `SimPriors` do SC, abandonos e trocas) são recalculadas deixando-a de fora. O fator de inflação do
undercut do resultado principal **não** é calibrado no backtest (é o 1,3 acima); o fator calibrado nas
outras corridas aparece só como diagnóstico, rotulado.

**A verdade** é a ordem **na pista** (mais voltas, depois quem cruzou antes), como em
`docs/verdade-do-backtest.md`; o resultado oficial vem ao lado onde divergir.

**O estado no instante t** (`strategy/snapshot.py`, função pura): para a volta N, o instante t é o em que
o **líder completa a volta N**. O sistema só conhece as voltas com relógio até t. Os demais carros estão,
em geral, na volta N-1 (um retardatário em N-2 ou menos), e o que acontece na volta N deles (tempo,
parada, composto) é futuro. O simulador começa cada carro na sua última volta completa. Para o undercut, o
instante é o em que o **atacante** completa a volta N. Um segundo conjunto de números, com o estado por
contagem de voltas (cada piloto na sua volta N, que dá ao retardatário informação de depois de t), é
reportado como **diagnóstico do vazamento**, não como resultado.

**As variantes de `k`.** O resultado principal usa o encolhimento. Uma variante só é declarada melhor se o
intervalo de 95% da **diferença pareada por corrida** (bootstrap) exclui o zero, **no mesmo sentido**, no
erro de tempo de volta **e** no Brier da vitória; do contrário, o texto diz "sem diferença detectável" e
o encolhimento permanece.

**O que se reporta** (e como):

1. **Erro de previsão do tempo de volta**, por horizonte (1 a 5 voltas), em voltas limpas do mesmo stint:
   viés, erro absoluto médio, e a cobertura do intervalo de 90% (nominal contra observada), por variante de
   `k`, com bootstrap sobre corridas.
2. **Brier da vitória e do pódio por andamento da corrida** (faixas de 10%), com intervalo de bootstrap
   **sobre corridas** (a unidade é a corrida, não a volta), contra os dois comparadores: **"o líder atual
   vence"** (p = 1 para quem lidera no instante t) e **"todos iguais"** (p = 1/n).
3. **A curva de calibração** (probabilidade prevista contra frequência observada) **com o n de cada
   faixa**.
4. **A avaliação do undercut** nas paradas reais, no modo **ao vivo** como resultado principal e no modo
   oráculo como diagnóstico, com a **ressalva do viés** (só se observam as paradas que aconteceram, e as
   equipes tendem a parar quando funciona: a precisão é otimista, a revocação não é mensurável, a
   calibração é condicional a "a equipe decidiu parar").
5. **A sensibilidade à taxa de SC** (0,5x, 1x, 2x) no Brier por andamento.

**A regra sobre o comparador.** Em cada faixa de andamento, o Monte Carlo **supera** "o líder atual
vence" se o intervalo de 95% da diferença pareada de Brier (Monte Carlo menos líder) fica inteiro abaixo de
zero; **não supera** se o intervalo contém o zero; e é **pior** se fica inteiro acima. O relatório diz isso,
faixa a faixa, **com todas as letras**, inclusive quando o resultado é ruim.

## O que já tinha sido visto antes do congelamento

Duas execuções de **verificação de que o código roda**, ambas em **uma** corrida (Miami):
`strategy.undercut_eval` (achou 2 eventos, ambos sob pista não verde, 0 primários) e
`strategy.forecast_eval` / uma previsão do Monte Carlo nas voltas 15, 30 e 45 (conferi a soma 1,000 e 3,000,
o tempo e as fontes do modelo). **Nenhuma métrica de acerto foi olhada para escolher parâmetros.** As
medidas dos dados que alimentam as constantes (perda no pit, SC, abandonos) são anteriores e não são
resultados do backtest.

## Limitações conhecidas já neste ponto (e que o relatório repete)

- O SC depois da volta 10 vem de **3 eventos**; as paradas sob SC são 15.
- As regras de **parada e de ultrapassagem** e o **desgaste sob SC** são heurísticas.
- **VSC (8 períodos) e bandeira vermelha (2)** ficam fora do simulador, e abandono não causa SC.
- Vinte e uma corridas é uma amostra pequena: os intervalos de bootstrap serão largos.

## Ajustes pós-resultado

Nenhum até agora (este documento precede o resultado). Se houver, ficam registrados em `docs/backtest.md`.
