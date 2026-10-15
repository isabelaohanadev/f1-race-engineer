# Monte Carlo da corrida: o que simula, o que é heurística e o que NÃO foi validado

Código: `strategy/montecarlo.py` (simulação), `strategy/history.py` (SC, abandonos e trocas de composto,
sem a corrida avaliada), `strategy/forecast_runner.py` (executor "só a última") e
`strategy/forecast_eval.py` (avaliador, rodado no backtest: `docs/backtest.md`).
Testes: `tests/strategy/test_montecarlo.py`, `test_forecast_runner.py`, `test_forecast_eval.py`,
`test_history.py`.

## O que uma simulação faz

Da volta N até a última, `n_sims` (5.000) corridas em arrays pilotos × simulações. Cada simulação
sorteia, cada coisa num stream de números aleatórios próprio:

| O quê | De onde vem | Medido ou heurística |
|---|---|---|
| nível de ritmo, inclinação de cada piloto, diferença entre compostos, perda no pit | o modelo da volta N (`PaceModel`, `estimate_pit_loss`), com os erros que ele declara | medido |
| ruído de cada volta | `sigma_lap` do piloto | medido |
| Safety Car: se há, quando, por quanto tempo | Poisson(`lambda`) com `lambda` da posteriori Gamma; início e duração das outras corridas | medido (poucos dados: ver abaixo) |
| o que o SC faz: ritmo (`rho`), espaçamento (`gap`), custo da parada | `strategy/history.py`, medidos contra o campo | medido |
| desgaste do pneu sob SC | `sc_wear` = 0,3 da volta verde | **heurística** |
| abandono: probabilidade e volta | Beta e frações históricas | medido |
| composto seguinte de cada parada | contagem histórica, regra de dois compostos | medido |
| quando cada carro para | ganho de trocar agora até o fim contra a perda no pit, mais uma folga sorteada de 3 s; quem ainda não usou dois compostos para até 3 voltas do fim | **heurística** |
| tráfego e ultrapassagem | a regra do undercut: sem 0,7 s/volta de vantagem de ritmo não passa e fica a 0,7 s atrás | **heurística** (não medida) |

O vencedor é quem completa mais voltas e, empatando, cruza a linha antes (a ordem na pista, como a
verdade do backtest). Abandonados ficam fora da pista (não seguram ninguém) e são classificados pelas
voltas que completaram.

### O Safety Car como evento da corrida

Não é uma taxa por volta. O número de SCs da corrida é Poisson(`lambda`), o início sai da distribuição
histórica de frações da corrida e a duração das durações históricas. Na volta N os SCs que faltam são
Poisson(`lambda x (1 - F(N))`). A incerteza de `lambda` entra **em cada simulação**: cada uma sorteia
`lambda` de Gamma(eventos + 0,5, corridas) (posteriori de Jeffreys), então a incerteza da taxa vira
incerteza da probabilidade de vitória. A distribuição do início é uma mistura do histórico (com início
espalhado em ±3 voltas) e de um uniforme, com peso do uniforme equivalente a 4 eventos: com 8 eventos
históricos, uma distribuição só empírica concentraria o SC em 3 voltas exatas.

**Sensibilidade:** `sensitivity()` roda a mesma simulação com a taxa de SC em 0,5x, 1x e 2x, nos
**mesmos números aleatórios** (o multiplicador age depois do sorteio, e a contagem sai da inversão da
CDF de Poisson, que gasta um número por simulação em vez de `rng.poisson`).

## Reprodutibilidade e comparação entre variantes

`SeedSequence([ano, etapa, volta])`, com um stream independente por componente (parâmetros, paradas,
composto, SC, abandono, ruído de volta). Cada stream consome **sempre a mesma quantidade** de números,
qualquer que seja o modelo: o `k` físico, o estimado e o encolhimento mudam valores, não a quantidade
sorteada. Testado: os sorteios das três variantes são idênticos (array a array), então a diferença entre
elas é sinal, não ruído de simulação. Cada probabilidade vem com `sqrt(p(1-p)/n)`.

Um erro que o teste de simetria pegou: com relógios exatamente iguais, quem ficava na frente era o de
código alfabeticamente menor (um carro ganhava 80% de dois idênticos). Agora o desempate é sorteado.

## Desempenho (medido, 5.000 simulações, 20 carros)

| Situação | Voltas por frente | Tempo |
|---|---|---|
| Volta 12 (corrida quase inteira) | 48 | ~1,0 s |
| Volta 30 | 30 | ~0,6 s |
| Volta 50 | 10 | ~0,2 s |
| Volta 58 (fim de corrida) | 2 | ~0,05 s |

O custo é linear nas voltas que faltam (~21 ms por volta com 20 carros; os sorteios, 10 ms). Uma volta
dura ~90 s: sobra uma ordem de grandeza. Por isso o desenho é numpy vetorizado em
pilotos × simulações com um laço sobre as voltas (e, dentro dela, sobre as posições do grid para o
tráfego), e não um laço sobre simulações. Medido numa máquina de desenvolvimento; o teste de CI só
exige que uma corrida inteira caiba em 15 s.

## Onde roda, e o que acontece quando chega a volta seguinte

`LatestOnlyRunner` roda a simulação numa **thread** (o numpy solta o GIL nas contas grandes), no
processo do consumidor de estratégia, que é separado da API: nunca no event loop da API, e a leitura do
Redis no consumidor também não trava (testado: o loop gira enquanto a simulação computa).

A política pedida: "se uma simulação ainda estiver rodando quando chegar a volta seguinte, descarte a
antiga e não acumule fila". No máximo uma simulação roda e uma espera; a nova **cancela** a em curso (a
simulação confere o sinal a cada volta simulada) e **substitui** a que esperava, que nunca roda. Um
resultado que termina depois de uma volta mais nova ter chegado é `stale` e não é entregue (o cliente
não vê a probabilidade andar para trás). Uma simulação que falha não derruba o consumidor.
A duração vai para o histograma `f1_strategy_simulation_duration_seconds` (rótulo só `result`:
`completed`, `cancelled`, `stale` ou `failed`; nunca piloto, corrida ou volta) e a idade do último
resultado entregue para `f1_strategy_last_result_age_seconds`.

## Como validei (e o que isso vale)

- **Contas de papel e fórmulas fechadas, independentes do código:** a probabilidade de SC da simulação
  bate com `1 - (1 + m/corridas)^-(eventos + 0,5)` (a posteriori Gamma-Poisson, com e sem o
  multiplicador de 0,5x e 2x); a de abandono, com `r*m / (1 - r*(1-m))`; a contagem de Poisson, com
  `e^-lambda` e a média. As funções puras do tempo (cronograma do SC, compressão sob SC, retenção no
  tráfego, volta do abandono) têm resposta calculada à mão.
- **Propriedades:** soma das vitórias 1 e do pódio 3 (inclusive quando quase todos abandonam); líder
  60 s à frente com poucas voltas, ~100%; dois carros iguais, 50/50 dentro do erro de Monte Carlo (com e
  sem tráfego); mesma semente, mesmo resultado; futuro corrompido não muda a simulação.
- **Cenários que isolam uma regra:** um perseguidor mais rápido só passa com a vantagem de ritmo que a
  regra pede; o SC ajuda o perseguidor e mais quanto maior a taxa; a parada sob SC é barata (mais carros
  param); quem usou um composto só para antes do fim; o pneu muito novo não para.
- **Mutações:** 32 defeitos plantados no simulador, 11 no executor e nas métricas e 16 no avaliador (59
  ao todo), cada um derrubando algum teste. Na primeira rodada 10 do simulador sobreviveram: 2 eram
  mutantes equivalentes (uma máscara redundante, que removi), 4 apontavam código só testável por dentro
  de `simulate` (extraí funções puras), 3 pediam cenários que faltavam (a ordem de quem abandona na
  mesma volta, o carro abandonado que não segura ninguém, o mínimo de idade do pneu) e 1 revelou um
  **erro de modelagem meu**: o carro que entra nos boxes na volta N já tem a volta de entrada no relógio,
  e a volta de saída somava a perda inteira de novo (dupla contagem; agora soma só `perda - excesso da
  volta de entrada`, e um cenário em que isso muda o pódio a protege).
- **Erros que os testes pegaram:** o desempate de relógios iguais (80/20 entre dois carros idênticos);
  um índice fora do array quando um carro já estava nos boxes e não havia paradas voluntárias.
- **Execução em dado real, uma corrida (Miami), só para ver que roda:** 20 pilotos com ritmo, soma 1,000
  e 3,000, 0,2 a 0,8 s por previsão com 5.000 simulações, fontes dos modelos nos três tipos. Isso mostra
  que **não quebra e cabe no orçamento**, não que acerta.

## O que NÃO foi validado

- **Até o backtest, nenhuma probabilidade tinha sido medida contra corridas reais**, e o que o
  backtest mediu (`docs/backtest.md`) é ruim: **o Monte Carlo não supera "o líder atual vence" em nenhuma
  fase da corrida** (8 faixas de 10 não se distinguem, e nas duas últimas, 80% a 100%, ele é pior). O
  que está acima é conta de papel, propriedades (soma 1, pódio 3, simetria, determinismo, sem vazamento)
  e mutações: mostram que o código faz o que descrevo, não que a descrição acerta as corridas. O
  diagnóstico posterior (`docs/backtest-diagnostico.md`) aponta o ponto do modelo, não a dispersão, como
  fonte do problema; é uma hipótese, sem correção aplicada.
- **O SC depois da volta 10 vem de 3 eventos** (8 períodos em 7 corridas, 5 deles nas voltas 1 a 7). A
  probabilidade de SC no meio da corrida é muito incerta; a posteriori Gamma carrega isso, mas não o
  cura. As paradas sob SC são 15: a razão de custo (0,58) tem um desvio de ~0,4.
- **VSC e bandeira vermelha ficam de fora**; abandono não causa SC (são independentes na simulação).
- **As regras de parada e de ultrapassagem são heurísticas**, não medidas: equipes decidem por rivais,
  tráfego e estratégia que o ganho de tempo não vê, e só o carro imediatamente à frente segura.
- **O desgaste sob SC (0,3) é uma heurística**; não se mede desgaste num tempo de volta sob SC.
- **O estado por contagem de voltas** (a volta N de cada piloto) dá ao retardatário informação que o
  líder ainda não teria completado: ao vivo se saberia menos. O backtest usa o estado **no instante t**
  (`strategy/snapshot.py`) e mediu o vazamento do outro: Brier 0,030 melhor com a contagem de voltas.
- **Sem o `k` como incerteza no sorteio:** o `k` cancela entre carros no combustível, mas entra na
  comparação entre pneus; as variantes de `k` são comparadas rodando o simulador com cada uma (mesmos
  números aleatórios), e não sorteando o `k` dentro de uma simulação.
- **O número de pilotos simulados** é o dos que têm ritmo; quem não tem fica de fora (`without_model`)
  e, se vencer, conta como erro no avaliador.
