# Como construir um modelo de decisão calibrado com Exu

Um roteiro fase por fase. É agnóstico: nenhuma língua, domínio ou encoder é
assumido. Tudo aqui ou está implementado neste repositório ou está marcado como
decisão de projeto que cabe a você tomar.

O Exu implementa o RLCD, *Reinforcement Learning for Calibrated Decisions*, o
método de treino que a [TypeSafe](https://typesafe.ai/) usa no Jev. A recompensa é
uma scoring rule estritamente própria, então reportar probabilidades honestas é a
única forma de maximizá-la.

O contrato de decisão tipada em si vem do [Jev, da
TypeSafe](https://typesafe.ai/): as primitivas `choice`, `score` e `noul`,
respostas que carregam uma distribuição mais um campo de confiança separado, e o
gate de ação em cima dessa confiança. É deles também o nome do método, RLCD. A
receita aberta seguida aqui é o
[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0, do Nandakishor): o
formato da sequência com um marcador de máscara por opção, o orçamento de tokens,
a recompensa composta e os buckets de temperatura. Os números atribuídos à
referência abaixo são do Laya, não medidos neste repositório. O crédito completo
está no README.

Um artigo relacionado, citado pela ideia de roteamento e não pelo desenho daqui:
Nandakishor M, *Confidence-Aware Routing for Large Language Model Reliability
Enhancement* ([arXiv:2510.01237](https://arxiv.org/abs/2510.01237), 2025), do
mesmo autor do Laya. Ele estima a confiança antes de gerar e roteia a consulta
por quatro caminhos: geração local, retrieval, um modelo maior, ou revisão
humana. Este repositório não implementa nada disso; o gate de agir ou escalar é a
mesma ideia reduzida a um encoder.

## A forma da coisa

```
estado + pergunta tipada
        |
        v
uma sequência por pergunta:  tipo | instrução | [MASK] opção | [MASK] opção | estado
        |
        v
encoder bidirecional (fine-tune completo)
        |
        v
embedding do tipo de pergunta, duas camadas transformer extras
        |
        +--> estado oculto em cada marcador -> scorer -> um logit por opção
        |                                                       |
        |                                        logits / temperatura -> softmax
        |
        +--> CLS + estatísticas da distribuição -> agir ou escalar
```

Por que isso existe: decisão reflexa (rotear ticket, marcar fraude, medir
urgência) precisa de probabilidade, não de prosa. Gerar token custa centenas de
milissegundos, exige parser, e a confiança que um modelo de chat escreve é mais
texto. Aqui a saída já é uma distribuição, e não há o que alucinar.

## Fase 1: fechar o contrato

Defina o que entra e o que sai antes de encostar em um modelo.

| Primitiva | Pergunta | Saída |
| --- | --- | --- |
| `choice` | escolher uma entre N opções explícitas | distribuição sobre as opções |
| `score` | posicionar o estado numa rubrica ordinal | distribuição sobre os níveis |
| `noul` | pergunta booleana | distribuição sobre Não e Sim |

As três são o mesmo mecanismo: pontuar opções e fazer softmax. `noul` é uma
`choice` de duas opções fixas; `score` é uma `choice` cujas opções são
"nível i: texto". O que muda é a palavra de tipo no prompt, o embedding de tipo,
um termo extra na recompensa no caso ordinal, e o pós-processamento da saída.

Decisões de contrato que valem copiar:

- Perguntas e opções são definidas na hora da requisição. O modelo lê o texto das
  opções, então tarefa nova não precisa de camada de saída nova nem de retreino.
- O estado pode ser string, objeto ou lista. Estruturas viram JSON, serializado
  sem escapar caracteres não ASCII, senão `ç` e `ã` viram sequência de escape e
  queimam o orçamento de tokens.
- Critérios estruturados também viram JSON compacto, nunca a representação
  interna da linguagem. Zero e falso são valores legítimos; só nulo e string
  vazia significam "sem descrição".
- Uma sequência por pergunta. N perguntas sobre o mesmo estado viram um lote de N
  sequências num forward só. O custo é reencodar o estado N vezes, e é por isso
  que a latência cresce com o número de perguntas mesmo em lote.
- A resposta informa tokens de entrada e zero tokens de saída, o que mantém a
  contabilidade de custo simples.

## Fase 2: escolher o encoder

Critérios, em ordem:

1. Bidirecional (encoder-only). Cada token precisa enxergar o estado e todas as
   opções ao mesmo tempo.
2. Token de máscara no vocabulário. O truque do marcador reaproveita ele, e o
   encoder já aprendeu no pré-treino a tratá-lo como posição onde algo precisa
   ser previsto.
3. Tokenizer eficiente na sua língua. Meça tokens por palavra em texto real do
   seu domínio.
4. Contexto suficiente para instrução, opções e estado juntos.
5. Tamanho contra latência, e licença.

Meça antes de casar. Encoder excelente numa língua desaba em outra. Escolha dois
candidatos, meça a fertilidade do tokenizer no seu texto real, rode um treino
curto com cada um, e só então decida. O default do `ExuConfig` é um BERT
multilíngue como ponto de partida neutro, não como recomendação.

Para uma língua só, um encoder especializado costuma ganhar em velocidade e em
acurácia. Confira a licença e o token de máscara de qualquer candidato antes de
usar.

## Fase 3: desenhar a sequência

Por pergunta: CLS, a palavra de tipo e a instrução, um separador, depois cada
opção precedida de um marcador de máscara e seguida de um separador, depois o
estado, e um separador final.

A posição do marcador é guardada, porque é dali que sai o logit daquela opção.
Como o encoder é bidirecional, o estado oculto no marcador já carrega a opção, as
opções concorrentes, a instrução e o estado.

Orçamento de tokens, que é onde mora o limite real da arquitetura:

- Dois limites: o tamanho máximo total e um orçamento para o cabeçalho (instrução
  mais opções). Este repositório usa 512 e 192 por padrão.
- Cada opção tem teto próprio (48 tokens aqui, contando o marcador). A instrução
  tem garantidos pelo menos 8 tokens do que sobrar do cabeçalho.
- Quem cede primeiro é a opção, não a instrução: reservada a garantia acima, as
  opções são limitadas ao resto, com piso de 3 tokens de texto cada. Se nem isso
  couber, o `SequenceBuilder` levanta erro em vez de encolher toda opção até
  ficar ilegível.
- O estado fica com o resto. Na inferência ele é truncado à direita nesta
  implementação. Truncar à esquerda preserva turnos recentes e faz sentido para
  conversa.
- Se um marcador não couber no tamanho máximo, o builder levanta erro em vez de
  responder errado em silêncio.

Esse último ponto morde mais cedo do que parece. O contrato aceita até 255
opções, mas o orçamento de cabeçalho padrão comporta umas trinta: passando disso,
o `build` levanta `too many options for configured header_budget`. O Laya, em vez
disso, encolhe toda opção até um piso e responde, e é isso que produz o 0.425
achatado dele nos 77 rótulos do Banking77; aqui a falha é explícita, e a saída é
aumentar o `header_budget` ou decidir em duas etapas. Mantenha `choice` abaixo de
umas 20 opções, e acima disso vá de hierarquia.

Segurança: remova a string literal do token de máscara de todo texto vindo de
fora. Sem isso, uma entrada pode injetar marcadores falsos. O `SequenceBuilder`
faz isso no estado, na instrução e na descrição das opções.

## Fase 4: a cabeça de decisão

Em cima do encoder, treinada do zero:

1. Soma-se um embedding do tipo de pergunta (três entradas) a todas as posições.
2. Duas camadas transformer extras, pre-norm, uma cabeça de atenção a cada 64
   dimensões, feed-forward de 4 vezes o tamanho oculto, dropout 0.1, respeitando
   a máscara de padding. O papel delas é deixar os marcadores conversarem entre
   si, já condicionados ao tipo.
3. Lê-se o estado oculto na posição de cada marcador.
4. Um scorer pequeno (norm, linear, GELU, linear para um escalar) transforma cada
   marcador em um logit.
5. Opções que só existem por causa do padding do lote recebem um logit muito
   negativo.
6. Softmax sobre as opções de cada pergunta.

Ordem de grandeza: a cabeça extra tem dezenas de milhões de parâmetros, o scorer
cerca de um milhão, contra centenas de milhões no encoder. O encoder não fica
congelado: ele recebe fine-tune completo, com taxa de aprendizado menor que a da
cabeça.

## Fase 5: dados

Qualquer dataset rotulado vira pergunta tipada. Classificação vira `choice`, com
os rótulos como opções e uma descrição escrita à mão para cada um. Rótulo binário
vira `noul`. Nota ou escala vira `score`. Inferência textual vira `choice` de três
opções. O alvo pode ser one-hot ou uma distribuição soft, quando há vários
anotadores ou um modelo professor.

Aumentações, para o modelo não aprender atalho:

- Embaralhar a ordem das opções a cada passada. Sem isso o modelo aprende posição.
  No Laya a resposta muda em 15% a 23% dos casos sob permutação.
- Parafrasear as instruções.
- Alternar o estado entre texto cru e JSON aninhado.
- Injetar perguntas distratoras.

Separe o held-out por família inteira de tarefa, não só por exemplo. É a única
forma de medir o zero-shot de verdade, e os números do Laya são um bom choque de
realidade: 83.8% nas tarefas vistas contra 65.1% nas nunca vistas, com os
checkpoints base ficando abaixo do baseline de classe majoritária nas famílias
nunca vistas.

## Fase 6: a recompensa

Uma scoring rule recebe a distribuição reportada `q` e o resultado `y` e devolve
um número. Ela é estritamente própria quando a nota esperada é máxima só se `q`
for a distribuição verdadeira. Essa propriedade amarra recompensa a honestidade.

O contraexemplo ajuda: recompensa binária (1 se acertou, 0 se errou) tem valor
esperado linear em `q`, então o ótimo é jogar toda a massa na classe mais
provável. RL ingênuo maximiza acurácia e destrói calibração.

A recompensa composta aqui soma três peças:

- log score: o log da probabilidade dada ao alvo (soma ponderada, com alvo soft).
  Pune quem dá pouca probabilidade ao que aconteceu. Um piso limita a punição.
- score esférico: o produto interno entre alvo e `q` dividido pela norma de `q`,
  entre 0 e 1. Premia massa no lugar certo sem os picos de gradiente do log.
- ranked probability score, só em perguntas `score`: distância quadrática entre as
  distribuições acumuladas, dividida por `K - 1`. Ensina que errar por um nível é
  melhor que errar por três.

As definições exatas e os pesos estão em [algorithm.md](algorithm.md).

## Fase 7: o loop de treino

A política amostra perturbações em vez de tokens. Para cada pergunta:

1. Amostre `G` perturbações gaussianas dos logits com desvio `sigma`. Masque-as
   nas opções válidas e projete-as para somar zero, porque uma constante somada
   aos logits não muda o softmax.
2. Faça softmax de cada versão perturbada: `G` distribuições candidatas por
   pergunta.
3. Pontue cada candidata contra o alvo com a recompensa composta, sem gradiente.
4. Vantagem: recompensa menos a baseline do grupo, normalizada por um desvio
   padrão (global do lote no fine-tune do Laya, por grupo no GRPO).
5. Loss: menos a média de vantagem vezes a log-probabilidade gaussiana da amostra
   dado o logit atual. A amostra entra destacada do grafo, então o gradiente
   passa só pelo logit atual.

A intuição: o gradiente da log-probabilidade aponta na direção do ruído.
Perturbações que renderam acima da média do grupo puxam os logits para o lado
delas, e as abaixo da média empurram para longe.

O `sigma` decai ao longo do treino, e os hiperparâmetros diferem entre treino base
e fine-tune. Veja `PolicyConfig.base()` e `PolicyConfig.finetune()`.

Uma leitura crítica, para te poupar tempo. A recompensa é diferenciável em
relação aos logits, então dá para retropropagar direto, sem amostrar nada. O log
score é exatamente a cross-entropy com o sinal trocado. O que o RL faz aqui é
estimar, com ruído, o gradiente de uma versão suavizada do mesmo objetivo, mais
os termos esférico e RPS. A afirmação de que cross-entropy deixa o modelo
confiante demais e a scoring rule não se sustenta no nível do objetivo: com alvo
one-hot e dados separáveis, os dois empurram para a certeza. As diferenças reais
são o ruído agindo como regularizador, o piso do log e os termos extras. Então
rode o baseline direto primeiro, depois a versão com política, e compare ECE e
NLL no held-out. Se o RL não ganhar com folga, fique com o simples. É a ordem que
este repositório incentiva com `--mode baseline` e `--mode rlcd`.

## Fase 8: agir ou escalar

Treine a cabeça de ação com uma matriz de custo: agir e acertar rende um ganho,
agir e errar custa uma perda, escalar custa algo. O ponto de equilíbrio sai daí:
aja quando a probabilidade de acerto passar de `(perda - escalar) / (ganho +
perda)`. Com ganho 1, perda 3 e escalar 0.5, isso dá 0.625. Trocar os custos troca
o limiar, e é assim que a regra de negócio entra no modelo.

Este repositório entrega o modelo de custo e o baseline trivial (`should_act`
sobre a probabilidade máxima calibrada). O treino da cabeça aprendida fica como
extensão: ela só ganha o lugar se bater esse baseline.

Dois avisos. A probabilidade de ação de um checkpoint com fine-tune merece
desconfiança até a cabeça ser retreinada depois que o encoder mudou, porque a
distribuição de entrada dela mudou. E o limiar precisa ser aplicado sobre a mesma
grandeza que a calibração mediu, que é a probabilidade máxima, não a confiança
por entropia.

Para a versão de roteamento dessa ideia num cenário de LLM, com quatro caminhos e
a confiança estimada antes da geração, veja Nandakishor M, *Confidence-Aware
Routing for Large Language Model Reliability Enhancement*
([arXiv:2510.01237](https://arxiv.org/abs/2510.01237), 2025). Cenário diferente,
mesma forma de decisão, e não implementado aqui.

## Fase 9: multi-turno (opcional)

Para prever desfecho de conversa (conversão, churn) turno a turno:

- Fatie cada conversa em prefixos e mostre ao modelo só até o turno atual.
  Alimentar a conversa inteira no turno 1 vaza o futuro.
- O alvo de cada prefixo vem de TD(lambda): o último prefixo recebe o desfecho
  real, e cada anterior recebe uma mistura da previsão do modelo para o prefixo
  seguinte com o alvo desse seguinte. Com lambda igual a 1 todo prefixo é
  treinado direto contra o desfecho final, sem bootstrap nas previsões do próprio
  modelo.
- Aqui faz sentido truncar o estado à esquerda, para manter os turnos recentes.

Isso não está implementado neste repositório.

## Fase 10: calibração por temperatura

Depois do treino, com os pesos congelados, ajuste um escalar `T` que divide os
logits antes do softmax. Rode o modelo num conjunto held-out, guarde logits e
alvos, minimize a log-verossimilhança negativa sobre o `log T`, limite o
resultado, e exija um número mínimo de amostras, caindo para `T = 1` abaixo disso.

Um `T` só não basta. A confiança excessiva depende de quantas opções a pergunta
tem, então as temperaturas são agrupadas por tipo de pergunta e faixa de número
de opções, com fallback por tipo. Os buckets têm precedência sobre o valor por
tipo.

Três armadilhas, todas visíveis no Laya:

- Ajustar num pedaço do próprio treino. O modelo é mais confiante no que já viu,
  então `T` sai perto de 1 e a calibração parece melhor do que é. Use held-out.
- Um mapa de buckets velho sombreando valores por tipo recém-ajustados. Limpe
  quando não quiser.
- Um valor ajustado encostando no limite. Isso é alarme, não resultado.

É o passo mais barato e de maior retorno do projeto inteiro.

## Fase 11: avaliação

Mantenha pelo menos: acurácia do argmax, no geral e por primitiva e por família;
acurácia soft, Brier, NLL ou KL, e variação total para alvos de distribuição (a
última não está implementada aqui); ECE sobre a probabilidade máxima; para
`score`, erro absoluto médio do valor esperado (a proporção dentro de um nível não
está implementada aqui); robustez à ordem sob permutação; cobertura seletiva nos
80% e 50% mais confiantes; e percentis de latência com 1, 5, 10 e 50 perguntas
por chamada.

Baselines obrigatórios: o chute uniforme, a priori por pergunta ou a classe
majoritária, e o teto de concordância do anotador ou do professor (este último não
está implementado aqui). Sem o baseline de classe majoritária, ninguém teria
notado que os checkpoints base do Laya perdem para ele em famílias nunca vistas.

Ao comparar modelos, use perguntas idênticas byte a byte e seed fixa.

## Fase 12: runtime e empacotamento

Formato do checkpoint: um arquivo de config com o identificador do encoder, o
número de camadas da cabeça, os dois limites de tokens, os custos de ação e as
temperaturas (aqui a precisão de carga é resolvida a partir do dispositivo, então
não é guardada); um arquivo único de pesos em safetensors com tudo dentro, encoder
incluído; uma pasta só com a arquitetura do encoder, para o carregamento nunca
baixar pesos pré-treinados que seriam sobrescritos de qualquer forma; e o
tokenizer.

Valide antes de carregar: chaves obrigatórias na config, prefixos esperados nos
pesos, formato de cada tensor, carga estrita. Uma mensagem de erro clara vale ouro
quando alguém aponta para o checkpoint errado.

Dispositivo e precisão: tente CUDA, depois MPS, depois CPU. bfloat16 só em CUDA
com capability 8 ou mais, float16 abaixo disso, float32 em CPU e MPS, autocast só
em CUDA. Se faltar memória, caia para CPU e informe o motivo real e o custo.

Um detalhe de encoder específico: desligue a compilação automática se o seu
encoder tiver. Ela perde em lote pequeno e pode travar em algumas plataformas.

Sobre o campo de confiança, cuidado. Probabilidade máxima e um menos a entropia
normalizada são escalas diferentes, e a de entropia não é "probabilidade de estar
certo". Exponha as duas e documente qual é qual, e ponha o limiar sobre a que foi
calibrada.

## Fase 13: produção

- Gate: acima do limiar age sozinho, abaixo vai para fila humana. O limiar vem da
  conta de custo da fase 8, por tipo de decisão.
- Logue a distribuição inteira, não só o rótulo. É isso que permite recalibrar
  depois.
- A fila humana produz rótulo. Use para reajustar as temperaturas de tempos em
  tempos e para vigiar desvio de calibração.
- Carregue o modelo uma vez e mantenha residente.
- Para um modelo de uma língua só, a detecção de script por faixa Unicode é uma
  guarda de entrada exata e que custa microssegundos. Heurística de língua por
  stopwords é frágil; use um detector de verdade se precisar separar línguas de
  alfabeto latino.

## Ordem de execução

1. Contrato e formato de sequência fechados, com testes do builder: truncamento,
   muitas opções, marcador fora do limite, token de máscara na entrada, critérios
   estruturados.
2. Encoder escolhido com fertilidade de tokenizer medida em texto real.
3. Conversor de dataset para perguntas tipadas, com aumentações e held-out por
   família.
4. Baseline direto. Critério de saída: bater a classe majoritária com folga nas
   famílias vistas.
5. Versão com política (RLCD). Critério: bater o baseline em ECE ou NLL no
   held-out, senão não entra.
6. Temperatura por bucket em held-out. Critério: o ECE cai e nenhum valor encosta
   no limite.
7. Avaliação completa, incluindo robustez à ordem e cobertura seletiva.
8. Cabeça de ação, só se bater o gating por probabilidade máxima.
9. Runtime com validação de checkpoint, subido e exercitado de verdade, latência
   medida.
10. Fine-tune por domínio como etapa de produto, repetindo os passos 6 e 7.
