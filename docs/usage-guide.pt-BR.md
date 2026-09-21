# Usando este repositório para treinar um modelo

Este é o guia procedural: você tem uma tarefa e alguns dados rotulados, e quer um
checkpoint treinado e calibrado no fim. É de propósito curto em teoria.

- Para *por que* o desenho é o que é, e para cada armadilha que vale conhecer,
  leia o [exu-guide.pt-BR.md](exu-guide.pt-BR.md).
- Para o contrato campo a campo dos dados, o formato do checkpoint e cada
  métrica, siga os links de cada passo abaixo. Nada aqui repete aquilo.

Todo comando deste arquivo foi executado neste repositório antes de ser escrito.

## 0. Confira o ambiente primeiro

```bash
make setup    # uv sync --extra dev
make test     # 113 testes, poucos segundos
make smoke    # treina + calibra + avalia um modelo minúsculo na CPU
```

O `make smoke` é o importante: ele roda o pipeline de verdade, com tokenizer
real, encoder real e a política real, em `tests/fixtures/smoke.jsonl`. Se ele
passa, a máquina consegue treinar. Segundos na CPU, funciona sem GPU.

Ele escreve em `artifacts/`, que está no gitignore. Nada mais neste guia escreve
fora do diretório que você passar em `--output`.

## 1. Escreva os dados

Um registro JSONL por decisão. O alvo é uma distribuição, não um rótulo, então
alvo soft de vários anotadores ou de um modelo professor é cidadão de primeira
classe.

```json
{"id": "t-1", "state": "Fui cobrado duas vezes pela mesma fatura.",
 "question": {"kind": "choice", "instruction": "Para onde vai este ticket?",
   "options": [{"name": "cobrança", "description": "pagamento, fatura ou reembolso"},
               {"name": "suporte", "description": "acesso ou indisponibilidade"}]},
 "target": [1.0, 0.0], "split": "train", "family": "roteamento", "language": "pt"}
```

Regras que importam, com a lista completa em [dataset-format.md](dataset-format.md):

- `target` precisa somar um, não pode ser negativo, e precisa ter um valor por
  opção. O treino recusa qualquer outra coisa, com o `id` do registro na mensagem.
- `split` é o que `--train-split`, `--validation-split` e `--calibration-split`
  selecionam. Pode juntar vários splits num arquivo só; os flags filtram.
- `family` é a família de tarefa. **Separe o held-out por família, não por
  exemplo**, senão o seu número de zero-shot é mentira. Esta é a forma mais comum
  de enganar a si mesmo com esta arquitetura.
- Escreva a `description` de cada opção à mão. É o que o modelo lê de fato na hora
  de decidir, e normalmente é dali que vem a acurácia.

Para converter um dataset que já existe, monte objetos `TrainingExample` direto
de registros, ou com `DecisionQuestion`:

```python
from exu import DecisionQuestion, Option, TrainingExample, write_jsonl

question = DecisionQuestion.choice(
    "Para onde vai este ticket?",
    [
        Option("cobrança", "pagamento, fatura ou reembolso"),
        Option("suporte", "acesso ou indisponibilidade"),
    ],
)
examples = [
    TrainingExample.from_record(
        {
            "id": f"t-{index}",
            "state": text,
            "question": {
                "kind": question.kind.value,
                "instruction": question.instruction,
                "options": [
                    {"name": option.name, "description": option.description}
                    for option in question.options
                ],
            },
            "target": [1.0, 0.0],
            "split": "train",
            "family": "roteamento",
        }
    )
    for index, text in enumerate(texts)
]
write_jsonl("data/tickets.jsonl", examples)
```

O `TrainingExample.to_record()` devolve o mesmo dicionário, o que ajuda quando
você quer reescrever um arquivo com outra divisão de splits.

## 2. Escolha o encoder

`--encoder` aceita um id do Hugging Face ou uma pasta local. O default é
`google-bert/bert-base-multilingual-cased`: ponto de partida neutro, não
recomendação.

O encoder precisa ser bidirecional e o tokenizer dele precisa ter token de
máscara. O builder levanta `tokenizer must provide mask_token_id` caso contrário,
o que já elimina modelos decoder-only e encoders estilo Electra.

Meça antes de casar. A fertilidade do tokenizer no seu texto decide quanto estado
cabe no orçamento, e um checkpoint excelente numa língua pode desabar em outra.
Veja a fase 2 do [exu-guide.pt-BR.md](exu-guide.pt-BR.md).

## 3. Treine o baseline direto

A régua. Nada mais conta antes de esse número existir.

```bash
uv run exu-train \
  --mode baseline \
  --train data/tickets.jsonl --train-split train \
  --validation data/tickets.jsonl --validation-split validation \
  --calibration data/tickets.jsonl --calibration-split validation \
  --output artifacts/baseline \
  --encoder google-bert/bert-base-multilingual-cased \
  --epochs 4 --batch-size 8 --option-shuffle --calibrate
```

- `--output` não pode existir. Uma falha no meio remove a pasta em vez de deixar
  meio checkpoint.
- `--calibration` deve ser held-out. Ajustar temperatura nos dados de treino é a
  armadilha que faz a calibração parecer melhor do que é.
- `--option-shuffle` permuta as opções de uma pergunta a cada passada. Sem isso o
  modelo aprende posição em vez de critério.
- `--scorer marker-cls` é para dado de treino com **rótulos balanceados e sem
  sobreposição lexical** entre as opções e o estado. Ali o scorer default, `marker`,
  nunca sai do chute uniforme (NLL de validação preso exatamente em `log K`): o
  MultiNLI balanceado ficou em 0.32 de accuracy onde o `marker-cls` chegou a 0.75.
  Se a primeira época terminar em `log K`, é esta a flag a tentar. A escolha fica
  gravada no checkpoint, então a inferência não precisa de flag.
- `--device` é `auto` por padrão: CUDA, depois MPS, depois CPU.

Botões úteis, com os defaults: `--epochs 4`, `--batch-size 8`, `--grad-accum 1`,
`--seed 17`, `--encoder-lr 2.5e-5`, `--head-lr 1e-4`, `--weight-decay 0.01`,
`--max-grad-norm 1.0`, `--max-length 512`, `--header-budget 192`,
`--calibration-min-samples 64`, `--log-every 0`.

## 4. Treine a política

Mesmo pipeline, `--mode rlcd` no lugar de `baseline`, e os botões da política
passam a importar.

```bash
uv run exu-train \
  --mode rlcd \
  --train data/tickets.jsonl --train-split train \
  --validation data/tickets.jsonl --validation-split validation \
  --calibration data/tickets.jsonl --calibration-split validation \
  --output artifacts/exu \
  --encoder google-bert/bert-base-multilingual-cased \
  --epochs 4 --batch-size 8 --option-shuffle --calibrate \
  --samples-per-question 4 --sigma-start 0.4 --sigma-end 0.1 \
  --ce-weight 1.0 --advantage-norm batch
```

Os defaults são a receita do fine-tune: 4 amostras por pergunta, sigma caindo de
0.4 para 0.1, termo de cross-entropy com peso cheio e normalização de vantagem
pelo lote inteiro. O `PolicyConfig.base()` guarda a receita do treino base: 8
amostras, sigma de 1.0 a 0.3, sem cross-entropy.

Só mantenha a política se ela bater o baseline em ECE ou NLL no held-out. Muitas
vezes não bate, e isso é um resultado, não um fracasso. A leitura crítica do
porquê está no fim da fase 7 do [exu-guide.pt-BR.md](exu-guide.pt-BR.md).

## 5. Leia o relatório de treino

O `training.json` fica ao lado do checkpoint, dentro de `--output`. Ele registra a
config, uma entrada por época, as métricas de validação, e o ajuste de
temperatura quando `--calibrate` foi usado.

```bash
python -c "import json; d=json.load(open('artifacts/exu/training.json')); print(d['validation'])"
```

O que olhar:

- Bloco `validation`: `count`, `nll`, `brier`, `accuracy`, `soft_accuracy`, `ece`,
  mais `rps` e `ordinal_mae` em perguntas `score`. São os números para comparar
  com a rodada do baseline.
- `train_epochs[*].mean_reward`: deve subir, ou pelo menos não desabar.
- `train_epochs[*].confident_miss_rate`: a fração dos componentes reivindicados pelo
  alvo a que o modelo deu menos de `1e-4`, nas próprias previsões.
  `candidate_confident_miss_rate` é a mesma coisa sobre as candidatas amostradas, só
  no modo `rlcd`. Com rótulo hard, componente é linha. Uma taxa que não cai de uma
  época para outra aponta para linha com rótulo errado ou para um modelo que não
  consegue se recuperar. Com `--advantage-norm batch`, cada linha dessas também
  atrasa o resto do batch dela.
- `calibration.fits`: uma entrada por bucket, com `samples`, `nll_before`,
  `nll_after`, `fallback` e `at_bound`. Um ajuste com `at_bound: true` significa
  que a temperatura encostou no limite do clamp. Isso é sinal para olhar, nunca
  resultado para publicar. Um ajuste com `fallback: true` teve amostra de menos
  para ser levado a sério.

## 6. Avalie no held-out

```bash
uv run exu-evaluate \
  --checkpoint artifacts/exu \
  --data data/tickets.jsonl --split test \
  --order-permutations 4 --latency \
  --output artifacts/report.json
```

O relatório traz: métricas gerais, por tipo de pergunta, por família, os três
baselines triviais (uniforme, priori por pergunta, classe majoritária), cobertura
seletiva, robustez à ordem e percentis de latência. Cada campo está definido em
[evaluation.md](evaluation.md).

Como ler, nesta ordem:

1. **Bata os baselines.** Se `accuracy` ou `nll` não bate a classe majoritária e a
   priori por pergunta, o resto não importa.
2. **Olhe `ece` ao lado de `accuracy`.** Acurácia vem do argmax, ECE da
   probabilidade máxima. Modelo que acerta 80% das vezes com ECE 0.2 não deve ser
   liberado sozinho com base na própria confiança.
3. **`order_robustness.stability` perto de 1.** Abaixo de uns 0.9 o modelo está
   lendo posição, não critério. Treine com `--option-shuffle` e mais épocas.
4. **`coverage.top_0.5` acima de `coverage.all`.** Se responder só a metade mais
   confiante não aumenta a acurácia, a confiança não serve para gating.
5. **Latência**, se você pretende servir.

Cuidado com avaliar no mesmo split em que treinou. Os números vão ser ótimos e
sem significado.

## 7. Sirva o modelo

```python
from exu import DecisionQuestion, DecisionRuntime, Option

runtime = DecisionRuntime.load("artifacts/exu")
decision = runtime.decide(
    "Fui cobrado duas vezes pela mesma fatura.",
    DecisionQuestion.choice(
        "Para onde vai este ticket?",
        [
            Option("cobrança", "pagamento, fatura ou reembolso"),
            Option("suporte", "acesso ou indisponibilidade"),
        ],
    ),
)
print(decision.label, decision.probabilities, decision.confidence, decision.should_act)
```

- `runtime.decide_many([(estado, pergunta), ...])` responde um lote de perguntas
  num forward pass só. Prefira isso a um loop: o custo é reencodar o estado uma
  vez por pergunta nos dois casos, e o lote é bem mais rápido por pergunta.
- Carregue uma vez e mantenha o runtime residente. Uma carga fria custa segundos.
- `confidence` e `entropy_confidence` são escalas diferentes. O `should_act` usa
  a probabilidade máxima calibrada. Leia a seção de confiança do
  [evaluation.md](evaluation.md) antes de pôr um limiar de negócio em qualquer
  uma das duas.
- `decision.expected_level` só faz sentido em perguntas `score`.

### Pela linha de comando

O `exu-decide` é o mesmo runtime atrás de um CLI. Uma pergunta, escrita com flags:

```bash
exu-decide --checkpoint artifacts/exu \
  --state "Fui cobrado duas vezes pela mesma fatura." \
  --instruction "Para onde vai este ticket?" \
  --option "cobrança=pagamento, fatura ou reembolso" \
  --option "suporte=acesso ou indisponibilidade"
```

`--option NOME[=DESCRIÇÃO]` se repete, e só o primeiro `=` separa os dois.
`--kind score` recebe `--level TEXTO`, do menor para o maior, e `--kind noul`
responde `No` ou `Yes`, a menos que `--false-option` e `--true-option` digam outra
coisa. `--state-file` lê o estado de um arquivo.

Muitas perguntas, um registro JSON por linha, de um arquivo ou do stdin com `-`:

```bash
exu-decide --checkpoint artifacts/exu --input perguntas.jsonl --output decisoes.jsonl
```

O registro precisa de `state` e `question`, no formato do dataset. O `id` é
devolvido quando existe e o resto é ignorado, então um arquivo de dataset funciona
como está. A saída é um objeto JSON por pergunta, na ordem da entrada, com `label`,
`confidence`, `entropy_confidence`, `should_act`, `expected_level`, e
`probabilities` e `logits` indexados pelo nome da opção.

`--top-only` devolve só a vencedora: `{"label": "cobrança", "confidence": 0.87}`,
mais o `id` quando existe. `--metrics` acrescenta um objeto `metrics` a cada
resposta (`batch_ms`, `batch_size`, `per_question_ms`) e imprime uma linha JSON de
resumo no stderr (`questions`, `batches`, `load_ms`, `inference_ms`,
`first_batch_ms`, `questions_per_second`), então o stdout continua com um esquema
só. O tempo cobre a resposta inteira: montar as sequências, o forward e ler o
resultado de volta. O primeiro lote numa GPU também paga o aquecimento, por isso
sai à parte. Uma pergunta não tem latência própria dentro de um lote, então
`per_question_ms` é o tempo do lote dividido pelo tamanho dele.

A entrada inteira é validada antes do primeiro forward. Linha quebrada é nomeada
(`error: line 7: missing field 'question'`), o código de saída é 1 e nada é
respondido, então um pipeline nunca recebe meio resultado. Linha de comando
malformada sai com 2. Toda chamada carrega o checkpoint, o que custa segundos: um
serviço deve manter um `DecisionRuntime` residente.

O formato do checkpoint, a validação feita antes de carregar e as regras de
dispositivo e precisão estão em [checkpoints.md](checkpoints.md).

## 8. Quando dá errado

| Sintoma | Causa e o que fazer |
| --- | --- |
| `too many options for configured header_budget` | As opções da pergunta não cabem no cabeçalho. Com os defaults, umas 30 opções é o teto. Aumente o `--header-budget` (e o `--max-length` junto), ou decida em duas etapas. |
| `tokenizer must provide mask_token_id` | O encoder é decoder-only ou não tem token de máscara. Escolha um encoder bidirecional que tenha. |
| `each target distribution must sum to one` | Corrija o `target` do registro citado no erro. |
| `output path already exists` | O `--output` se recusa a sobrescrever. Escolha um diretório novo. |
| Acurácia no chute, validação parada | As opções não se distinguem: textos parecidos demais, descrição de menos, ou o orçamento espremeu. Confira o orçamento de cabeçalho e o `--option-shuffle`, depois o encoder. |
| ECE alto, acurácia boa | Calibre num held-out de verdade (`--calibration` mais `--calibrate`), e confira ajustes com `at_bound` no relatório. |
| `stability` baixa | O modelo aprendeu posição. Ligue o `--option-shuffle`. |
| CUDA sem memória | `--device cpu`, ou `--batch-size` menor, ou `--max-length` menor, ou um encoder menor. |
| `CUDA was requested but is unavailable` | Use `--device auto`. |
| A loss não anda | Confira a taxa de aprendizado do encoder contra a da cabeça, e se o `target` não é tudo zero ou tudo um em todos os registros. |

## 9. O que não commitar

Datasets, checkpoints, `artifacts/`, `runs/`, `_site/`, `dist/`. Tudo no
gitignore; mantenha assim. Datasets e checkpoints carregam licença própria, e um
checkpoint carrega a licença do encoder junto. Antes de publicar um artefato,
confirme que o encoder, o tokenizer, os dados e os pesos resultantes são
compatíveis.

Fixtures sintéticas pequenas são para encanamento. Não são evidência de
qualidade, e nenhum número medido nelas pertence a um README.
