# Checkpoints

A checkpoint is a folder. It carries everything needed to run offline: config,
weights, temperatures and tokenizer.

```
checkpoint/
  rlcd.json            format version, model config, sequence config, temperature map
  model.safetensors    every weight, encoder included, one file
  encoder_config/      architecture only, so loading never downloads base weights
  tokenizer/           tokenizer files
  training.json        written by rlcd-train: seed, mode, hyperparameters, metrics
```

## Metadata

```json
{
  "format_version": 1,
  "model": {
    "encoder_name": "google-bert/bert-base-multilingual-cased",
    "num_decision_layers": 2,
    "dropout": 0.1,
    "temperature": 1.0,
    "action_costs": {"gain": 1.0, "loss": 3.0, "escalate": 0.5}
  },
  "sequence": {"max_length": 512, "header_budget": 192, "...": "..."},
  "temperature": {"default": 1.0, "by_type": {"choice": 0.83}, "by_bucket": {"noul:2": 1.04}}
}
```

`temperature` takes precedence in this order: the `(type, option count bucket)`
cell, then the type, then `default`. Buckets are the ranges `2`, `3-5`, `6-10`
and `11+`.

## Loading validates before it loads

`load_checkpoint` refuses a folder that is not a checkpoint, an unsupported format
version, a missing weights or encoder-config file, and a state dict whose keys do
not match the architecture. The error names the missing and unexpected keys. A
clear error is worth a lot when someone points at the wrong folder.

## Saving

```python
from rlcd import save_checkpoint

save_checkpoint(
    "artifacts/my-model",
    model,
    sequence_config=sequence_config,
    temperature=temperature,
    tokenizer=tokenizer,
)
```

The path must not exist. A partial failure removes the folder instead of leaving
a half-written checkpoint.

`save_file` moves the tensors to CPU itself, so a GPU-resident model saves fine;
moving it explicitly first keeps the peak host memory predictable.

## Loading

```python
from rlcd import load_checkpoint, load_tokenizer, SequenceBuilder

checkpoint = load_checkpoint("artifacts/my-model", device="cpu")
tokenizer = load_tokenizer("artifacts/my-model")
builder = SequenceBuilder(tokenizer, checkpoint.sequence_config)
```

`load_checkpoint` returns the model in eval mode, plus the model config, sequence
config and temperature map. It never touches the network.

## Device and precision

`resolve_device` tries CUDA, then MPS, then CPU. `resolve_dtype` returns bfloat16
on CUDA capability 8 or more, float16 below that, and float32 on CPU and MPS.
`autocast_context` only wraps CUDA.

## Shipping

A checkpoint does not inherit the code's license. Before publishing one, confirm
that the encoder weights, the tokenizer, the training data and the resulting
weights are all compatible with the license you intend to publish under. Record
the base encoder and the dataset provenance in the model card, next to the
held-out metrics and their baselines.
