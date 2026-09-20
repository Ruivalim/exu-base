# Documentation

Two ways in. If you are here to train something, follow the
[usage guide](usage-guide.md) step by step. If you want the whole picture first,
with the reasoning and the traps, read the [RLCD guide](rlcd-guide.md). Then pick
the reference that matches what you are doing.

| Document | Read it when |
| --- | --- |
| [usage-guide.md](usage-guide.md) | You have data and want a trained, calibrated checkpoint: the commands, the report, the failure modes |
| [rlcd-guide.md](rlcd-guide.md) | You are starting a model and want the whole picture, with the traps |
| [algorithm.md](algorithm.md) | You want the reward and policy-gradient math, and what the code does with it |
| [dataset-format.md](dataset-format.md) | You are writing or converting data |
| [checkpoints.md](checkpoints.md) | You are saving, loading or shipping a model |
| [evaluation.md](evaluation.md) | You are about to claim a number, or decide whether to ship |
| [rlcd-guide.pt-BR.md](rlcd-guide.pt-BR.md) | You prefer to read the walkthrough in Portuguese |
| [usage-guide.pt-BR.md](usage-guide.pt-BR.md) | You prefer the training walkthrough in Portuguese |

## The short version

1. Fix the contract: state plus typed question in, distribution out.
2. Pick a bidirectional encoder with a mask token and a tokenizer that is
   efficient in your language.
3. Build one sequence per question with a marker per option, under an explicit
   token budget.
4. Convert labeled data into typed questions with distributions as targets.
5. Train the direct baseline first. It is the bar.
6. Add the RLCD policy and keep it only if it beats the baseline on held-out ECE
   or NLL.
7. Fit temperatures on held-out data, per type and option-count bucket.
8. Evaluate with the trivial baselines beside your numbers.
9. Export a portable checkpoint and exercise the real binary.
