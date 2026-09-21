# Building a calibrated decision model with Exu

A phase-by-phase walkthrough. It is agnostic: no language, no domain, no encoder
is assumed. Everything below is either implemented in this repository or marked
as a design decision you have to make.

Exu implements RLCD, *Reinforcement Learning for Calibrated Decisions*, the
training method [TypeSafe](https://typesafe.ai/) uses for Jev. The reward is a
strictly proper scoring rule, so reporting honest probabilities is the only way
to maximize it.

The typed-decision contract itself comes from [TypeSafe's
Jev](https://typesafe.ai/): the `choice`, `score` and `noul` primitives, answers
that carry a distribution plus a separate confidence, and gating action on that
confidence. The open build recipe followed here is
[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0, by Nandakishor): the
sequence layout with a mask marker per option, the token budget, the composed
reward and the temperature buckets. The reference numbers quoted below are
Laya's, not measured in this repository. The README has the full credit.

One related paper, cited for the routing idea rather than for the design here:
Nandakishor M, *Confidence-Aware Routing for Large Language Model Reliability
Enhancement* ([arXiv:2510.01237](https://arxiv.org/abs/2510.01237), 2025), by the
same author as Laya. It estimates confidence before generation and routes a
query across four pathways: local generation, retrieval, a larger model, or human
review. This repository implements none of that; its act-or-escalate gate is the
same idea reduced to one encoder.

## The shape of the thing

```
state + typed question
        |
        v
one sequence per question:  type | instruction | [MASK] option | [MASK] option | state
        |
        v
bidirectional encoder (full fine-tune)
        |
        v
question-type embedding, two extra transformer layers
        |
        +--> hidden state at each marker -> scorer -> one logit per option
        |                                              |
        |                                   logits / temperature -> softmax
        |
        +--> CLS + distribution statistics -> act or escalate
```

Why bother: a reflex decision (route a ticket, flag fraud, score urgency) needs
probabilities, not prose. Generating a token costs hundreds of milliseconds, needs
a parser, and the confidence a chat model writes is more text. Here the output is
already a distribution, and nothing can be hallucinated.

## Phase 1: close the contract

Decide what goes in and what comes out before touching a model.

| Primitive | Question | Output |
| --- | --- | --- |
| `choice` | pick one of N explicit options | distribution over options |
| `score` | place the state on an ordinal rubric | distribution over levels |
| `noul` | boolean question | distribution over No and Yes |

All three are the same mechanism: score options, softmax. `noul` is a `choice`
with two fixed options; `score` is a `choice` whose options are "level i: text".
What differs is the type word in the prompt, the type embedding, an extra reward
term for the ordinal case, and the output post-processing.

Contract decisions worth copying:

- Questions and options are defined at request time. The model reads option text,
  so a new task does not need a new output layer or a retraining run.
- The state can be a string, an object or a list. Structures become JSON,
  serialized without escaping non-ASCII, or `ç` and `ã` become escape sequences
  and burn the token budget.
- Structured criteria also become compact JSON, never a language's internal
  representation. Zero and false are legitimate values; only null and empty
  string mean "no description".
- One sequence per question. N questions about the same state become a batch of N
  sequences in one forward pass. The cost is re-encoding the state N times, which
  is why latency grows with the number of questions even in a batch.
- The response reports input tokens and zero output tokens, which keeps cost
  accounting simple.

## Phase 2: choose the encoder

Criteria, in order:

1. Bidirectional (encoder-only). Every token must see the state and all options
   at once.
2. A mask token in the vocabulary. The marker trick reuses it, and the encoder
   already learned to treat it as a position where something must be predicted.
3. A tokenizer efficient in your language. Measure tokens per word on real text
   from your domain.
4. Enough context for instructions, options and state together.
5. Size against latency, plus license.

Measure before committing. Encoders that are excellent in one language collapse
in another. Pick two candidates, measure tokenizer fertility on your real text,
run a short training on each, and then choose. The default in `ExuConfig` is a
multilingual BERT as a neutral starting point, not a recommendation.

For a single language, a specialized encoder usually wins on both speed and
accuracy. Check the license and the mask token for any candidate before use.

## Phase 3: design the sequence

Per question: CLS, the type word and instructions, a separator, then each option
preceded by a mask marker and followed by a separator, then the state, then a
final separator.

The marker position is saved, because that is where the option's logit comes
from. Because the encoder is bidirectional, the hidden state at the marker
already carries the option, the competing options, the instruction and the state.

Token budget, where the architecture's real limit lives:

- Two limits: a maximum total length and a budget for the header (instructions
  plus options). This repository defaults to 512 and 192.
- Each option has its own ceiling (48 tokens here, counting the marker). The
  instruction is guaranteed at least 8 tokens of whatever is left of the header.
- Who yields first is the option, not the instruction: once the guarantee above
  is reserved, options are capped to the remainder, with a floor of 3 tokens of
  text each. If even that does not fit, `SequenceBuilder` raises instead of
  silently shrinking every option into something unreadable.
- The state takes the rest. At inference it is truncated from the right in this
  implementation. Left truncation keeps recent turns and makes sense for
  conversations.
- If a marker cannot fit inside the maximum length, the builder raises instead of
  answering wrongly in silence.

That last point bites earlier than it looks. The contract accepts up to 255
options, but the default header budget holds roughly thirty of them: past that,
`build` raises `too many options for configured header_budget`. Laya instead
shrinks every option to a floor and answers, which is what produces its flat
0.425 on Banking77's 77 labels; here the failure is explicit, and the fix is to
raise `header_budget` or to decide in two stages. Keep `choice` below roughly 20
options, and above that go hierarchical.

Security: remove the literal mask token string from all untrusted text. Without
it, an input can inject fake markers. `SequenceBuilder` does this for state,
instruction and option descriptions.

## Phase 4: the decision head

On top of the encoder, trained from scratch:

1. Add a question-type embedding (three entries) to every position.
2. Two extra pre-norm transformer layers, one attention head per 64 dimensions,
   feed-forward of 4x the hidden size, dropout 0.1, respecting the padding mask.
   Their job is to let markers talk to each other, already conditioned on type.
3. Read the hidden state at each marker position.
4. A small scorer (norm, linear, GELU, linear to one scalar) turns each marker
   into a logit.
5. Options that exist only because of batch padding get a very negative logit.
6. Softmax over each question's options.

`--scorer marker-cls` adds one term to every option's logit, and leaves the rest as
it is:

```
logit_k += < W e_k , LayerNorm(h_cls) > / sqrt(H)
```

`e_k` is the mean of the *input* embeddings of the option's own text tokens, centred
across the options of the question and rescaled.

It exists because the marker scorer starts from a symmetric saddle. The scorer is
shared, so at the start every option gets nearly the same logit, and the gradient
that reaches anything the options have in common is `sum_k (q_k - y_k) * c = 0`.
Something has to make the options distinguishable first. A skewed label prior does
it, because scoring the frequent option higher pays at once. Lexical overlap
between an option and the state does it too ("gratitude" and "thank you"). With
**balanced labels and no such overlap** nothing does, and the model stays at the
uniform guess however long it trains. Measured: MultiNLI with balanced classes
stays at 0.32 accuracy and reaches 0.69 when only its training labels are skewed,
and BoolQ learns at its natural 62% of yes and collapses to `log 2` when its
training split is balanced. `marker-cls` breaks the symmetry by construction and
learns in all four cases (0.75 on balanced MultiNLI). Two details carry it: the
option is identified by input embeddings, which are the same in every example, and
the identities are centred, because what the options share cancels in the softmax.
Numbers in `BENCHMARKS.md`. The default is still `marker`, which was as good or
slightly better where it does learn.

Order of magnitude: the extra head is tens of millions of parameters, the scorer
about one million, against hundreds of millions in the encoder. The encoder is not
frozen: it receives a full fine-tune with a lower learning rate than the head.

## Phase 5: data

Any labeled dataset becomes typed questions. Classification becomes `choice` with
the labels as options and a hand-written description for each. A binary label
becomes `noul`. A rating or scale becomes `score`. Textual inference becomes a
three-option `choice`. The target can be one-hot or a soft distribution when there
are several annotators or a teacher model.

Augmentations, so the model does not learn a shortcut:

- Shuffle option order every pass. Without it the model learns position. In Laya
  the answer changed in 15% to 23% of cases under permutation.
- Paraphrase instructions.
- Alternate the state between raw text and nested JSON.
- Inject distractor questions.

Split held-out by entire task family, not just by example. That is the only way
to measure real zero-shot behavior, and Laya's numbers are a useful reality
check: 83.8% on seen tasks against 65.1% on unseen ones, with base checkpoints
scoring below the majority-class baseline on unseen families.

## Phase 6: the reward

A scoring rule takes a reported distribution `q` and the outcome `y` and returns
a number. It is strictly proper when the expected score is maximized only by
reporting the true distribution. That property ties reward to honesty.

The counterexample is instructive: a binary reward (1 if correct, 0 otherwise)
has expected value linear in `q`, so the optimum is to put all mass on the most
likely class. Naive RL maximizes accuracy and destroys calibration.

The composite reward here adds three pieces:

- log score: the log of the probability given to the target (a weighted sum for
  soft targets). It punishes low probability on what happened. It is computed in
  log space and has no floor, so a confident miss keeps its real price and keeps
  teaching.
- spherical score: the inner product of target and `q` divided by the norm of
  `q`, in `[0, 1]`. It rewards mass in the right place without the log's gradient
  spikes.
- ranked probability score, `score` questions only: quadratic distance between
  cumulative distributions, divided by `K - 1`. It teaches that missing by one
  level beats missing by three.

See [algorithm.md](algorithm.md) for the exact definitions and weights.

## Phase 7: the training loop

The policy samples perturbations instead of tokens. For each question:

1. Sample `G` Gaussian perturbations of the logits with standard deviation
   `sigma`. Mask them to the valid options and project them to sum to zero,
   because a constant added to logits does not change the softmax.
2. Softmax each perturbed version: `G` candidate distributions per question.
3. Score each candidate against the target with the composite reward, no gradient.
4. Advantage: reward minus the group baseline, normalized by a standard
   deviation (batch-wide in Laya's fine-tune, per-group in GRPO).
5. Loss: negative mean of advantage times the Gaussian log-probability of the
   sample given the current logits. The sample is detached, so the gradient flows
   only through the current logits.

The intuition: the gradient of the log-probability points along the noise.
Perturbations that scored above the group mean pull the logits toward themselves,
and the ones below push away.

`sigma` anneals over training, and the hyperparameters differ between base and
fine-tune training. See `PolicyConfig.base()` and `PolicyConfig.finetune()`.

A critical reading, to save you time. The reward is differentiable in the logits,
so you can backpropagate directly with no sampling. The log score is exactly the
cross-entropy with the sign flipped. What the RL does is estimate, noisily, the
gradient of a smoothed version of the same objective plus the spherical and RPS
terms. The claim that cross-entropy makes a model overconfident and a scoring rule
does not is not true at the level of the objective: with a one-hot target and
separable data, both push toward certainty. The real differences are the noise
acting as a regularizer and the extra terms. So run the direct
baseline first, then the policy version, and compare ECE and NLL on held-out. If
RL does not win clearly, keep the simple one. That is the order this repository
encourages with `--mode baseline` and `--mode rlcd`.

## Phase 8: act or escalate

Train the action head with a cost matrix: acting and being right earns a gain,
acting and being wrong costs a loss, escalating costs something. The break-even
point follows: act when the probability of being right passes
`(loss - escalate) / (gain + loss)`. With gain 1, loss 3 and escalate 0.5, that is
0.625. Changing the costs changes the threshold, which is how a business rule
enters the model.

This repository ships the cost model and the trivial baseline (`should_act` on
the calibrated maximum probability). Learned action-head training is left as an
extension: the learned head only earns its place if it beats that baseline.

Two warnings. The action probability of a fine-tuned checkpoint deserves
suspicion until the head is retrained after the encoder changes, because its
input distribution moved. And the threshold must be applied to the same quantity
the calibration measured, which is the maximum probability, not the entropy
confidence.

For the routing version of this idea in an LLM setting, with four pathways and
the confidence estimated before generation, see Nandakishor M, *Confidence-Aware
Routing for Large Language Model Reliability Enhancement*
([arXiv:2510.01237](https://arxiv.org/abs/2510.01237), 2025). Different setting,
same shape of decision, and not implemented here.

## Phase 9: multi-turn (optional)

To predict a conversation outcome (conversion, churn) turn by turn:

- Slice each conversation into prefixes and show the model only up to the current
  turn. Feeding the whole conversation at turn 1 leaks the future.
- The target for each prefix comes from TD(lambda): the last prefix gets the real
  outcome, and each earlier one gets a mix of the model's prediction for the next
  prefix and that prefix's target. With lambda equal to 1 every prefix is trained
  directly against the final outcome, without bootstrapping on the model's own
  predictions.
- Left truncation of the state makes sense here, to keep recent turns.

This is not implemented in this repository.

## Phase 10: temperature calibration

After training, with the weights frozen, fit a scalar `T` that divides the logits
before the softmax. Run the model on a held-out set, keep logits and targets,
minimize the negative log-likelihood over `log T`, clamp the result, and require a
minimum number of samples, falling back to `T = 1` below it.

One `T` is not enough. Confidence miscalibration depends on how many options a
question has, so temperatures are bucketed by question type and option-count
range, with a per-type fallback. Buckets take precedence over the per-type value.

Three traps, all visible in Laya:

- Fitting on a slice of the training set. The model is more confident about what
  it has seen, so `T` lands near 1 and calibration looks better than it is. Use
  held-out data.
- A stale bucket map shadowing freshly fitted per-type values. Clear it when it
  is not wanted.
- A fitted value sitting on a bound. That is an alarm, not a result.

This is the cheapest and highest-return step in the whole project.

## Phase 11: evaluation

Keep at least: accuracy of the argmax, overall and per primitive and per family;
soft accuracy, Brier, NLL or KL, and total variation for distribution targets
(the last is not implemented here); ECE over the maximum probability; for
`score`, mean absolute error of the expected level (the proportion within one
level is not implemented here); order robustness under permutation; selective
coverage at the most confident 80% and 50%; and latency percentiles for 1, 5, 10
and 50 questions per call.

Mandatory baselines: uniform, the per-question prior or majority class, fitted on
training labels and never on the rows being scored, and the
agreement ceiling of the annotator or teacher (this last one is not implemented
here). Without the majority-class baseline, nobody would have noticed that Laya's
base checkpoints lose to it on unseen families.

When comparing models, use identical questions byte for byte and a fixed seed.

## Phase 12: runtime and packaging

Checkpoint format: a config file with the encoder identifier, head layer count,
both token limits, action costs and temperatures (the checkpoint is not stored at
a fixed precision: it loads in float32 and the device chooses the autocast dtype
for a forward pass, so there is nothing to record); one safetensors
weight file with everything inside, encoder included; one folder with the encoder
architecture only, so loading never downloads pretrained weights that would be
overwritten anyway; and the tokenizer.

Validate before loading: required config keys, expected weight prefixes, tensor
shapes, strict load. A clear error is worth a lot when someone points at the
wrong checkpoint.

Device and precision: try CUDA, then MPS, then CPU. bfloat16 only on CUDA
capability 8 or more, float16 below that, float32 on CPU and MPS, autocast only on
CUDA. If memory runs out, fall back to CPU and report the real reason and cost.

One encoder-specific detail: turn off automatic compilation if your encoder has
it. It loses on small batches and can hang on some platforms.

On the confidence field, be careful. Maximum probability and one minus normalized
entropy are different scales, and the entropy one is not "probability of being
right". Expose both and document which is which, and put thresholds on the one
that was calibrated.

## Phase 13: production

- Gate: above the threshold act alone, below it queue for a human. The threshold
  comes from the cost calculation in phase 8, per decision type.
- Log the whole distribution, not just the label. That is what makes later
  recalibration possible.
- The human queue produces labels. Use them to refit temperatures periodically
  and to watch for calibration drift.
- Load the model once and keep it resident.
- For a single-language model, script detection by Unicode range is an exact and
  microsecond-cheap input guard. Loose language heuristics by stopwords are
  fragile; use a real detector if you must separate Latin-script languages.

## Execution order

1. Contract and sequence format closed, with builder tests: truncation, many
   options, marker out of range, mask token in the input, structured criteria.
2. Encoder chosen with tokenizer fertility measured on real text.
3. Dataset converter into typed questions, with augmentations and held-out by
   family.
4. Direct baseline. Exit criterion: beat the majority class with margin on seen
   families.
5. Policy version (RLCD). Criterion: beat the baseline on held-out ECE or NLL, or
   it does not ship.
6. Temperature per bucket on held-out. Criterion: ECE drops and no value sits on
   a bound.
7. Full evaluation, including order robustness and selective coverage.
8. Action head, only if it beats gating by maximum probability.
9. Runtime with checkpoint validation, started and exercised for real, latency
   measured.
10. Per-domain fine-tune as a product step, repeating steps 6 and 7.
