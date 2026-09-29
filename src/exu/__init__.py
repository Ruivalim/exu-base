"""Exu: a toolkit for calibrated decision models.

Typed-decision models that return probability distributions instead of
generated text, trained with a strictly proper scoring rule as the reward.

Quick start::

    from exu import DecisionQuestion, DecisionRuntime, Option

    runtime = DecisionRuntime.load("artifacts/my-checkpoint")
    question = DecisionQuestion.choice(
        "Where should this ticket go?",
        [Option("billing", "payment, invoice, refund"), Option("support", "access or outage")],
    )
    decision = runtime.decide("I was charged twice for the same invoice.", question)
    print(decision.label, decision.confidence)

Training is a CLI::

    exu-train --mode rlcd --train data.jsonl --validation data.jsonl ...
    exu-evaluate --checkpoint artifacts/my-checkpoint --data data.jsonl
"""

from .architecture import DecisionModel, build_model, builder_for, model_from_pretrained
from .bandit import (
    BanditConfig,
    bandit_loss,
    estimate_target,
    simulated_bandit_loss,
    verdict_loss,
)
from .calibration import (
    TemperatureFit,
    TemperatureMap,
    fit_temperature,
    fit_temperature_map,
)
from .checkpoint import Checkpoint, load_checkpoint, load_tokenizer, save_checkpoint
from .data import DecisionDataset, TrainingBatch, TrainingExample, load_jsonl, write_jsonl
from .decoder import DecoderDecisionModel, DecoderSequenceBuilder, EncodedOptions
from .metrics import (
    DecisionMetrics,
    ReferencePrior,
    classification_metrics,
    fit_reference_prior,
    prior_in_sample_baseline,
    selective_coverage,
    uniform_baseline,
)
from .model import (
    ActionCosts,
    DecisionOutput,
    ExuConfig,
    ExuModel,
    masked_log_softmax,
    masked_softmax,
)
from .policy import (
    PolicyConfig,
    PolicyMetrics,
    group_advantages,
    policy_gradient_loss,
    sigma_for,
)
from .runtime import Decision, DecisionRuntime
from .scoring import (
    composite_score,
    log_score,
    proper_scoring_loss,
    ranked_probability_score,
    spherical_score,
)
from .sequence import DecisionBuilder, EncodedQuestion, SequenceBuilder, SequenceConfig
from .types import DecisionQuestion, DecisionType, Option

__all__ = [
    "ActionCosts",
    "BanditConfig",
    "Checkpoint",
    "Decision",
    "DecisionBuilder",
    "DecisionDataset",
    "DecisionMetrics",
    "DecisionModel",
    "DecisionOutput",
    "DecisionQuestion",
    "DecisionRuntime",
    "DecisionType",
    "DecoderDecisionModel",
    "DecoderSequenceBuilder",
    "EncodedOptions",
    "EncodedQuestion",
    "Option",
    "PolicyConfig",
    "PolicyMetrics",
    "ReferencePrior",
    "ExuConfig",
    "ExuModel",
    "SequenceBuilder",
    "SequenceConfig",
    "TemperatureFit",
    "TemperatureMap",
    "TrainingBatch",
    "TrainingExample",
    "bandit_loss",
    "build_model",
    "builder_for",
    "classification_metrics",
    "composite_score",
    "estimate_target",
    "fit_temperature",
    "fit_reference_prior",
    "fit_temperature_map",
    "group_advantages",
    "load_checkpoint",
    "load_jsonl",
    "load_tokenizer",
    "log_score",
    "masked_log_softmax",
    "masked_softmax",
    "model_from_pretrained",
    "policy_gradient_loss",
    "proper_scoring_loss",
    "prior_in_sample_baseline",
    "ranked_probability_score",
    "save_checkpoint",
    "selective_coverage",
    "sigma_for",
    "simulated_bandit_loss",
    "spherical_score",
    "uniform_baseline",
    "verdict_loss",
    "write_jsonl",
]

__version__ = "0.1.0"
