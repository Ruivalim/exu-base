"""Exu: a toolkit for calibrated decision models.

Encoder-only typed-decision models that return probability distributions instead
of generated text, trained with a strictly proper scoring rule as the reward.

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

from .calibration import (
    TemperatureFit,
    TemperatureMap,
    fit_temperature,
    fit_temperature_map,
)
from .checkpoint import Checkpoint, load_checkpoint, load_tokenizer, save_checkpoint
from .data import DecisionDataset, TrainingBatch, TrainingExample, load_jsonl, write_jsonl
from .metrics import (
    DecisionMetrics,
    classification_metrics,
    majority_class_baseline,
    prior_baseline,
    random_baseline,
    selective_coverage,
)
from .model import ActionCosts, DecisionOutput, ExuConfig, ExuModel, masked_softmax
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
from .sequence import EncodedQuestion, SequenceBuilder, SequenceConfig
from .types import DecisionQuestion, DecisionType, Option

__all__ = [
    "ActionCosts",
    "Checkpoint",
    "Decision",
    "DecisionDataset",
    "DecisionMetrics",
    "DecisionOutput",
    "DecisionQuestion",
    "DecisionRuntime",
    "DecisionType",
    "EncodedQuestion",
    "Option",
    "PolicyConfig",
    "PolicyMetrics",
    "ExuConfig",
    "ExuModel",
    "SequenceBuilder",
    "SequenceConfig",
    "TemperatureFit",
    "TemperatureMap",
    "TrainingBatch",
    "TrainingExample",
    "classification_metrics",
    "composite_score",
    "fit_temperature",
    "fit_temperature_map",
    "group_advantages",
    "load_checkpoint",
    "load_jsonl",
    "load_tokenizer",
    "log_score",
    "majority_class_baseline",
    "masked_softmax",
    "policy_gradient_loss",
    "proper_scoring_loss",
    "prior_baseline",
    "random_baseline",
    "ranked_probability_score",
    "save_checkpoint",
    "selective_coverage",
    "sigma_for",
    "spherical_score",
    "write_jsonl",
]

__version__ = "0.1.0"
