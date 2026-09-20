from __future__ import annotations

import pytest

from rlcd import DecisionQuestion, DecisionType, Option


def test_score_creates_ordered_levels() -> None:
    question = DecisionQuestion.score("Rate the risk", ["low", "high"])

    assert question.kind is DecisionType.SCORE
    assert [option.render() for option in question.options] == [
        "Level 0: low",
        "Level 1: high",
    ]


def test_score_accepts_a_language_specific_level_prefix() -> None:
    question = DecisionQuestion.score("Avalie o risco", ["baixo", "alto"], level_prefix="Nível")

    assert question.options[0].render() == "Nível 0: baixo"


def test_noul_has_exactly_two_fixed_options() -> None:
    question = DecisionQuestion.noul("Is this fraud?")

    assert question.kind is DecisionType.NOUL
    assert [option.name for option in question.options] == ["No", "Yes"]
    assert question.option_count == 2


def test_noul_rejects_a_different_option_count() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        DecisionQuestion(DecisionType.NOUL, "q", (Option("A"), Option("B"), Option("C")))


def test_question_rejects_duplicate_option_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        DecisionQuestion.choice("Pick", [Option("A"), Option("A")])


def test_question_rejects_blank_instruction() -> None:
    with pytest.raises(ValueError, match="instruction"):
        DecisionQuestion.choice("   ", [Option("A"), Option("B")])


def test_option_drops_a_blank_description() -> None:
    assert Option("A", "   ").render() == "A"
    assert Option("A", "criterion").render() == "A: criterion"
