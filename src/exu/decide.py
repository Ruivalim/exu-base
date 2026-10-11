"""Answer questions with a trained checkpoint, from the command line.

Two ways in, one way out:

- one question, written with flags (``--state``, ``--instruction``, ``--option``);
- many questions, one JSON record per line, from a file or from stdin. A record
  needs ``state`` and ``question``, in the dataset format. ``id`` is echoed when it
  is there, and everything else (``target``, ``split``...) is ignored, so a dataset
  file works as it is, ``context`` included. A record can instead carry ``questions``, a list of
  questions about its one state, each with an optional ``id`` of its own; its
  answers are named ``<record id>/<question id>``, or by the question's position
  from 0 when it has no id.

The questions of one record go into the same forward pass whenever the batch size
allows, so a decoder checkpoint reads their state once. The output is one JSON
object per question, in input order. The whole input is
parsed and validated before the first forward pass: a broken line is named and
nothing is answered, so a pipeline never receives half a result.

``--top-only`` keeps just the winning option and its calibrated probability.
``--metrics`` adds, to every answer, how long its batch took, and prints one JSON
summary line to stderr, so stdout stays one schema. The time covers the whole
answer: building the sequences, the forward pass and reading the result back. The
first batch on a GPU also pays for warm-up, which is why it is reported apart.

This is a thin shell around :class:`exu.runtime.DecisionRuntime`. A service that
answers requests should load the runtime once and keep it, not call this per
request: loading a checkpoint costs seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from .data import context_text, question_from_record, state_text
from .runtime import Decision, DecisionRuntime
from .types import DecisionQuestion, DecisionType, Option

Question = tuple[str | None, str, DecisionQuestion, str | None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer typed questions with an Exu checkpoint, one JSON object per question."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="checkpoint directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--batch-size", type=_positive_int, default=16, help="questions per forward pass"
    )
    parser.add_argument("--output", type=Path, help="write the JSON lines here instead of stdout")
    parser.add_argument(
        "--top-only",
        action="store_true",
        help="answer with the winning option and its probability only",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="time every batch, and print a JSON summary line to stderr",
    )

    many = parser.add_argument_group("many questions")
    many.add_argument(
        "--input", help="JSONL with one {state, question} record per line, or - for stdin"
    )

    one = parser.add_argument_group("one question")
    one.add_argument("--state", help="the text the question is about")
    one.add_argument("--state-file", type=Path, help="read the state from this UTF-8 file")
    one.add_argument("--context", help="what the answer depends on besides the state")
    one.add_argument("--context-file", type=Path, help="read the context from this UTF-8 file")
    one.add_argument("--kind", choices=[kind.value for kind in DecisionType], default="choice")
    one.add_argument("--instruction", help="the question the model reads")
    one.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAME[=DESCRIPTION]",
        help="one option of a choice question, repeat it. The description is the criterion",
    )
    one.add_argument(
        "--level",
        action="append",
        default=[],
        metavar="TEXT",
        help="one level of a score question, lowest first, repeat it",
    )
    one.add_argument("--false-option", default="No", help="noul only")
    one.add_argument("--true-option", default="Yes", help="noul only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    single = _single_question(parser, args)

    try:
        records = [[single]] if single else list(_read_records(args.input))
        questions = [question for record in records for question in record]
        if not questions:
            raise ValueError("the input has no questions")
        started = time.perf_counter()
        runtime = DecisionRuntime.load(args.checkpoint, device=args.device)
        load_ms = (time.perf_counter() - started) * 1000.0
        batch_times: list[float] = []
        with _destination(args.output) as stream:
            start = 0
            for chunk in _chunks(records, args.batch_size):
                started = time.perf_counter()
                decisions = _decide(runtime, chunk, first_line=start)
                batch_ms = (time.perf_counter() - started) * 1000.0
                batch_times.append(batch_ms)
                for (identifier, *_rest), decision in zip(chunk, decisions, strict=True):
                    row = _render(identifier, decision, top_only=args.top_only)
                    if args.metrics:
                        row["metrics"] = {
                            "batch_ms": batch_ms,
                            "batch_size": len(chunk),
                            "per_question_ms": batch_ms / len(chunk),
                        }
                    stream.write(json.dumps(row, ensure_ascii=False))
                    stream.write("\n")
                start += len(chunk)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.metrics:
        print(json.dumps(_summary(len(questions), load_ms, batch_times)), file=sys.stderr)
    return 0


def _summary(questions: int, load_ms: float, batch_times: Sequence[float]) -> dict[str, float]:
    inference_ms = sum(batch_times)
    return {
        "questions": questions,
        "batches": len(batch_times),
        "load_ms": load_ms,
        "inference_ms": inference_ms,
        "first_batch_ms": batch_times[0],
        "questions_per_second": questions / (inference_ms / 1000.0) if inference_ms > 0 else 0.0,
    }


def parse_option(text: str) -> Option:
    """``name`` or ``name=description``. Only the first ``=`` separates the two."""
    name, separator, description = text.partition("=")
    if not name.strip():
        raise ValueError(f"an option needs a name: {text!r}")
    return Option(name.strip(), description.strip() if separator else None)


def _single_question(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Question | None:
    written = any(
        (
            args.state,
            args.state_file,
            args.context,
            args.context_file,
            args.instruction,
            args.option,
            args.level,
        )
    )
    if args.input is not None:
        if written:
            parser.error("--input does not combine with the flags that write one question")
        return None
    if not written:
        parser.error("give --input, or write one question with --state and --instruction")
    if args.instruction is None:
        parser.error("one question needs --instruction")
    if args.context is not None and args.context_file is not None:
        parser.error("--context and --context-file do not combine")
    if (args.state is None) == (args.state_file is None):
        parser.error("one question needs exactly one of --state and --state-file")
    kind = DecisionType(args.kind)
    if kind is not DecisionType.CHOICE and args.option:
        parser.error("--option belongs to a choice question")
    if kind is not DecisionType.SCORE and args.level:
        parser.error("--level belongs to a score question")
    try:
        if kind is DecisionType.CHOICE:
            question = DecisionQuestion.choice(
                args.instruction, [parse_option(text) for text in args.option]
            )
        elif kind is DecisionType.SCORE:
            question = DecisionQuestion.score(args.instruction, args.level)
        else:
            question = DecisionQuestion.noul(args.instruction, args.false_option, args.true_option)
        state = args.state if args.state is not None else args.state_file.read_text("utf-8")
        context = args.context
        if args.context_file is not None:
            context = args.context_file.read_text("utf-8")
        return None, state_text(state), question, context_text(context)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def _read_records(source: str) -> Iterator[list[Question]]:
    if source == "-":
        yield from _parse_lines(sys.stdin)
        return
    with Path(source).open(encoding="utf-8") as stream:
        yield from _parse_lines(stream)


def _parse_lines(stream: TextIO) -> Iterator[list[Question]]:
    """One list of questions per record: one for ``question``, several for ``questions``."""
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("a record must be a JSON object")
            identifier = record.get("id")
            identifier = None if identifier is None else str(identifier)
            state = state_text(record["state"])
            context = context_text(record.get("context"))
            if "questions" not in record:
                yield [(identifier, state, question_from_record(record["question"]), context)]
                continue
            if "question" in record:
                raise ValueError("a record has question or questions, not both")
            yield _record_questions(identifier, state, context, record["questions"])
        except KeyError as error:
            raise ValueError(f"line {line_number}: missing field {error.args[0]!r}") from error
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"line {line_number}: {error}") from error


def _record_questions(
    identifier: str | None, state: str, context: str | None, raw: Any
) -> list[Question]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("questions must be a non-empty list")
    questions: list[Question] = []
    seen: set[str] = set()
    for position, item in enumerate(raw):
        try:
            own = item.get("id") if isinstance(item, dict) else None
            name = str(position) if own is None else str(own)
            if name in seen:
                raise ValueError(f"the id {name!r} appears twice")
            seen.add(name)
            if identifier is not None:
                name = f"{identifier}/{name}"
            elif own is None:
                name = None
            questions.append((name, state, question_from_record(item), context))
        except KeyError as error:
            raise ValueError(f"question {position}: missing field {error.args[0]!r}") from error
        except (TypeError, ValueError) as error:
            raise ValueError(f"question {position}: {error}") from error
    return questions


def _chunks(records: Sequence[list[Question]], size: int) -> Iterator[list[Question]]:
    """Batches of at most ``size`` questions that do not split a record unless it is larger."""
    chunk: list[Question] = []
    for record in records:
        if chunk and len(chunk) + len(record) > size:
            yield chunk
            chunk = []
        for question in record:
            chunk.append(question)
            if len(chunk) == size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _decide(runtime: DecisionRuntime, chunk: Sequence[Question], first_line: int) -> list[Decision]:
    pairs = [(state, question, context) for _identifier, state, question, context in chunk]
    try:
        return runtime.decide_many(pairs)
    except ValueError:
        # A question that does not fit the token budget fails the whole batch. Find
        # it, so the error names a question instead of a batch.
        for offset, pair in enumerate(pairs):
            try:
                runtime.decide_many([pair])
            except ValueError as error:
                raise ValueError(f"question {first_line + offset + 1}: {error}") from error
        raise


def _render(identifier: str | None, decision: Decision, *, top_only: bool) -> dict[str, Any]:
    row: dict[str, Any] = {} if identifier is None else {"id": identifier}
    if top_only:
        row.update(label=decision.label, confidence=decision.confidence)
        return row
    row.update(
        kind=decision.kind,
        label=decision.label,
        confidence=decision.confidence,
        entropy_confidence=decision.entropy_confidence,
        should_act=decision.should_act,
        expected_level=decision.expected_level,
        probabilities=dict(zip(decision.option_names, decision.probabilities, strict=True)),
        logits=dict(zip(decision.option_names, decision.logits, strict=True)),
    )
    return row


@contextmanager
def _destination(path: Path | None) -> Iterator[TextIO]:
    """stdout, or a file. It is opened only after the input parsed and the model loaded."""
    if path is None:
        yield sys.stdout
        return
    with path.open("w", encoding="utf-8") as stream:
        yield stream


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
