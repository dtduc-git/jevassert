"""Paired comparison of two recordings (same pack): accuracy deltas + McNemar.

Typical use: compare a rubric/wording change or a model-version bump.
Both recordings must cover the same (case, question) items; only items present
and answerable on both sides are compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .metrics import INPUT_USD_PER_MTOK, Item, build_items
from .packs import Pack


@dataclass(frozen=True)
class QuestionDelta:
    qid: str
    n: int
    accuracy_a: float
    accuracy_b: float

    @property
    def delta(self) -> float:
        return self.accuracy_b - self.accuracy_a


@dataclass(frozen=True)
class CompareResult:
    n_items: int
    accuracy_a: float
    accuracy_b: float
    wins: int  # A correct, B wrong
    losses: int  # A wrong, B correct
    p_value: float
    cost_per_case_a_usd: float | None
    cost_per_case_b_usd: float | None
    per_question: dict[str, QuestionDelta] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.accuracy_b - self.accuracy_a


def compare(
    pack: Pack,
    predictions_a: dict[str, dict[str, Any]],
    predictions_b: dict[str, dict[str, Any]],
) -> CompareResult:
    items_a = _keyed(build_items(pack, predictions_a)[0])
    items_b = _keyed(build_items(pack, predictions_b)[0])
    shared = sorted(set(items_a) & set(items_b))
    if not shared:
        raise ValueError("no shared (case, question) items between the two recordings")

    paired = [(items_a[key], items_b[key]) for key in shared]
    wins = sum(1 for a, b in paired if a.correct and not b.correct)
    losses = sum(1 for a, b in paired if not a.correct and b.correct)

    per_question: dict[str, QuestionDelta] = {}
    for qid in pack.questions:
        q_pairs = [(a, b) for a, b in paired if a.qid == qid]
        if not q_pairs:
            continue
        per_question[qid] = QuestionDelta(
            qid=qid,
            n=len(q_pairs),
            accuracy_a=sum(1.0 for a, _ in q_pairs if a.correct) / len(q_pairs),
            accuracy_b=sum(1.0 for _, b in q_pairs if b.correct) / len(q_pairs),
        )

    return CompareResult(
        n_items=len(paired),
        accuracy_a=sum(1.0 for a, _ in paired if a.correct) / len(paired),
        accuracy_b=sum(1.0 for _, b in paired if b.correct) / len(paired),
        wins=wins,
        losses=losses,
        p_value=mcnemar_exact(wins, losses),
        cost_per_case_a_usd=_cost_per_case(pack, predictions_a),
        cost_per_case_b_usd=_cost_per_case(pack, predictions_b),
        per_question=per_question,
    )


def _keyed(items: list[Item]) -> dict[tuple[str, str], Item]:
    return {(item.case_id, item.qid): item for item in items}


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar test (binomial, p=0.5) — no scipy needed."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def _cost_per_case(pack: Pack, predictions: dict[str, dict[str, Any]]) -> float | None:
    tokens = 0
    saw_usage = False
    for case in pack.cases:
        usage = (predictions.get(case.id) or {}).get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int | float):
            tokens += int(usage["input_tokens"])
            saw_usage = True
    return (tokens * INPUT_USD_PER_MTOK / 1_000_000 / len(pack.cases)) if saw_usage else None
