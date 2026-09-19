"""Metrics for recorded predictions: accuracy, calibration, coverage, gates.

Definitions used throughout:

- ``decision`` — what code would act on: ``choice`` field, ``noul >= 0.5``,
  or the rounded ``score``.
- ``decision_prob`` — the probability the model assigned to the decision it
  made (for Noul, ``max(p, 1 - p)``). This is what coverage/ECE are computed on.
- ECE — expected calibration error over equal-mass bins of ``decision_prob``
  against decision correctness. Brier is reported for Noul questions only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .packs import Pack

INPUT_USD_PER_MTOK = 0.042  # TypeSafe early-access price; output tokens are free.
COVERAGE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SMALL_N_ITEMS = 50  # below this, report that CI/ECE are coarse


@dataclass(frozen=True)
class Item:
    case_id: str
    qid: str
    qtype: str
    expected: Any
    got: Any
    correct: bool
    decision_prob: float


@dataclass(frozen=True)
class QuestionMetrics:
    qid: str
    qtype: str
    n: int
    missing: int
    accuracy: float
    mean_decision_prob: float
    ece: float
    brier: float | None


@dataclass(frozen=True)
class CoverageRow:
    threshold: float
    coverage: float
    precision: float | None


@dataclass(frozen=True)
class ThresholdCoverage:
    """Coverage when auto-accepting with the pack's own per-label thresholds."""

    n_total: int
    n_auto: int
    coverage: float
    precision: float | None


@dataclass(frozen=True)
class ThresholdSuggestion:
    """Highest-coverage threshold that still reaches a target precision."""

    threshold: float
    coverage: float
    precision: float
    n_accepted: int


@dataclass(frozen=True)
class OverallMetrics:
    n_cases: int
    n_items: int
    case_errors: int
    missing_items: int
    accuracy: float
    ece: float
    mean_decision_prob: float
    total_cost_usd: float | None
    cost_per_case_usd: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    per_question: dict[str, QuestionMetrics] = field(default_factory=dict)
    coverage: tuple[CoverageRow, ...] = ()
    threshold_coverage: ThresholdCoverage | None = None
    accuracy_ci: tuple[float, float] | None = None
    ece_ci: tuple[float, float] | None = None


@dataclass(frozen=True)
class GateResult:
    gate: str
    ok: bool | None  # None = not computable from this recording
    detail: str


def build_items(pack: Pack, predictions: dict[str, dict[str, Any]]) -> tuple[list[Item], int, int]:
    """Turn predictions into comparable items.

    Returns (items, case_errors, missing_items). Cases with an error record or
    without answers count once as a case error; their expected questions count
    as missing items.
    """
    items: list[Item] = []
    case_errors = 0
    missing = 0

    for case in pack.cases:
        record = predictions.get(case.id)
        if record is None or record.get("error") or record.get("answers") is None:
            case_errors += 1
            missing += len(case.expect)
            continue
        answers = record["answers"]
        for qid, expected in case.expect.items():
            answer = answers.get(qid)
            if not isinstance(answer, dict):
                missing += 1
                continue
            item = _item_for(case.id, qid, pack.questions[qid], expected, answer)
            if item is not None:
                items.append(item)
            else:
                missing += 1
    return items, case_errors, missing


def _item_for(
    case_id: str, qid: str, question: dict[str, Any], expected: Any, answer: dict[str, Any]
) -> Item | None:
    qtype = question["type"]
    if qtype == "noul":
        if "noul" not in answer:
            return None
        probability = float(answer["noul"])
        decision = probability >= 0.5
        return Item(
            case_id=case_id,
            qid=qid,
            qtype=qtype,
            expected=expected,
            got=decision,
            correct=decision == expected,
            decision_prob=max(probability, 1.0 - probability),
        )

    if qtype == "choice":
        probabilities = _float_map(answer.get("probabilities"))
        if not probabilities:
            return None
        decision = answer.get("choice") or max(probabilities, key=probabilities.get)
        return Item(
            case_id=case_id,
            qid=qid,
            qtype=qtype,
            expected=expected,
            got=decision,
            correct=decision == expected,
            decision_prob=probabilities.get(decision, max(probabilities.values())),
        )

    if qtype == "score":
        probabilities = _float_map(answer.get("probabilities"))
        if not probabilities:
            return None
        levels = list(question["levels"])
        best_key = max(probabilities, key=probabilities.get)
        try:
            decision = levels[int(best_key)]
        except (ValueError, IndexError):
            return None
        return Item(
            case_id=case_id,
            qid=qid,
            qtype=qtype,
            expected=expected,
            got=decision,
            correct=decision == expected,
            decision_prob=probabilities[best_key],
        )

    return None  # pragma: no cover - loader rejects unknown types


def _float_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def compute(
    pack: Pack,
    predictions: dict[str, dict[str, Any]],
    bootstrap: int = 1000,
    seed: int = 0,
) -> OverallMetrics:
    items, case_errors, missing_items = build_items(pack, predictions)

    per_question: dict[str, QuestionMetrics] = {}
    for qid in pack.questions:
        subset = [item for item in items if item.qid == qid]
        expected_count = sum(1 for case in pack.cases if qid in case.expect)
        q_missing = expected_count - len(subset)
        if not subset:
            per_question[qid] = QuestionMetrics(
                qid=qid,
                qtype=pack.questions[qid]["type"],
                n=0,
                missing=q_missing,
                accuracy=0.0,
                mean_decision_prob=0.0,
                ece=0.0,
                brier=None,
            )
            continue
        corrects = [item.correct for item in subset]
        probs = [item.decision_prob for item in subset]
        brier = None
        if pack.questions[qid]["type"] == "noul":
            brier = sum(
                (_noul_probability(item) - float(item.expected)) ** 2 for item in subset
            ) / len(subset)
        per_question[qid] = QuestionMetrics(
            qid=qid,
            qtype=pack.questions[qid]["type"],
            n=len(subset),
            missing=q_missing,
            accuracy=_mean(corrects),
            mean_decision_prob=_mean(probs),
            ece=expected_calibration_error(probs, corrects),
            brier=brier,
        )

    all_probs = [item.decision_prob for item in items]
    all_correct = [item.correct for item in items]

    total_input_tokens = 0
    saw_usage = False
    latencies: list[float] = []
    for case in pack.cases:
        record = predictions.get(case.id) or {}
        usage = record.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int | float):
            total_input_tokens += int(usage["input_tokens"])
            saw_usage = True
        if isinstance(record.get("latency_ms"), int | float):
            latencies.append(float(record["latency_ms"]))

    total_cost = (total_input_tokens * INPUT_USD_PER_MTOK / 1_000_000) if saw_usage else None
    cost_per_case = (total_cost / len(pack.cases)) if total_cost is not None else None

    return OverallMetrics(
        n_cases=len(pack.cases),
        n_items=len(items),
        case_errors=case_errors,
        missing_items=missing_items,
        accuracy=_mean(all_correct),
        ece=expected_calibration_error(all_probs, all_correct),
        mean_decision_prob=_mean(all_probs),
        total_cost_usd=total_cost,
        cost_per_case_usd=cost_per_case,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        per_question=per_question,
        coverage=coverage_rows(items),
        threshold_coverage=_threshold_coverage(pack, items) if pack.thresholds else None,
        accuracy_ci=bootstrap_interval(all_correct, _mean, bootstrap, seed),
        ece_ci=bootstrap_interval(
            list(zip(all_probs, all_correct, strict=True)),
            lambda sample: expected_calibration_error(
                [pair[0] for pair in sample], [pair[1] for pair in sample]
            ),
            bootstrap,
            seed,
        ),
    )


def _threshold_coverage(pack: Pack, items: list[Item]) -> ThresholdCoverage:
    """Auto-accept coverage under the pack's `thresholds` (SPEC semantics).

    An item is auto-accepted when the label the model answered has a floor in
    the pack thresholds and the decision probability clears it. Labels without
    a floor (typically `unknown`) always route to review.
    """
    accepted: list[Item] = []
    for item in items:
        if item.qtype == "noul":
            label = "true" if item.got else "false"
        else:
            label = str(item.got)
        floor = pack.thresholds.get(item.qid, {}).get(label)
        if floor is not None and item.decision_prob >= floor:
            accepted.append(item)
    return ThresholdCoverage(
        n_total=len(items),
        n_auto=len(accepted),
        coverage=len(accepted) / len(items) if items else 0.0,
        precision=_mean([item.correct for item in accepted]) if accepted else None,
    )


def _noul_probability(item: Item) -> float:
    return item.decision_prob if item.got else 1.0 - item.decision_prob


def _mean(values: list[Any]) -> float:
    return sum(float(v) for v in values) / len(values) if values else 0.0


def expected_calibration_error(
    probabilities: list[float], corrects: list[bool], bins: int = 10
) -> float:
    """ECE over equal-mass bins of decision probability vs correctness."""
    if not probabilities:
        return 0.0
    pairs = sorted(zip(probabilities, corrects, strict=True))
    n = len(pairs)
    k = min(bins, n)
    total = 0.0
    for index in range(k):
        chunk = pairs[index * n // k : (index + 1) * n // k]
        if not chunk:
            continue
        mean_p = sum(p for p, _ in chunk) / len(chunk)
        mean_c = sum(1.0 for _, correct in chunk if correct) / len(chunk)
        total += len(chunk) / n * abs(mean_p - mean_c)
    return total


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None when there are no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def bootstrap_interval(
    values: list[Any],
    statistic: Callable[[list[Any]], float],
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for a statistic; None when disabled or no data."""
    if not values or n_resamples <= 0:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(
        statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)
    )
    low_index = max(0, int((1 - confidence) / 2 * n_resamples))
    high_index = min(n_resamples - 1, int((1 + confidence) / 2 * n_resamples))
    return (stats[low_index], stats[high_index])


def suggest_threshold(
    items: list[Item], target_precision: float, min_accepted: int = 10
) -> ThresholdSuggestion | None:
    """Highest-coverage probability cut that still reaches the target precision.

    ``min_accepted`` keeps a handful of lucky items from suggesting a cut that
    no real workload could use.
    """
    if not items:
        return None
    best: ThresholdSuggestion | None = None
    for step in range(50, 100):
        threshold = step / 100
        accepted = [item for item in items if item.decision_prob >= threshold]
        if len(accepted) < min_accepted:
            continue
        precision = _mean([item.correct for item in accepted])
        if precision < target_precision:
            continue
        coverage = len(accepted) / len(items)
        if best is None or coverage > best.coverage:
            best = ThresholdSuggestion(
                threshold=threshold,
                coverage=coverage,
                precision=precision,
                n_accepted=len(accepted),
            )
    return best


def coverage_rows(items: list[Item]) -> tuple[CoverageRow, ...]:
    rows = []
    for threshold in COVERAGE_THRESHOLDS:
        accepted = [item for item in items if item.decision_prob >= threshold]
        precision = _mean([item.correct for item in accepted]) if accepted else None
        rows.append(
            CoverageRow(
                threshold=threshold,
                coverage=len(accepted) / len(items) if items else 0.0,
                precision=precision,
            )
        )
    return tuple(rows)


def evaluate_gates(pack: Pack, overall: OverallMetrics) -> list[GateResult]:
    results: list[GateResult] = []
    gates = pack.gates

    if "min_accuracy" in gates:
        value = float(gates["min_accuracy"])
        results.append(
            GateResult(
                "min_accuracy",
                overall.accuracy >= value,
                f"accuracy {overall.accuracy:.3f} >= {value:.3f}",
            )
        )
    if "max_ece" in gates:
        value = float(gates["max_ece"])
        results.append(
            GateResult("max_ece", overall.ece <= value, f"ece {overall.ece:.3f} <= {value:.3f}")
        )
    if "max_cost_per_case_usd" in gates:
        value = float(gates["max_cost_per_case_usd"])
        if overall.cost_per_case_usd is None:
            results.append(GateResult("max_cost_per_case_usd", None, "no usage data in recording"))
        else:
            results.append(
                GateResult(
                    "max_cost_per_case_usd",
                    overall.cost_per_case_usd <= value,
                    f"cost/case ${overall.cost_per_case_usd:.6f} <= ${value:.6f}",
                )
            )
    if "max_p95_latency_ms" in gates:
        value = float(gates["max_p95_latency_ms"])
        if overall.p95_latency_ms is None:
            results.append(GateResult("max_p95_latency_ms", None, "no latency data in recording"))
        else:
            results.append(
                GateResult(
                    "max_p95_latency_ms",
                    overall.p95_latency_ms <= value,
                    f"p95 {overall.p95_latency_ms:.0f}ms <= {value:.0f}ms",
                )
            )
    if "min_coverage_at_precision" in gates:
        spec = gates["min_coverage_at_precision"]
        required_precision = float(spec["precision"])
        min_coverage = float(spec["min_coverage"])
        best = None
        for row in overall.coverage:
            if row.precision is not None and row.precision >= required_precision:
                best = row
        if best is None:
            results.append(
                GateResult(
                    "min_coverage_at_precision",
                    False,
                    f"no threshold on the grid reaches precision >= {required_precision:.2f}",
                )
            )
        else:
            results.append(
                GateResult(
                    "min_coverage_at_precision",
                    best.coverage >= min_coverage,
                    f"coverage {best.coverage:.2f} at precision >= {required_precision:.2f} "
                    f"(threshold {best.threshold:.2f}) >= {min_coverage:.2f}",
                )
            )
    if "min_accuracy_ci_lower" in gates:
        value = float(gates["min_accuracy_ci_lower"])
        if overall.accuracy_ci is None:
            results.append(
                GateResult("min_accuracy_ci_lower", None, "bootstrap disabled (--bootstrap 0)")
            )
        else:
            low = overall.accuracy_ci[0]
            results.append(
                GateResult(
                    "min_accuracy_ci_lower",
                    low >= value,
                    f"accuracy CI lower {low:.3f} >= {value:.3f}",
                )
            )
    for qid, per_question_gates in (gates.get("per_question") or {}).items():
        metrics = overall.per_question.get(qid)
        where = f"per_question.{qid}"
        if metrics is None or metrics.n == 0:
            for gate_name in per_question_gates:
                results.append(GateResult(f"{where}.{gate_name}", None, "no items in recording"))
            continue
        if "min_accuracy" in per_question_gates:
            value = float(per_question_gates["min_accuracy"])
            results.append(
                GateResult(
                    f"{where}.min_accuracy",
                    metrics.accuracy >= value,
                    f"accuracy {metrics.accuracy:.3f} >= {value:.3f}",
                )
            )
        if "max_ece" in per_question_gates:
            value = float(per_question_gates["max_ece"])
            results.append(
                GateResult(
                    f"{where}.max_ece",
                    metrics.ece <= value,
                    f"ece {metrics.ece:.3f} <= {value:.3f}",
                )
            )
    return results
