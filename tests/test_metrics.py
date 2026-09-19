import dataclasses
from pathlib import Path

import pytest
from helpers import choice, noul, record, score, write_pack

from jevassert.metrics import (
    build_items,
    compute,
    evaluate_gates,
    expected_calibration_error,
    suggest_threshold,
)
from jevassert.packs import load_pack

QUESTIONS = {
    "tone": {
        "type": "choice",
        "instructions": "Tone?",
        "options": {"good": "x", "bad": "y", "unknown": "cannot tell"},
    },
    "urgent": {"type": "noul", "instructions": "Urgent?"},
    "severity": {
        "type": "score",
        "instructions": "Severity?",
        "levels": ["low", "medium", "high", "unknown"],
    },
}
CASES = [
    {
        "id": "c-1",
        "state": {"text": "a"},
        "expect": {"tone": "good", "urgent": True, "severity": "high"},
    },
    {
        "id": "c-2",
        "state": {"text": "b"},
        "expect": {"tone": "bad", "urgent": False, "severity": "medium"},
    },
]
THRESHOLDS = {
    "tone": {"good": 0.6, "bad": 0.6},
    "urgent": {"true": 0.9, "false": 0.9},
    "severity": {"low": 0.7, "medium": 0.7, "high": 0.7},
}
GATES = {
    "min_accuracy": 0.8,
    "max_ece": 0.1,
    "min_coverage_at_precision": {"precision": 0.9, "min_coverage": 0.1},
}
USAGE = {"input_tokens": 1000, "output_tokens": 5}


def build_pack(tmp_path: Path, gates: dict | None = GATES, thresholds: dict | None = THRESHOLDS):
    return load_pack(
        write_pack(tmp_path / "test-pack", QUESTIONS, CASES, thresholds=thresholds, gates=gates)
    )


def predictions() -> dict:
    return {
        "c-1": record(
            "c-1",
            {
                "tone": choice("good", {"good": 0.6, "bad": 0.3, "unknown": 0.1}),
                "urgent": noul(0.9),
                "severity": score(2.4, {"0": 0.1, "1": 0.3, "2": 0.6, "3": 0.0}),
            },
            usage=USAGE,
        ),
        "c-2": record(
            "c-2",
            {
                "tone": choice("good", {"good": 0.5, "bad": 0.4, "unknown": 0.1}),
                "urgent": noul(0.4),
                "severity": score(1.2, {"0": 0.2, "1": 0.6, "2": 0.2, "3": 0.0}),
            },
            usage=USAGE,
        ),
    }


def test_overall_and_per_question_accuracy(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path), predictions())
    assert metrics.n_items == 6
    assert metrics.accuracy == pytest.approx(5 / 6)
    assert metrics.per_question["tone"].accuracy == pytest.approx(0.5)
    assert metrics.per_question["urgent"].accuracy == pytest.approx(1.0)
    assert metrics.per_question["severity"].accuracy == pytest.approx(1.0)


def test_score_decision_is_argmax_level(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path), predictions())
    assert metrics.per_question["severity"].n == 2
    assert metrics.per_question["severity"].mean_decision_prob == pytest.approx(0.6)


def test_noul_brier_and_calibration(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path), predictions())
    # c-1: (0.9 - 1)^2 = 0.01; c-2: (0.4 - 0)^2 = 0.16
    assert metrics.per_question["urgent"].brier == pytest.approx(0.085)


def test_cost_and_latency(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path), predictions())
    assert metrics.total_cost_usd == pytest.approx(2000 * 0.042 / 1_000_000)
    assert metrics.cost_per_case_usd == pytest.approx(1000 * 0.042 / 1_000_000)
    assert metrics.p50_latency_ms == pytest.approx(100.0)


def test_coverage_rows(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path), predictions())
    rows = {row.threshold: row for row in metrics.coverage}
    assert rows[0.5].coverage == pytest.approx(1.0)
    assert rows[0.5].precision == pytest.approx(5 / 6)
    assert rows[0.9].coverage == pytest.approx(1 / 6)
    assert rows[0.9].precision == pytest.approx(1.0)
    assert rows[0.95].coverage == pytest.approx(0.0)
    assert rows[0.95].precision is None


def test_threshold_coverage_uses_pack_floors(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path), predictions())
    tc = metrics.threshold_coverage
    assert tc is not None
    # auto: c-1 tone (p .6 >= .6) and c-1 urgent (p .9 >= .9) — everything else
    # either misses its floor or answers a label without one
    assert tc.n_total == 6
    assert tc.n_auto == 2
    assert tc.coverage == pytest.approx(2 / 6)
    assert tc.precision == pytest.approx(1.0)


def test_threshold_coverage_is_none_without_thresholds(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path, thresholds=None), predictions())
    assert metrics.threshold_coverage is None


def test_gates_pass_and_fail(tmp_path: Path) -> None:
    pack = build_pack(tmp_path)
    gates = {gate.gate: gate for gate in evaluate_gates(pack, compute(pack, predictions()))}
    assert gates["min_accuracy"].ok is True
    assert gates["max_ece"].ok is False  # equal-mass bins on six items land far off
    assert gates["min_coverage_at_precision"].ok is True


def test_error_and_missing_items_are_counted(tmp_path: Path) -> None:
    pack = build_pack(tmp_path)
    partial = {"c-1": predictions()["c-1"]}
    metrics = compute(pack, partial)
    assert metrics.case_errors == 1
    assert metrics.missing_items == 3
    assert metrics.n_items == 3
    assert metrics.per_question["tone"].missing == 1


def test_empty_input_reports_zero() -> None:
    assert expected_calibration_error([], []) == 0.0


def test_ece_equal_mass_hand_computed() -> None:
    probabilities = [0.9, 0.8, 0.1, 0.2]
    corrects = [True, True, False, False]
    # sorted pairs -> bins of two: (0.1,F)(0.2,F) gap 0.15; (0.8,T)(0.9,T) gap 0.15
    assert expected_calibration_error(probabilities, corrects, bins=2) == pytest.approx(0.15)


def test_gates_skip_when_usage_missing(tmp_path: Path) -> None:
    pack = build_pack(tmp_path, gates={"max_cost_per_case_usd": 0.001})
    metrics = compute(pack, predictions())
    metrics_without_cost = dataclasses.replace(metrics, cost_per_case_usd=None)
    gates = evaluate_gates(pack, metrics_without_cost)
    assert gates[0].ok is None
    assert "no usage data" in gates[0].detail


def test_bootstrap_ci_bounds_and_disable(tmp_path: Path) -> None:
    metrics = compute(build_pack(tmp_path), predictions(), bootstrap=200, seed=7)
    assert metrics.accuracy_ci is not None
    low, high = metrics.accuracy_ci
    assert low <= metrics.accuracy <= high
    assert metrics.ece_ci is not None

    disabled = compute(build_pack(tmp_path), predictions(), bootstrap=0)
    assert disabled.accuracy_ci is None
    assert disabled.ece_ci is None


def test_min_accuracy_ci_lower_gate(tmp_path: Path) -> None:
    pack = build_pack(tmp_path, gates={"min_accuracy_ci_lower": 0.5})
    gates = {g.gate: g for g in evaluate_gates(pack, compute(pack, predictions()))}
    assert gates["min_accuracy_ci_lower"].ok is True


def test_suggest_threshold_picks_highest_coverage(tmp_path: Path) -> None:
    items, _, _ = build_items(build_pack(tmp_path), predictions())
    suggestion = suggest_threshold(items, target_precision=0.99, min_accepted=3)
    assert suggestion is not None
    # excluding the one wrong item (p=0.5) reaches precision 1.0 at p >= 0.51
    assert suggestion.threshold == pytest.approx(0.51)
    assert suggestion.coverage == pytest.approx(5 / 6)
    assert suggestion.precision == pytest.approx(1.0)
    assert suggestion.n_accepted == 5


def test_suggest_threshold_none_when_unreachable(tmp_path: Path) -> None:
    items, _, _ = build_items(build_pack(tmp_path), predictions())
    assert suggest_threshold(items, target_precision=1.01) is None


def test_per_question_gates(tmp_path: Path) -> None:
    gates = {
        "per_question": {
            "tone": {"min_accuracy": 0.6, "max_ece": 0.5},
            "urgent": {"min_accuracy": 0.99},
        }
    }
    pack = build_pack(tmp_path, gates=gates)
    results = {g.gate: g for g in evaluate_gates(pack, compute(pack, predictions()))}
    assert results["per_question.tone.min_accuracy"].ok is False  # tone accuracy is 0.5
    assert results["per_question.tone.max_ece"].ok is True
    assert results["per_question.urgent.min_accuracy"].ok is True
