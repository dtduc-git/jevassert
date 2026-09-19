"""Render check results as markdown (for pack evidence) and JUnit XML (for CI)."""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from .metrics import GateResult, OverallMetrics, QuestionMetrics, ThresholdSuggestion
from .packs import Pack


def render_markdown(
    pack: Pack,
    overall: OverallMetrics,
    gates: list[GateResult],
    suggestion: ThresholdSuggestion | None = None,
    target_precision: float | None = None,
    recorded_models: list[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# jevassert report — {pack.id} v{pack.version}")
    lines.append("")
    models = list(recorded_models or ([pack.tested] if pack.tested else []))
    model_label = ", ".join(f"`{m}`" for m in models) if models else f"`{pack.record_model}`"
    if models and pack.tested and models == [pack.tested]:
        suffix = f" (recorded against `{pack.tested}`)"
    elif pack.tested:
        suffix = f" (pack pinned to `{pack.tested}`)"
    else:
        suffix = " (provisional — no pinned version)"
    lines.append(f"- model: {model_label}{suffix}")
    lines.append(
        f"- cases: {overall.n_cases} ({overall.case_errors} errors, "
        f"{overall.missing_items} missing answers)"
    )
    lines.append(
        f"- items: {overall.n_items} — accuracy **{overall.accuracy:.3f}**, "
        f"ECE **{overall.ece:.3f}**"
    )
    if overall.accuracy_ci is not None:
        low, high = overall.accuracy_ci
        ece_part = ""
        if overall.ece_ci is not None:
            ece_part = f", ECE CI {overall.ece_ci[0]:.3f}–{overall.ece_ci[1]:.3f}"
        lines.append(f"- bootstrap 95%: accuracy CI {low:.3f}–{high:.3f}{ece_part}")
    if overall.cost_per_case_usd is not None:
        lines.append(
            f"- cost: ${overall.cost_per_case_usd:.6f}/case (${overall.total_cost_usd:.4f} total)"
        )
    if overall.p95_latency_ms is not None:
        lines.append(
            f"- latency: p50 {overall.p50_latency_ms:.0f}ms, p95 {overall.p95_latency_ms:.0f}ms"
        )
    lines.append("")
    lines.append("## Per question")
    lines.append("")
    lines.append("| question | type | n | missing | accuracy | mean p(decision) | ECE | Brier |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for metrics in overall.per_question.values():
        lines.append(_question_row(metrics))
    lines.append("")
    lines.append("## Coverage at decision probability")
    lines.append("")
    lines.append("| accept if p >= | coverage | precision |")
    lines.append("|---|---|---|")
    for row in overall.coverage:
        precision = f"{row.precision:.3f}" if row.precision is not None else "—"
        lines.append(f"| {row.threshold:.2f} | {row.coverage:.3f} | {precision} |")
    if overall.threshold_coverage is not None:
        tc = overall.threshold_coverage
        lines.append("")
        lines.append("## Author thresholds (pack threshold floors)")
        lines.append("")
        precision = f"{tc.precision:.3f}" if tc.precision is not None else "—"
        lines.append(
            f"- auto-accepted {tc.n_auto}/{tc.n_total} ({tc.coverage:.3f}), precision {precision}"
        )
        lines.append("- labels without a floor (usually `unknown`) always route to review")
    if suggestion is not None and target_precision is not None:
        lines.append("")
        lines.append(f"## Suggested threshold (target precision {target_precision:.2f})")
        lines.append("")
        lines.append(
            f"- accept when p >= {suggestion.threshold:.2f}: coverage {suggestion.coverage:.3f}, "
            f"precision {suggestion.precision:.3f} (n={suggestion.n_accepted})"
        )
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    if not gates:
        lines.append("No gates declared in gates.yaml.")
    for gate in gates:
        mark = "PASS" if gate.ok else ("SKIP" if gate.ok is None else "FAIL")
        lines.append(f"- **{mark}** `{gate.gate}` — {gate.detail}")
    lines.append("")
    return "\n".join(lines)


def _question_row(metrics: QuestionMetrics) -> str:
    brier = f"{metrics.brier:.3f}" if metrics.brier is not None else "—"
    return (
        f"| {metrics.qid} | {metrics.qtype} | {metrics.n} | {metrics.missing} "
        f"| {metrics.accuracy:.3f} | {metrics.mean_decision_prob:.3f} "
        f"| {metrics.ece:.3f} | {brier} |"
    )


def render_junit(pack: Pack, gates: list[GateResult]) -> str:
    classname = quoteattr(f"jevassert.{pack.id}")
    cases = []
    failures = 0
    for gate in gates:
        if gate.ok is False:
            failures += 1
            cases.append(
                f"  <testcase name={quoteattr(gate.gate)} classname={classname}>"
                f"<failure message={quoteattr(gate.detail)}/></testcase>"
            )
        else:
            skipped = '<skipped message="not computable"/>' if gate.ok is None else ""
            cases.append(
                f"  <testcase name={quoteattr(gate.gate)} classname={classname}>"
                f"{skipped}</testcase>"
            )
    body = "\n".join(cases)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name={quoteattr(pack.id)} tests="{len(gates)}" failures="{failures}">\n'
        f"{body}\n</testsuite>\n"
    )


def escape_comment(text: str) -> str:
    """Escape text for embedding; exposed for tests."""
    return escape(text)
