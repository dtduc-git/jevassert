"""CLI: ``jevassert record | check | compare``."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .backends import AdapterClient
from .client import JevClient, JevError
from .compare import CompareResult
from .compare import compare as compare_recordings
from .metrics import (
    INPUT_USD_PER_MTOK,
    SMALL_N_ITEMS,
    OverallMetrics,
    ThresholdSuggestion,
    build_items,
    compute,
    evaluate_gates,
    suggest_threshold,
)
from .packs import Pack, PackError, load_pack
from .report import render_junit, render_markdown
from .runner import load_predictions, record, write_predictions


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (PackError, FileNotFoundError, ValueError, JevError) as exc:
        print(f"jevassert: error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevassert",
        description="Record/replay regression tests for Jev question packs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    record_parser = sub.add_parser(
        "record", help="call Jev for every case and write a predictions JSONL"
    )
    record_parser.add_argument("pack", help="pack directory (containing pack.yaml + cases.jsonl)")
    record_parser.add_argument("-o", "--out", default="predictions.jsonl")
    record_parser.add_argument("--concurrency", type=int, default=8)
    record_parser.add_argument(
        "--limit", type=int, default=None, help="record only the first N cases"
    )
    record_parser.add_argument(
        "--model",
        default=None,
        help="model version to record (default: pack.tested or jev-latest)",
    )
    record_parser.add_argument(
        "--backend",
        choices=("typesafe", "openai", "anthropic", "bedrock"),
        default="typesafe",
        help="typesafe = TypeSafe API; openai = any OpenAI-compatible endpoint "
        "(--base-url, e.g. Ollama); anthropic = Claude API; bedrock = Claude via "
        "AWS Bedrock (AWS_PROFILE/AWS_REGION, use an inference profile id). The "
        "last three go through system-one-adapter (pip install 'jevassert[adapter]') "
        "and require --model",
    )
    record_parser.add_argument(
        "--base-url",
        default=None,
        help="override TYPESAFE_BASE_URL (or the OpenAI-compatible endpoint for --backend openai)",
    )
    record_parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cases already recorded successfully in --out (retry errors only)",
    )
    record_parser.add_argument(
        "--rpm", type=int, default=0, help="pace requests per minute (0 = unlimited)"
    )
    record_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="estimate tokens and cost from the pack without sending anything",
    )
    record_parser.add_argument(
        "--shuffle-options",
        type=int,
        default=None,
        metavar="SEED",
        help="robustness pass: permute Choice option order with this seed",
    )
    record_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="record N independent rounds (extra rounds go to FILE-r2.jsonl, ...); "
        "prints the discordant-decision count across rounds",
    )
    record_parser.set_defaults(func=_cmd_record)

    check_parser = sub.add_parser("check", help="evaluate a recording against the pack gates")
    check_parser.add_argument("pack")
    check_parser.add_argument("-p", "--predictions", default="predictions.jsonl")
    check_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    check_parser.add_argument("--junit", default=None, help="write JUnit XML to this path")
    check_parser.add_argument("--report", default=None, help="write a markdown report to this path")
    check_parser.add_argument(
        "--failures", action="store_true", help="list every mismatch after the summary"
    )
    check_parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="bootstrap resamples for confidence intervals (0 disables)",
    )
    check_parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="suggest the highest-coverage threshold that reaches this precision",
    )
    check_parser.add_argument(
        "--input-price",
        type=float,
        default=INPUT_USD_PER_MTOK,
        help=f"USD per million input tokens (default {INPUT_USD_PER_MTOK}, the Jev list price)",
    )
    check_parser.add_argument(
        "--output-price",
        type=float,
        default=0.0,
        help="USD per million output tokens (default 0: Jev output tokens are free)",
    )
    check_parser.add_argument(
        "--no-gates",
        action="store_true",
        help="skip gates.yaml entirely: metrics and report only, always exit 0 "
        "(for benchmarks and replays that must not enforce pack gates)",
    )
    check_parser.add_argument(
        "--partition",
        choices=("all", "dev", "test"),
        default="all",
        help="evaluate only the dev or test half (deterministic hash split) — "
        "tune thresholds on dev, verify on test",
    )
    check_parser.add_argument(
        "--partition-seed", type=int, default=0, help="seed for the dev/test split"
    )
    check_parser.add_argument(
        "--partition-ratio",
        type=float,
        default=0.5,
        help="dev share of the deterministic split (default 0.5)",
    )
    check_parser.set_defaults(func=_cmd_check)

    compare_parser = sub.add_parser("compare", help="paired comparison of two recordings")
    compare_parser.add_argument("pack")
    compare_parser.add_argument("--a", required=True, help="baseline predictions JSONL")
    compare_parser.add_argument("--b", required=True, help="candidate predictions JSONL")
    compare_parser.set_defaults(func=_cmd_compare)

    return parser


def _cmd_record(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    if args.dry_run:
        cases, tokens = _estimate_tokens(pack, args.limit)
        cost = tokens * INPUT_USD_PER_MTOK / 1_000_000
        print(
            f"dry-run: {cases} cases, ~{tokens} input tokens, ~${cost:.4f} "
            "at $0.042/M input tokens (rough estimate, nothing sent)"
        )
        return 0
    if args.repeat < 1:
        print("jevassert: error: --repeat must be >= 1", file=sys.stderr)
        return 2

    if args.backend == "typesafe":
        client: JevClient | AdapterClient = JevClient(base_url=args.base_url)
    else:
        if not args.model:
            print(f"jevassert: error: --backend {args.backend} requires --model", file=sys.stderr)
            return 2
        client = AdapterClient(args.backend, args.model, base_url=args.base_url)
    base_out = Path(args.out)
    total_errors = 0
    try:
        for round_number in range(1, args.repeat + 1):
            round_out = _round_path(base_out, round_number)
            resume_from = None
            if args.resume and round_out.is_file():
                resume_from = load_predictions(round_out)

            def progress(done: int, total: int, current: int = round_number) -> None:
                prefix = f"round {current}: " if args.repeat > 1 else ""
                print(
                    f"\rrecording {prefix}{done}/{total} cases",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

            records = record(
                pack,
                client,
                concurrency=args.concurrency,
                limit=args.limit,
                progress=progress,
                model=args.model,
                resume_from=resume_from,
                rpm=args.rpm,
                shuffle_options=args.shuffle_options,
            )
            print(file=sys.stderr)
            write_predictions(round_out, records)
            errors = sum(1 for item in records if item["error"])
            total_errors += errors
            print(
                f"{len(records)} cases -> {round_out}" + (f" ({errors} errors)" if errors else "")
            )
            for item in records:
                if item["error"]:
                    print(f"  {item['case_id']}: {item['error']}", file=sys.stderr)
    finally:
        client.close()

    if args.repeat > 1:
        print(_stability_summary(pack, base_out, args.repeat))
    return 1 if total_errors else 0


def _round_path(base: Path, round_number: int) -> Path:
    if round_number == 1:
        return base
    return base.with_name(f"{base.stem}-r{round_number}{base.suffix}")


def _stability_summary(pack: Pack, base_out: Path, repeats: int) -> str:
    base = load_predictions(_round_path(base_out, 1))
    flips = 0
    compared = 0
    for round_number in range(2, repeats + 1):
        other_path = _round_path(base_out, round_number)
        if not other_path.is_file():
            continue
        try:
            result = compare_recordings(pack, base, load_predictions(other_path))
        except ValueError:
            continue
        flips += result.wins + result.losses
        compared = max(compared, result.n_items)
    if not compared:
        return "stability: no comparable rounds"
    return (
        f"stability: {flips} discordant decisions across {repeats} rounds "
        f"({compared} items compared, {flips / compared:.3f} flip rate)"
    )


def _estimate_tokens(pack: Pack, limit: int | None) -> tuple[int, int]:
    cases = pack.cases[:limit] if limit else pack.cases
    question_tokens = len(json.dumps(pack.to_api_questions(), ensure_ascii=False)) // 4 + 1
    tokens = sum(
        len(json.dumps(case.state, ensure_ascii=False)) // 4 + question_tokens for case in cases
    )
    return len(cases), tokens


def _cmd_check(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    predictions = load_predictions(args.predictions)
    _warn_on_recording_mismatch(pack, predictions)
    if args.partition != "all":
        pack = _partition_pack(pack, args.partition, args.partition_seed, args.partition_ratio)
        case_ids = {case.id for case in pack.cases}
        predictions = {
            case_id: item for case_id, item in predictions.items() if case_id in case_ids
        }
        print(
            f"partition: {args.partition} — {len(pack.cases)} cases "
            f"(seed {args.partition_seed}, ratio {args.partition_ratio})",
            file=sys.stderr,
        )
    overall = compute(
        pack,
        predictions,
        bootstrap=args.bootstrap,
        input_usd_per_mtok=args.input_price,
        output_usd_per_mtok=args.output_price,
    )
    gates = [] if args.no_gates else evaluate_gates(pack, overall)
    suggestion = None
    if args.target_precision is not None:
        items, _, _ = build_items(pack, predictions)
        suggestion = suggest_threshold(items, args.target_precision)
    models = _recorded_models(predictions)

    if args.junit:
        Path(args.junit).write_text(render_junit(pack, gates, args.no_gates), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(
            render_markdown(
                pack, overall, gates, suggestion, args.target_precision, models, args.no_gates
            ),
            encoding="utf-8",
        )

    if args.json:
        payload = _payload(pack, overall, gates, suggestion, gates_skipped=args.no_gates)
        payload["recorded_models"] = models
        payload["pricing"] = {
            "input_usd_per_mtok": args.input_price,
            "output_usd_per_mtok": args.output_price,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            _human_summary(
                pack,
                overall,
                gates,
                args.predictions,
                suggestion,
                args.target_precision,
                models,
                args.input_price,
                args.output_price,
                gates_skipped=args.no_gates,
            )
        )
        if args.failures:
            print()
            print(_failures(pack, predictions))

    return 1 if any(gate.ok is False for gate in gates) else 0


def _warn_on_recording_mismatch(pack: Pack, predictions: dict[str, dict[str, Any]]) -> None:
    """Never let a stale or partial recording pass silently."""
    pack_ids = {case.id for case in pack.cases}
    extra = sorted(set(predictions) - pack_ids)
    missing = sorted(pack_ids - set(predictions))
    if extra:
        print(
            f"warning: recording has {len(extra)} case(s) not in the pack: {', '.join(extra[:5])}",
            file=sys.stderr,
        )
    if missing:
        print(
            f"warning: recording is missing {len(missing)} case(s): {', '.join(missing[:5])}",
            file=sys.stderr,
        )
    models = {record.get("model") for record in predictions.values() if record.get("model")}
    if pack.tested and models and models != {pack.tested}:
        print(
            f"warning: recording model(s) {sorted(models)} != pack.tested {pack.tested}",
            file=sys.stderr,
        )
    if pack.tested is None:
        print(
            "note: pack.tested is null — pin it to the recorded model version "
            "once evidence is reviewed",
            file=sys.stderr,
        )


def _partition_pack(pack: Pack, which: str, seed: int, ratio: float) -> Pack:
    """Deterministic dev/test half so thresholds are tuned and verified apart."""
    selected = tuple(case for case in pack.cases if _in_partition(case.id, which, seed, ratio))
    return dataclasses.replace(pack, cases=selected)


def _in_partition(case_id: str, which: str, seed: int, ratio: float) -> bool:
    digest = hashlib.blake2s(f"{seed}:{case_id}".encode(), digest_size=8).digest()
    is_dev = int.from_bytes(digest, "big") / 2**64 < ratio
    return is_dev if which == "dev" else not is_dev


def _failures(pack: Pack, predictions: dict[str, dict[str, Any]]) -> str:
    items, _, _ = build_items(pack, predictions)
    misses = [item for item in items if not item.correct]
    states = {case.id: case.state for case in pack.cases}
    lines = [f"failures: {len(misses)}/{len(items)} items"]
    for item in misses:
        lines.append(
            f"  {item.case_id} {item.qid}: expected={item.expected!r} got={item.got!r} "
            f"p={item.decision_prob:.2f}"
        )
        lines.append(f"    state: {json.dumps(states[item.case_id], ensure_ascii=False)[:200]}")
    return "\n".join(lines)


def _cmd_compare(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    result = compare_recordings(pack, load_predictions(args.a), load_predictions(args.b))
    print(_compare_summary(pack, result, args.a, args.b))
    return 0


def _payload(
    pack: Pack,
    overall: OverallMetrics,
    gates: list[Any],
    suggestion: ThresholdSuggestion | None = None,
    *,
    gates_skipped: bool = False,
) -> dict[str, Any]:
    return {
        "pack": pack.id,
        "version": pack.version,
        "record_model": pack.record_model,
        "tested": pack.tested,
        "n_cases": overall.n_cases,
        "n_items": overall.n_items,
        "case_errors": overall.case_errors,
        "missing_items": overall.missing_items,
        "accuracy": overall.accuracy,
        "accuracy_ci": list(overall.accuracy_ci) if overall.accuracy_ci else None,
        "ece": overall.ece,
        "ece_ci": list(overall.ece_ci) if overall.ece_ci else None,
        "suggestion": (
            {
                "threshold": suggestion.threshold,
                "coverage": suggestion.coverage,
                "precision": suggestion.precision,
                "n_accepted": suggestion.n_accepted,
            }
            if suggestion
            else None
        ),
        "mean_decision_prob": overall.mean_decision_prob,
        "cost_per_case_usd": overall.cost_per_case_usd,
        "total_cost_usd": overall.total_cost_usd,
        "p50_latency_ms": overall.p50_latency_ms,
        "p95_latency_ms": overall.p95_latency_ms,
        "per_question": {
            qid: {
                "type": metrics.qtype,
                "n": metrics.n,
                "missing": metrics.missing,
                "accuracy": metrics.accuracy,
                "mean_decision_prob": metrics.mean_decision_prob,
                "ece": metrics.ece,
                "brier": metrics.brier,
            }
            for qid, metrics in overall.per_question.items()
        },
        "coverage": [
            {
                "threshold": row.threshold,
                "coverage": row.coverage,
                "precision": row.precision,
            }
            for row in overall.coverage
        ],
        "threshold_coverage": (
            {
                "n_total": overall.threshold_coverage.n_total,
                "n_auto": overall.threshold_coverage.n_auto,
                "coverage": overall.threshold_coverage.coverage,
                "precision": overall.threshold_coverage.precision,
            }
            if overall.threshold_coverage
            else None
        ),
        "gates": [{"gate": gate.gate, "ok": gate.ok, "detail": gate.detail} for gate in gates],
        "gates_skipped": gates_skipped,
    }


def _recorded_models(predictions: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        {record["model"] for record in predictions.values() if isinstance(record.get("model"), str)}
    )


def _human_summary(
    pack: Pack,
    overall: OverallMetrics,
    gates: list[Any],
    predictions_path: str,
    suggestion: ThresholdSuggestion | None = None,
    target_precision: float | None = None,
    models: list[str] | None = None,
    input_price: float = INPUT_USD_PER_MTOK,
    output_price: float = 0.0,
    *,
    gates_skipped: bool = False,
) -> str:
    lines: list[str] = []
    tested = f", tested {pack.tested}" if pack.tested else ", provisional"
    record_model = ", ".join(models) if models else pack.record_model
    lines.append(f"jevassert — {pack.id} v{pack.version} (record model {record_model}{tested})")
    lines.append(
        f"recording: {predictions_path} — {overall.n_cases} cases, "
        f"{overall.case_errors} errors, {overall.missing_items} missing, {overall.n_items} items"
    )
    lines.append("")
    lines.append(
        f"{'question':<22} {'type':<7} {'n':>3} {'acc':>7} {'p(dec)':>7} {'ece':>7} {'brier':>7}"
    )
    for qid, metrics in overall.per_question.items():
        brier = f"{metrics.brier:.3f}" if metrics.brier is not None else "—"
        lines.append(
            f"{qid:<22} {metrics.qtype:<7} {metrics.n:>3} {metrics.accuracy:>7.3f} "
            f"{metrics.mean_decision_prob:>7.3f} {metrics.ece:>7.3f} {brier:>7}"
        )
    lines.append(
        f"{'overall':<22} {'':<7} {overall.n_items:>3} {overall.accuracy:>7.3f} "
        f"{overall.mean_decision_prob:>7.3f} {overall.ece:>7.3f}"
    )
    if overall.accuracy_ci is not None:
        low, high = overall.accuracy_ci
        ece_part = ""
        if overall.ece_ci is not None:
            ece_part = f", ECE CI {overall.ece_ci[0]:.3f}–{overall.ece_ci[1]:.3f}"
        lines.append(f"bootstrap 95%: accuracy CI {low:.3f}–{high:.3f}{ece_part}")
    if overall.n_items < SMALL_N_ITEMS:
        lines.append(
            f"note: {overall.n_items} items only — accuracy CI is wide and ECE is coarse "
            "(see README: Reading the numbers)"
        )
    if overall.cost_per_case_usd is not None:
        price = f"${input_price}/M input tokens"
        if output_price:
            price = f"${input_price}/M input + ${output_price}/M output tokens"
        lines.append(
            f"cost/case ${overall.cost_per_case_usd:.6f} "
            f"(total ${overall.total_cost_usd:.4f} at {price})"
        )
    if overall.p95_latency_ms is not None:
        lines.append(
            f"latency p50 {overall.p50_latency_ms:.0f}ms, p95 {overall.p95_latency_ms:.0f}ms"
        )
    lines.append("")
    lines.append("coverage (accept when p >= threshold):")
    for row in overall.coverage:
        precision = f"{row.precision:.3f}" if row.precision is not None else "—"
        lines.append(
            f"  >= {row.threshold:.2f}: coverage {row.coverage:.3f}, precision {precision}"
        )
    if overall.threshold_coverage is not None:
        tc = overall.threshold_coverage
        precision = f"{tc.precision:.3f}" if tc.precision is not None else "—"
        lines.append("")
        lines.append("author thresholds (pack floors; labels without a floor go to review):")
        lines.append(
            f"  auto-accepted {tc.n_auto}/{tc.n_total} ({tc.coverage:.3f}), precision {precision}"
        )
    if suggestion is not None and target_precision is not None:
        lines.append("")
        lines.append(
            f"threshold suggestion (precision >= {target_precision:.2f}): "
            f"p >= {suggestion.threshold:.2f} → coverage {suggestion.coverage:.3f}, "
            f"precision {suggestion.precision:.3f} (n={suggestion.n_accepted})"
        )
    lines.append("")
    lines.append("gates:")
    if gates_skipped:
        lines.append("  (skipped: --no-gates)")
    elif not gates:
        lines.append("  (none declared in gates.yaml)")
    for gate in gates:
        mark = "PASS" if gate.ok else ("SKIP" if gate.ok is None else "FAIL")
        lines.append(f"  {mark} {gate.gate}: {gate.detail}")
    return "\n".join(lines)


def _compare_summary(pack: Pack, result: CompareResult, path_a: str, path_b: str) -> str:
    lines: list[str] = []
    lines.append(f"jevassert compare — {pack.id} v{pack.version}")
    lines.append(f"  A: {path_a}")
    lines.append(f"  B: {path_b}")
    lines.append(
        f"  items {result.n_items}: accuracy A {result.accuracy_a:.3f} "
        f"-> B {result.accuracy_b:.3f} (delta {result.delta:+.3f})"
    )
    lines.append(
        f"  discordant: A-only correct {result.wins}, B-only correct {result.losses} "
        f"— McNemar exact p = {result.p_value:.4f}"
    )
    if result.cost_per_case_a_usd is not None and result.cost_per_case_b_usd is not None:
        lines.append(
            f"  cost/case A ${result.cost_per_case_a_usd:.6f} "
            f"-> B ${result.cost_per_case_b_usd:.6f}"
        )
    lines.append("")
    lines.append(f"{'question':<22} {'n':>3} {'acc A':>7} {'acc B':>7} {'delta':>7}")
    for delta in result.per_question.values():
        lines.append(
            f"{delta.qid:<22} {delta.n:>3} {delta.accuracy_a:>7.3f} "
            f"{delta.accuracy_b:>7.3f} {delta.delta:>+7.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
