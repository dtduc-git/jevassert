# jevassert — agent notes

Record/replay regression tests for Jev (TypeSafe System One) question packs.
Public repo: https://github.com/dtduc-git/jevassert · Apache-2.0.

## Layout

- `src/jevassert/packs.py` — pack **spec v0** loader + validator. **Owns the
  loader** that other tools (jev-table, packs CI) import. The canonical spec
  lives in the jev-packs repo `SPEC.md`; this module implements it plus
  `gates.yaml` (this repo's format). `level_descriptions` is now part of SPEC v0
  (`jev-packs` commit `1d7ae77`). Do not change the format here unilaterally.
- `src/jevassert/client.py` — one HTTP client for `/v1/systemone`, retries on
  429/5xx, injectable transport. The API key comes from `TYPESAFE_API_KEY`
  only; never accept literal keys on the CLI, never log them.
- `src/jevassert/backends.py` — `AdapterClient`: same interface, but records a
  general-purpose LLM through the official `system-one-adapter`
  (`--backend openai|anthropic`, optional extra `jevassert[adapter]`). Used by
  the jev-packs benchmark; maps adapter errors onto `JevError` (auth fail-fast).
- `src/jevassert/runner.py` — `record` (concurrent, per-case, `model=` override)
  and predictions JSONL read/write. One line per case, stable case order.
- `src/jevassert/metrics.py` — items, accuracy, ECE (equal-mass bins), Brier
  (Noul), coverage table, author-threshold coverage, cost/latency (prices
  injectable for non-Jev backends), gates. Definitions are in the module
  docstring — keep them.
- `src/jevassert/compare.py` — paired comparison + exact McNemar (no scipy).
- `src/jevassert/report.py` — markdown report (spec `evidence.md`) + JUnit XML.
- `src/jevassert/cli.py` — `record | check | compare`, exit 0/1/2. `check`
  warns on model-version mismatch and missing/extra cases, `--failures` lists
  every mismatch for hand review.
- `examples/demo-triage`, `examples/demo-relevance` — spec-compliant example
  packs with `gates.yaml`, also used as fixtures.
- `tests/helpers.py` — pack/prediction builders; tests never touch the network.

## Commands

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run jevassert check examples/demo-triage -p <recording.jsonl> --failures
```

## Rules

- Tests must run offline: use `FakeTransport`/`FakeClient`, never a live key.
  `tests/conftest.py` deletes `TYPESAFE_API_KEY`/`TYPESAFE_BASE_URL` for every
  test — keep it that way.
- `check` never calls the model. Live calls happen only in `record`.
- Never write keys or recordings with private data into the repo;
  `predictions*.jsonl` is gitignored except under `examples/`.
- Claims in README/reports must come from a real recording on a public pack.
- Coverage thresholds and ECE bins are part of the public contract — changing
  them needs a test update and a note in the PR.
- Label first, gates second: never tune labels or gates to make a number look
  better. Every disagreement gets a hand review; fix the label only when the
  label is genuinely wrong, otherwise keep the miss as a test case.

## Planned work (target: all)

Reconcile with SPEC was done 2026-09-19. Status:

1. ~~`check --failures`~~ done. ~~Bootstrap CI + small-n note + `min_accuracy_ci_lower` gate~~ done.
2. ~~`record --shuffle-options` (seed)~~ done — compare against the base run.
   Paraphrase variants still open.
3. ~~Record fail-fast on 401/403 and missing key~~ done.
4. ~~`record --resume`, `--rpm`, `--dry-run` cost estimate~~ done.
5. ~~Per-question gates in `gates.yaml`~~ done (`min_accuracy`, `max_ece`).
6. ~~`record --repeat N` self-consistency~~ done (extra rounds + discordant
   summary); ~~dev/test split~~ done (`check --partition dev|test`).
7. ~~Threshold suggestion (`--target-precision`)~~ done.
8. ~~Record evidence for the jev-packs registry packs~~ done 2026-09-19:
   9 packs verified (2,990 cases / 6,430 items), evidence regenerated from the
   committed recordings, jev-packs validator green.
9. ~~Publish: GitHub repo + push~~ done 2026-09-19; ~~PyPI trusted publisher +
   `v0.1.0` release~~ done (PyPI `jevassert` 0.1.0, `uvx jevassert` verified).
   Action Marketplace listing still open.
10. **0.2.0 released 2026-09-19** (PyPI + tag + Release): adapter backends
    (`--backend openai|anthropic|bedrock`), `check --input-price/--output-price`,
    reports the recorded models, action.yml price inputs, `[adapter]` extra.
11. **0.2.1 (this release)**: `check --no-gates` skips `gates.yaml` entirely —
    no parsing, no validation, no evaluation (metrics/report only, always exit
    0); human summary, markdown report, JSON and JUnit all mark the skip;
    Action input `no-gates` (case-insensitive, requires version >= 0.2.1).
    Next: jev-packs replaces its symlink workaround and bumps its pin to
    v0.2.1, then move the `v0` tag (still at 0.1.0) to the new release.
