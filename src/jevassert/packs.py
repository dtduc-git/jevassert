"""Pack spec v0 loader: question packs + labeled cases.

Canonical spec: the jev-packs repo `SPEC.md` (spec v0). This module implements
the loader contract; the jevassert CLI, jev-table and packs CI all import it.

Layout::

    <pack>/
      pack.yaml      # metadata, state contract, questions, optional thresholds
      cases.jsonl    # golden cases, one JSON object per line
      gates.yaml     # optional — jevassert quality gates (format owned here)

``pack.yaml`` keys are exactly ``spec, id, version, license, tested,
description, state, questions, thresholds`` — unknown keys are rejected so typos
cannot pass silently. Questions use SPEC label shapes:

- ``noul``:  ``{type, instructions}`` — boolean
- ``choice``: ``{type, instructions, options: {label: meaning}}``
- ``score``: ``{type, instructions, levels: [label, ...]}``

Every closed set must include the label ``unknown`` (Jev cannot abstain).
``score`` questions may add ``level_descriptions: {label: "situation text"}``
(SPEC v0 rule 5, adopted in jev-packs commit 1d7ae77): the loader sends the
description to the API instead of the bare label. When absent, labels are sent
as-is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPEC_VERSION = 0
QUESTION_TYPES = ("noul", "choice", "score")
GATE_KEYS = (
    "min_accuracy",
    "max_ece",
    "max_cost_per_case_usd",
    "max_p95_latency_ms",
    "min_coverage_at_precision",
    "min_accuracy_ci_lower",
    "per_question",
)
PER_QUESTION_GATE_KEYS = ("min_accuracy", "max_ece")
ABSTAIN_LABEL = "unknown"
PACK_KEYS = {
    "spec",
    "id",
    "version",
    "license",
    "tested",
    "description",
    "state",
    "questions",
    "thresholds",
}
ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
KEY_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
TESTED_RE = re.compile(r"^jev-\d+\.\d+\.\d+$")


class PackError(ValueError):
    """Invalid pack.yaml, cases.jsonl or gates.yaml."""


@dataclass(frozen=True)
class Case:
    id: str
    state: dict[str, Any]
    expect: dict[str, Any]


@dataclass(frozen=True)
class Pack:
    path: Path
    id: str
    version: str
    license: str
    tested: str | None
    description: str
    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]
    thresholds: dict[str, dict[str, float]]
    gates: dict[str, Any]
    cases: tuple[Case, ...]

    @property
    def record_model(self) -> str:
        """Model to send when recording: the verified version when pinned."""
        return self.tested or "jev-latest"

    def to_api_questions(self) -> dict[str, dict[str, Any]]:
        """Map SPEC questions to the System One API ``questions`` shape."""
        api: dict[str, dict[str, Any]] = {}
        for qid, question in self.questions.items():
            qtype = question["type"]
            if qtype == "noul":
                api[qid] = {"type": "noul", "instructions": question["instructions"]}
            elif qtype == "choice":
                api[qid] = {
                    "type": "choice",
                    "instructions": question["instructions"],
                    "criteria": dict(question["options"]),
                }
            else:
                descriptions = question.get("level_descriptions") or {}
                api[qid] = {
                    "type": "score",
                    "instructions": question["instructions"],
                    "criteria": [descriptions.get(level, level) for level in question["levels"]],
                }
        return api

    def labels(self, qid: str) -> set[str]:
        """Valid answer labels for a question, normalized to strings."""
        question = self.questions[qid]
        if question["type"] == "noul":
            return {"true", "false"}
        if question["type"] == "choice":
            return set(question["options"])
        return set(question["levels"])


def load_pack(path: str | Path, *, require_cases: bool = True, skip_gates: bool = False) -> Pack:
    """Load and validate a pack directory (or a direct path to pack.yaml).

    Registry packs must carry golden cases (SPEC.md); consumers that classify
    unlabeled data (jev-table column specs) may pass ``require_cases=False``.
    """
    path = Path(path)
    pack_file = path / "pack.yaml" if path.is_dir() else path
    pack_dir = pack_file.parent
    if not pack_file.is_file():
        raise PackError(f"pack file not found: {pack_file}")
    try:
        raw = yaml.safe_load(pack_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - message passthrough
        raise PackError(f"{pack_file}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PackError(f"{pack_file}: top level must be a map")

    _check_keys(pack_file, raw, required=PACK_KEYS - {"thresholds"}, allowed=set(PACK_KEYS))

    if raw.get("spec") != SPEC_VERSION:
        raise PackError(f"{pack_file}: spec must be {SPEC_VERSION}, got {raw.get('spec')!r}")

    pack_id = _require_str(raw, "id", pack_file)
    if not ID_RE.match(pack_id):
        raise PackError(f"{pack_file}: id must be kebab-case, got {pack_id!r}")
    if path.is_dir() and pack_id != pack_dir.name:
        raise PackError(f"{pack_file}: id must equal the directory name ({pack_dir.name})")
    version = _require_str(raw, "version", pack_file)
    if not SEMVER_RE.match(version):
        raise PackError(f"{pack_file}: version must be semver, got {version!r}")
    license_id = _require_str(raw, "license", pack_file)
    description = _require_str(raw, "description", pack_file)

    tested = raw.get("tested")
    if tested is not None and not (isinstance(tested, str) and TESTED_RE.match(tested)):
        raise PackError(f"{pack_file}: tested must be null or 'jev-<semver>', got {tested!r}")

    state = _validate_state(raw.get("state"), pack_file)
    questions = _validate_questions(raw.get("questions"), pack_file)
    thresholds = _validate_thresholds(raw.get("thresholds"), questions, pack_file)
    gates = {} if skip_gates else _load_gates(pack_dir, questions)
    cases_file = pack_dir / "cases.jsonl"
    cases = _load_cases(cases_file, state, questions, pack_file) if cases_file.is_file() else ()
    if require_cases and not cases:
        raise PackError(f"{pack_file}: cases.jsonl with at least one case is required")

    return Pack(
        path=pack_dir,
        id=pack_id,
        version=version,
        license=license_id,
        tested=tested,
        description=description,
        state=state,
        questions=questions,
        thresholds=thresholds,
        gates=gates,
        cases=cases,
    )


def _check_keys(
    source: Path | str, obj: dict[str, Any], required: set[str], allowed: set[str]
) -> None:
    for key in sorted(required - obj.keys()):
        raise PackError(f"{source}: missing required key `{key}`")
    for key in sorted(obj.keys() - allowed):
        raise PackError(f"{source}: unexpected key `{key}` (typo? not part of spec v0)")


def _require_str(raw: dict[str, Any], key: str, source: Path | str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PackError(f"{source}: {key} must be a non-empty string")
    return value


def _validate_state(state: Any, source: Path) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise PackError(f"{source}: state must be a map")
    _check_keys(
        source, state, required={"description", "fields"}, allowed={"description", "fields"}
    )
    description = state.get("description")
    if not isinstance(description, str) or not description.strip():
        raise PackError(f"{source}: state.description must be a non-empty string")
    fields = state.get("fields")
    if (
        not isinstance(fields, list)
        or not fields
        or not all(isinstance(field, str) and KEY_RE.match(field) for field in fields)
    ):
        raise PackError(f"{source}: state.fields must be snake_case field names")
    if len(set(fields)) != len(fields):
        raise PackError(f"{source}: state.fields contains duplicates")
    return state


def _validate_questions(questions: Any, source: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(questions, dict) or not questions:
        raise PackError(f"{source}: questions must be a non-empty map")
    for qid, question in questions.items():
        where = f"{source}: question '{qid}'"
        if not KEY_RE.match(qid):
            raise PackError(f"{where}: question id must be snake_case")
        if not isinstance(question, dict):
            raise PackError(f"{where} must be a map")
        qtype = question.get("type")
        if qtype not in QUESTION_TYPES:
            raise PackError(f"{where}: type must be one of {', '.join(QUESTION_TYPES)}")
        _require_str(question, "instructions", where)

        if qtype == "noul":
            _check_keys(
                where, question, required={"type", "instructions"}, allowed={"type", "instructions"}
            )
        elif qtype == "choice":
            _check_keys(
                where,
                question,
                required={"type", "instructions", "options"},
                allowed={"type", "instructions", "options"},
            )
            _validate_choice_options(question.get("options"), where)
        else:
            _check_keys(
                where,
                question,
                required={"type", "instructions", "levels"},
                allowed={"type", "instructions", "levels", "level_descriptions"},
            )
            _validate_score_levels(question, where)
    return questions


def _validate_choice_options(options: Any, where: str) -> None:
    if not isinstance(options, dict) or len(options) < 2:
        raise PackError(f"{where}: choice options must be a map with at least 2 options")
    for label, meaning in options.items():
        if not isinstance(label, str) or not KEY_RE.match(label):
            raise PackError(f"{where}: option key {label!r} must be snake_case")
        if not isinstance(meaning, str) or not meaning.strip():
            raise PackError(f"{where}: option `{label}` needs a one-line meaning")
    if ABSTAIN_LABEL not in options:
        raise PackError(
            f"{where}: choice options must include `{ABSTAIN_LABEL}` (Jev cannot abstain)"
        )


def _validate_score_levels(question: dict[str, Any], where: str) -> None:
    levels = question.get("levels")
    if (
        not isinstance(levels, list)
        or not 2 <= len(levels) <= 10
        or not all(isinstance(level, str) and KEY_RE.match(level) for level in levels)
    ):
        raise PackError(f"{where}: score levels must be 2-10 snake_case labels")
    if ABSTAIN_LABEL not in levels:
        raise PackError(
            f"{where}: score levels must include `{ABSTAIN_LABEL}` (Jev cannot abstain)"
        )
    descriptions = question.get("level_descriptions")
    if descriptions is not None:
        if not isinstance(descriptions, dict):
            raise PackError(f"{where}: level_descriptions must be a map of level -> text")
        for level, text in descriptions.items():
            if level not in levels:
                raise PackError(f"{where}: level_descriptions key '{level}' is not a level")
            if not isinstance(text, str) or not text.strip():
                raise PackError(
                    f"{where}: level_descriptions['{level}'] must be a non-empty string"
                )


def _validate_thresholds(
    thresholds: Any, questions: dict[str, dict[str, Any]], source: Path
) -> dict[str, dict[str, float]]:
    if thresholds is None:
        return {}
    if not isinstance(thresholds, dict):
        raise PackError(f"{source}: thresholds must be a map")
    normalized: dict[str, dict[str, float]] = {}
    for qid, floors in thresholds.items():
        where = f"{source}: thresholds.{qid}"
        if qid not in questions:
            raise PackError(f"{where}: unknown question")
        if not isinstance(floors, dict) or not floors:
            raise PackError(f"{where}: must map label -> probability")
        valid_labels = _labels_for(questions[qid])
        normalized[qid] = {}
        for label, probability in floors.items():
            key = _normalize_label(label)
            if key not in valid_labels:
                raise PackError(f"{where}: label {label!r} is not a valid answer")
            if isinstance(probability, bool) or not isinstance(probability, int | float):
                raise PackError(f"{where}: threshold for {label!r} must be a number in (0, 1]")
            value = float(probability)
            if not 0 < value <= 1:
                raise PackError(f"{where}: threshold for {label!r} must be in (0, 1]")
            normalized[qid][key] = value
    return normalized


def _labels_for(question: dict[str, Any]) -> set[str]:
    if question["type"] == "noul":
        return {"true", "false"}
    if question["type"] == "choice":
        return set(question["options"])
    return set(question["levels"])


def _normalize_label(label: Any) -> str:
    if label is True:
        return "true"
    if label is False:
        return "false"
    return str(label)


def _load_gates(pack_dir: Path, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Load optional gates.yaml (jevassert's own file, absent from SPEC pack.yaml)."""
    gates_file = pack_dir / "gates.yaml"
    if not gates_file.is_file():
        return {}
    try:
        raw = yaml.safe_load(gates_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover
        raise PackError(f"{gates_file}: invalid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise PackError(f"{gates_file}: top level must be a map")
    unknown = set(raw) - set(GATE_KEYS)
    if unknown:
        raise PackError(f"{gates_file}: unknown gate(s): {', '.join(sorted(unknown))}")
    for key, value in raw.items():
        if key == "min_coverage_at_precision":
            if not isinstance(value, dict) or not {"precision", "min_coverage"} <= set(value):
                raise PackError(
                    f"{gates_file}: min_coverage_at_precision takes {{precision, min_coverage}}"
                )
        elif key == "per_question":
            if not isinstance(value, dict) or not value:
                raise PackError(f"{gates_file}: per_question must map question id -> gates")
            for qid, per_question in value.items():
                where = f"{gates_file}: per_question.{qid}"
                if qid not in questions:
                    raise PackError(f"{where}: unknown question")
                if not isinstance(per_question, dict) or not per_question:
                    raise PackError(f"{where}: must map gate name -> value")
                wrong = set(per_question) - set(PER_QUESTION_GATE_KEYS)
                if wrong:
                    raise PackError(f"{where}: unknown gate(s): {', '.join(sorted(wrong))}")
                for gate_value in per_question.values():
                    if isinstance(gate_value, bool) or not isinstance(gate_value, int | float):
                        raise PackError(f"{where}: gate values must be numbers")
        elif isinstance(value, bool) or not isinstance(value, int | float):
            raise PackError(f"{gates_file}: gate {key} must be a number")
    return raw


def _load_cases(
    cases_file: Path,
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    source: Path,
) -> tuple[Case, ...]:
    if not cases_file.is_file():
        raise PackError(f"{source}: cases.jsonl is required next to pack.yaml")
    fields = list(state["fields"])
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, line in enumerate(cases_file.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            raise PackError(f"{cases_file}:{lineno}: blank line")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PackError(f"{cases_file}:{lineno}: invalid JSON: {exc}") from exc
        where = f"{cases_file}:{lineno}"
        if not isinstance(raw, dict):
            raise PackError(f"{where}: case must be an object")
        _check_keys(
            where, raw, required={"id", "state", "expect"}, allowed={"id", "state", "expect"}
        )

        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise PackError(f"{where}: id must be a non-empty string")
        if case_id in seen:
            raise PackError(f"{where}: duplicate case id '{case_id}'")
        seen.add(case_id)

        case_state = raw.get("state")
        if not isinstance(case_state, dict) or set(case_state) != set(fields):
            raise PackError(f"{where}: state keys must be exactly {sorted(fields)}")

        expect = raw.get("expect")
        if not isinstance(expect, dict) or set(expect) != set(questions):
            raise PackError(f"{where}: expect keys must be exactly {sorted(questions)}")
        for qid, label in expect.items():
            _validate_expectation(label, questions[qid], qid, where)

        cases.append(Case(id=case_id, state=case_state, expect=expect))
    return tuple(cases)


def _validate_expectation(label: Any, question: dict[str, Any], qid: str, where: str) -> None:
    qtype = question["type"]
    if qtype == "noul":
        if not isinstance(label, bool):
            raise PackError(f"{where}: `{qid}` expects true/false")
    elif qtype == "choice":
        if not isinstance(label, str) or label not in question["options"]:
            raise PackError(f"{where}: `{qid}` expects one of {sorted(question['options'])}")
    else:
        if not isinstance(label, str) or label not in question["levels"]:
            raise PackError(f"{where}: `{qid}` expects one of {question['levels']}")


def label_index(question: dict[str, Any], label: str) -> int:
    """Index of a level label inside a score question (order = API order)."""
    return list(question["levels"]).index(label)
