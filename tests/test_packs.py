from pathlib import Path

import pytest
from helpers import write_pack

from jevassert.packs import PackError, label_index, load_pack

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

NOUL_Q = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}
NOUL_CASE = {"id": "c-1", "state": {"text": "hello"}, "expect": {"urgent": True}}


def test_example_packs_load() -> None:
    triage = load_pack(EXAMPLES / "demo-triage")
    assert triage.id == "demo-triage"
    assert len(triage.cases) == 10
    assert triage.questions["queue"]["type"] == "choice"
    assert triage.thresholds["queue"]["billing"] == 0.6
    assert triage.gates == {"min_accuracy": 0.8, "max_ece": 0.25}
    assert triage.record_model == "jev-latest"  # tested is null
    assert triage.cases[0].state["message"]

    rag = load_pack(EXAMPLES / "demo-relevance")
    assert rag.id == "demo-relevance"
    assert len(rag.cases) == 8
    assert rag.cases[0].state["question"]


def test_api_questions_mapping(tmp_path: Path) -> None:
    questions = {
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Team?",
            "options": {"billing": "money", "unknown": "can't tell"},
        },
        "severity": {
            "type": "score",
            "instructions": "Severity?",
            "levels": ["low", "high", "unknown"],
            "level_descriptions": {
                "low": "Minor or cosmetic.",
                "high": "Blocking outage.",
                "unknown": "Cannot tell.",
            },
        },
    }
    case = {
        "id": "c-1",
        "state": {"text": "x"},
        "expect": {"urgent": True, "team": "billing", "severity": "low"},
    }
    pack = load_pack(write_pack(tmp_path / "test-pack", questions, [case]))
    api = pack.to_api_questions()
    assert api["urgent"] == {"type": "noul", "instructions": "Urgent?"}
    assert api["team"]["criteria"] == {"billing": "money", "unknown": "can't tell"}
    assert api["severity"]["criteria"] == [
        "Minor or cosmetic.",
        "Blocking outage.",
        "Cannot tell.",
    ]


def test_label_index_follows_level_order() -> None:
    question = {"type": "score", "instructions": "?", "levels": ["low", "high", "unknown"]}
    assert label_index(question, "high") == 1


def test_record_model_follows_tested(tmp_path: Path) -> None:
    pack = load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], tested="jev-1.13.0"))
    assert pack.record_model == "jev-1.13.0"


def test_rejects_wrong_spec(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="spec must be 0"):
        load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], spec=1))


def test_rejects_unknown_pack_key(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="unexpected key `moel`"):
        load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], moel="jev-latest"))


def test_rejects_id_not_matching_directory(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="must equal the directory name"):
        load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], id="other-pack"))


def test_rejects_unknown_question_type(tmp_path: Path) -> None:
    questions = {"urgent": {"type": "likert", "instructions": "?"}}
    case = {"id": "c-1", "state": {"text": "x"}, "expect": {"urgent": True}}
    with pytest.raises(PackError, match="type must be one of"):
        load_pack(write_pack(tmp_path / "test-pack", questions, [case]))


def test_rejects_choice_without_unknown(tmp_path: Path) -> None:
    questions = {
        "team": {
            "type": "choice",
            "instructions": "Team?",
            "options": {"billing": "money", "technical": "bugs"},
        }
    }
    case = {"id": "c-1", "state": {"text": "x"}, "expect": {"team": "billing"}}
    with pytest.raises(PackError, match="must include `unknown`"):
        load_pack(write_pack(tmp_path / "test-pack", questions, [case]))


def test_rejects_score_without_unknown(tmp_path: Path) -> None:
    questions = {
        "severity": {"type": "score", "instructions": "?", "levels": ["low", "high"]},
    }
    case = {"id": "c-1", "state": {"text": "x"}, "expect": {"severity": "low"}}
    with pytest.raises(PackError, match="must include `unknown`"):
        load_pack(write_pack(tmp_path / "test-pack", questions, [case]))


def test_rejects_level_descriptions_for_unknown_level(tmp_path: Path) -> None:
    questions = {
        "severity": {
            "type": "score",
            "instructions": "?",
            "levels": ["low", "high", "unknown"],
            "level_descriptions": {"nope": "text"},
        },
    }
    case = {"id": "c-1", "state": {"text": "x"}, "expect": {"severity": "low"}}
    with pytest.raises(PackError, match="not a level"):
        load_pack(write_pack(tmp_path / "test-pack", questions, [case]))


def test_rejects_expectation_missing_a_question(tmp_path: Path) -> None:
    questions = {
        "urgent": {"type": "noul", "instructions": "?"},
        "team": {
            "type": "choice",
            "instructions": "Team?",
            "options": {"billing": "money", "unknown": "can't tell"},
        },
    }
    case = {"id": "c-1", "state": {"text": "x"}, "expect": {"urgent": True}}
    with pytest.raises(PackError, match="expect keys must be exactly"):
        load_pack(write_pack(tmp_path / "test-pack", questions, [case]))


def test_rejects_state_keys_not_matching_fields(tmp_path: Path) -> None:
    case = {"id": "c-1", "state": {"message": "hello"}, "expect": {"urgent": True}}
    with pytest.raises(PackError, match="state keys must be exactly"):
        load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [case]))


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    pack_dir = write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE, dict(NOUL_CASE)])
    with pytest.raises(PackError, match="duplicate case id"):
        load_pack(pack_dir)


def test_rejects_bad_threshold_label(tmp_path: Path) -> None:
    thresholds = {"urgent": {"maybe": 0.8}}
    with pytest.raises(PackError, match="not a valid answer"):
        load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], thresholds=thresholds))


def test_rejects_threshold_out_of_range(tmp_path: Path) -> None:
    thresholds = {"urgent": {"true": 1.5}}
    with pytest.raises(PackError, match=r"must be in \(0, 1\]"):
        load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], thresholds=thresholds))


def test_rejects_unknown_gate(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="unknown gate"):
        load_pack(
            write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], gates={"min_acuracy": 0.9})
        )


def test_per_question_gates_validated(tmp_path: Path) -> None:
    gates = {"per_question": {"urgent": {"min_accuracy": 0.9}}}
    pack = load_pack(write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE], gates=gates))
    assert pack.gates["per_question"]["urgent"]["min_accuracy"] == 0.9

    with pytest.raises(PackError, match="unknown question"):
        load_pack(
            write_pack(
                tmp_path / "test-pack",
                NOUL_Q,
                [NOUL_CASE],
                gates={"per_question": {"nope": {"min_accuracy": 0.9}}},
            )
        )

    with pytest.raises(PackError, match="unknown gate"):
        load_pack(
            write_pack(
                tmp_path / "test-pack",
                NOUL_Q,
                [NOUL_CASE],
                gates={"per_question": {"urgent": {"min_accuracy_ci_lower": 0.5}}},
            )
        )


def test_requires_cases_file(tmp_path: Path) -> None:
    pack_dir = write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE])
    (pack_dir / "cases.jsonl").unlink()
    with pytest.raises(PackError, match="cases.jsonl"):
        load_pack(pack_dir)


def test_require_cases_false_allows_missing_cases(tmp_path: Path) -> None:
    pack_dir = write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE])
    (pack_dir / "cases.jsonl").unlink()
    pack = load_pack(pack_dir, require_cases=False)
    assert pack.cases == ()
    assert pack.id == "test-pack"


def test_cases_file_errors_are_located(tmp_path: Path) -> None:
    pack_dir = write_pack(tmp_path / "test-pack", NOUL_Q, [NOUL_CASE])
    (pack_dir / "cases.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(PackError, match="invalid JSON"):
        load_pack(pack_dir)
