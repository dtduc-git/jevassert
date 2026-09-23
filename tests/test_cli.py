import json
import xml.etree.ElementTree as ET
from pathlib import Path

from helpers import noul, record, write_pack, write_predictions

from jevassert import cli
from jevassert.runner import load_predictions

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Urgent?"}}
CASES = [
    {"id": "c-1", "state": {"text": "a"}, "expect": {"urgent": True}},
    {"id": "c-2", "state": {"text": "b"}, "expect": {"urgent": False}},
]
GATES = {"min_accuracy": 0.9}
USAGE = {"input_tokens": 100, "output_tokens": 2}


def make_pack(tmp_path: Path, **overrides) -> Path:
    return write_pack(tmp_path / "test-pack", QUESTIONS, CASES, gates=GATES, **overrides)


def good_predictions(tmp_path: Path) -> Path:
    return write_predictions(
        tmp_path / "good.jsonl",
        [
            record("c-1", {"urgent": noul(0.9)}, usage=USAGE),
            record("c-2", {"urgent": noul(0.2)}, usage=USAGE),
        ],
    )


def bad_predictions(tmp_path: Path) -> Path:
    return write_predictions(
        tmp_path / "bad.jsonl",
        [
            record("c-1", {"urgent": noul(0.9)}, usage=USAGE),
            record("c-2", {"urgent": noul(0.9)}, usage=USAGE),
        ],
    )


def test_check_passes_with_good_recording(tmp_path: Path, capsys) -> None:
    code = cli.main(["check", str(make_pack(tmp_path)), "-p", str(good_predictions(tmp_path))])
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS min_accuracy" in out
    assert "accuracy" in out


def test_check_fails_with_bad_recording(tmp_path: Path, capsys) -> None:
    code = cli.main(["check", str(make_pack(tmp_path)), "-p", str(bad_predictions(tmp_path))])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL min_accuracy" in out


def test_check_failures_lists_mismatches(tmp_path: Path, capsys) -> None:
    code = cli.main(
        ["check", str(make_pack(tmp_path)), "-p", str(bad_predictions(tmp_path)), "--failures"]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "failures: 1/2 items" in out
    assert "c-2 urgent: expected=False got=True" in out


def test_check_warns_when_recording_is_missing_cases(tmp_path: Path, capsys) -> None:
    partial = write_predictions(
        tmp_path / "partial.jsonl", [record("c-1", {"urgent": noul(0.9)}, usage=USAGE)]
    )
    cli.main(["check", str(make_pack(tmp_path)), "-p", str(partial)])
    err = capsys.readouterr().err
    assert "missing 1 case" in err


def test_check_warns_on_model_mismatch(tmp_path: Path, capsys) -> None:
    pack = make_pack(tmp_path, tested="jev-1.13.0")
    predictions = write_predictions(
        tmp_path / "other.jsonl",
        [
            record("c-1", {"urgent": noul(0.9)}, usage=USAGE, model="jev-1.14.0"),
            record("c-2", {"urgent": noul(0.2)}, usage=USAGE, model="jev-1.14.0"),
        ],
    )
    cli.main(["check", str(pack), "-p", str(predictions)])
    assert "!= pack.tested" in capsys.readouterr().err


def test_check_writes_junit_and_report(tmp_path: Path) -> None:
    junit = tmp_path / "out" / "junit.xml"
    report = tmp_path / "out" / "report.md"
    junit.parent.mkdir()
    code = cli.main(
        [
            "check",
            str(make_pack(tmp_path)),
            "-p",
            str(bad_predictions(tmp_path)),
            "--junit",
            str(junit),
            "--report",
            str(report),
        ]
    )
    assert code == 1
    suite = ET.parse(junit).getroot()
    assert suite.attrib["failures"] == "1"
    assert "jevassert report — test-pack" in report.read_text(encoding="utf-8")


def test_check_no_gates_skips_failing_gates(tmp_path: Path, capsys) -> None:
    report = tmp_path / "no-gates.md"
    junit = tmp_path / "no-gates.xml"
    code = cli.main(
        [
            "check",
            str(make_pack(tmp_path)),
            "-p",
            str(bad_predictions(tmp_path)),
            "--report",
            str(report),
            "--junit",
            str(junit),
            "--no-gates",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0  # gates.yaml would fail min_accuracy
    assert "(skipped: --no-gates)" in out
    assert "FAIL" not in out
    assert "Gates skipped (`--no-gates`)." in report.read_text(encoding="utf-8")
    suite = ET.parse(junit).getroot()
    assert suite.attrib["tests"] == "1"
    assert suite.attrib["failures"] == "0"
    assert suite.find("testcase/skipped").attrib["message"] == "--no-gates"


def test_check_no_gates_json_marks_skipped(tmp_path: Path, capsys) -> None:
    code = cli.main(
        [
            "check",
            str(make_pack(tmp_path)),
            "-p",
            str(bad_predictions(tmp_path)),
            "--json",
            "--no-gates",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["gates"] == []
    assert payload["gates_skipped"] is True


def test_check_json_output(tmp_path: Path, capsys) -> None:
    code = cli.main(
        [
            "check",
            str(make_pack(tmp_path)),
            "-p",
            str(good_predictions(tmp_path)),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accuracy"] == 1.0
    assert payload["gates"][0]["ok"] is True
    assert "accuracy_ci" in payload
    assert "suggestion" in payload


def test_record_dry_run_needs_no_key_or_network(tmp_path: Path, capsys) -> None:
    code = cli.main(["record", str(make_pack(tmp_path)), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "nothing sent" in out
    assert "2 cases" in out


def test_record_repeat_writes_rounds_and_reports_stability(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class FakeClient:
        api_key = "test-key"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def system_one(self, state, questions, model):
            return (
                {
                    "model": "jev-1.13.0",
                    "answers": {"urgent": {"type": "noul", "noul": 0.9}},
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
                5.0,
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "JevClient", FakeClient)
    out = tmp_path / "rec.jsonl"
    code = cli.main(["record", str(make_pack(tmp_path)), "-o", str(out), "--repeat", "2"])
    assert code == 0
    assert out.is_file()
    assert (tmp_path / "rec-r2.jsonl").is_file()
    assert "stability: 0 discordant decisions across 2 rounds" in capsys.readouterr().out


def test_partition_splits_deterministically(tmp_path: Path, capsys) -> None:
    cases = [
        {
            "id": f"c-{index}",
            "state": {"text": str(index)},
            "expect": {"urgent": index % 2 == 0},
        }
        for index in range(1, 5)
    ]
    pack = write_pack(tmp_path / "part-pack", QUESTIONS, cases, gates=GATES)
    predictions = write_predictions(
        tmp_path / "part.jsonl",
        [
            record(
                f"c-{index}",
                {"urgent": noul(0.9 if index % 2 == 0 else 0.1)},
                usage=USAGE,
            )
            for index in range(1, 5)
        ],
    )
    dev_code = cli.main(["check", str(pack), "-p", str(predictions), "--partition", "dev"])
    dev_err = capsys.readouterr().err
    test_code = cli.main(["check", str(pack), "-p", str(predictions), "--partition", "test"])
    test_err = capsys.readouterr().err
    assert dev_code == 0 and test_code == 0
    assert "partition: dev — 2 cases" in dev_err
    assert "partition: test — 2 cases" in test_err

    again_code = cli.main(["check", str(pack), "-p", str(predictions), "--partition", "dev"])
    assert again_code == 0
    assert "partition: dev — 2 cases" in capsys.readouterr().err


def test_compare_reports_delta_and_mcnemar(tmp_path: Path, capsys) -> None:
    code = cli.main(
        [
            "compare",
            str(make_pack(tmp_path)),
            "--a",
            str(good_predictions(tmp_path)),
            "--b",
            str(bad_predictions(tmp_path)),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "McNemar exact p" in out
    assert "-0.500" in out  # delta from 1.000 to 0.500


def test_record_with_fake_client(tmp_path: Path, monkeypatch) -> None:
    class FakeClient:
        api_key = "test-key"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def system_one(self, state, questions, model):
            return (
                {
                    "model": "jev-1.13.0",
                    "answers": {"urgent": {"type": "noul", "noul": 0.9}},
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
                12.5,
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "JevClient", FakeClient)
    out = tmp_path / "recorded.jsonl"
    code = cli.main(["record", str(make_pack(tmp_path)), "-o", str(out)])
    assert code == 0
    loaded = load_predictions(out)
    assert sorted(loaded) == ["c-1", "c-2"]
    assert loaded["c-1"]["latency_ms"] == 12.5


def test_missing_predictions_file_is_a_usage_error(tmp_path: Path, capsys) -> None:
    code = cli.main(["check", str(make_pack(tmp_path)), "-p", str(tmp_path / "missing.jsonl")])
    assert code == 2
    assert "error" in capsys.readouterr().err
