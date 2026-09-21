"""STEP 4-C2 C3/B2 경계·독립성·semantic hash·동결 검증을 테스트한다."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from drowsiness_detection.evaluation_v2 import missing_policy_freeze as freeze
from drowsiness_detection.evaluation_v2.missing_policy_analysis import VideoState


def _config() -> dict:
    return {"version": 1, "context": deepcopy(freeze.EXPECTED_CONTEXT),
            "behavior": deepcopy(freeze.EXPECTED_BEHAVIOR), "global": deepcopy(freeze.EXPECTED_GLOBAL)}


def _state(video_id: str = "v0", missing: tuple[int, ...] = (), valid: int = 100,
           run: int = 0, split: str = "train", label: str = "drowsy") -> VideoState:
    return VideoState(video_id, split, label, tuple(index not in missing for index in range(32)),
                      tuple(range(32)), tuple(index * 0.3 for index in range(32)), valid, 100 - valid, run)


def test_context_c3_count_and_run_boundaries() -> None:
    config = _config()["context"]
    assert freeze.context_exclusion_reason(_state(missing=(1, 2, 3, 4, 8, 9, 10, 11)), config) == "NONE"
    assert freeze.context_exclusion_reason(_state(missing=(1, 2, 3, 4, 8, 9, 10, 11, 20)), config) == "TOO_MANY_CONTEXT_MISSING"
    assert freeze.context_exclusion_reason(_state(missing=(1, 2, 3, 4, 5)), config) == "CONTEXT_RUN_TOO_LONG"
    assert freeze.context_exclusion_reason(_state(missing=(1, 2, 3, 4, 5, 8, 9, 10, 11)), config) == "BOTH_CONTEXT_LIMITS_EXCEEDED"
    assert freeze.context_exclusion_reason(_state(missing=tuple(range(32))), config) == "NO_VALID_CONTEXT"


def test_behavior_b2_boundaries() -> None:
    config = _config()["behavior"]
    assert freeze.behavior_exclusion_reason(_state(valid=95, run=5), config) == "NONE"
    assert freeze.behavior_exclusion_reason(_state(valid=94, run=5), config) == "INSUFFICIENT_VALID_BEHAVIOR"
    assert freeze.behavior_exclusion_reason(_state(valid=95, run=6), config) == "BEHAVIOR_RUN_TOO_LONG"
    assert freeze.behavior_exclusion_reason(_state(valid=94, run=6), config) == "BOTH_BEHAVIOR_LIMITS_EXCEEDED"
    assert freeze.behavior_exclusion_reason(_state(valid=0, run=100), config) == "NO_VALID_BEHAVIOR"


def test_branch_independence_label_independence_and_test_guard() -> None:
    config = _config()
    policy_hash = freeze.sequence_missing_policy_hash(config)
    context_only = freeze.manifest_row(_state(missing=(), valid=94, run=6), config, policy_hash)
    behavior_only = freeze.manifest_row(_state(missing=tuple(range(9)), valid=100), config, policy_hash)
    assert context_only["context_eligible"] is True and context_only["behavior_eligible"] is False
    assert behavior_only["context_eligible"] is False and behavior_only["behavior_eligible"] is True
    other_label = freeze.manifest_row(_state(label="not_drowsy", missing=(), valid=94, run=6), config, policy_hash)
    assert (other_label["context_eligible"], other_label["behavior_eligible"]) == (True, False)
    assert "global_video_eligible" not in context_only
    with pytest.raises(ValueError, match="STEP_4C2_BLOCKED_TEST_LEAKAGE"):
        freeze.manifest_row(_state(split="test"), config, policy_hash)


def test_semantic_hash_deterministic_nonsemantic_and_sensitive() -> None:
    config = _config()
    freeze.validate_config(config)
    original = freeze.sequence_missing_policy_hash(config)
    shuffled = {"global": config["global"], "behavior": config["behavior"],
                "context": config["context"], "version": 1, "metadata": {"created_at": "tomorrow", "path": "elsewhere"}}
    freeze.validate_config(shuffled)
    assert freeze.sequence_missing_policy_hash(shuffled) == original
    changed = deepcopy(config)
    changed["context"]["max_missing_count"] = 7
    assert freeze.sequence_missing_policy_hash(changed) != original
    with pytest.raises(ValueError, match="Context C3"):
        freeze.validate_config(changed)


def _csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _population() -> list[VideoState]:
    states = [_state(f"v{index}", split="train" if index < 1452 else "val",
                     label="drowsy" if index % 2 == 0 else "not_drowsy") for index in range(1763)]
    states[0] = _state("n_246", missing=tuple(range(32)), valid=0, run=100, label="not_drowsy")
    for index in range(1, 86):
        old = states[index]
        states[index] = _state(old.video_id, missing=tuple(range(9)), split=old.split, label=old.label)
    for index in range(86, 328):
        old = states[index]
        states[index] = _state(old.video_id, valid=94, run=6, split=old.split, label=old.label)
    return states


def _inputs(tmp_path: Path, states: list[VideoState]) -> tuple[Path, Path]:
    canonical = tmp_path / "canonical"
    audit = tmp_path / "audit"
    c1 = audit / "step4c_policy_analysis"
    c1.mkdir(parents=True)
    (c1 / "step4c_policy_analysis_summary.json").write_text(json.dumps({
        "input_videos": 1763, "test_rows": 0, "canonical_policy_hash": freeze.POLICY_HASH,
        "policy_selected": False,
        "context_candidates": [{"candidate": "C3_MODERATE_GAP", "eligible_videos": 1677}],
        "behavior_candidates": [{"candidate": "B2_COVERAGE_95", "eligible_videos": 1520}],
    }), encoding="utf-8")
    _csv(c1 / "context_policy_candidate_impact.csv", ["candidate", "eligible_videos", "min_stratum_retention", "substituted_slot_fraction", "max_missing_run_retained", "max_missing_time_span_sec"],
         [{"candidate": "C3_MODERATE_GAP", "eligible_videos": 1677, "min_stratum_retention": 0.92,
           "substituted_slot_fraction": 0.013, "max_missing_run_retained": 4, "max_missing_time_span_sec": 1.0},
          {"candidate": "C4_COVERAGE_ONLY", "eligible_videos": 1704, "min_stratum_retention": 0.94,
           "substituted_slot_fraction": 0.016, "max_missing_run_retained": 8, "max_missing_time_span_sec": 2.2},
          {"candidate": "C2_SHORT_GAP", "eligible_videos": 1587, "min_stratum_retention": 0.86,
           "substituted_slot_fraction": 0.005, "max_missing_run_retained": 2, "max_missing_time_span_sec": 0.4}])
    _csv(c1 / "behavior_policy_candidate_impact.csv", ["candidate", "eligible_videos"],
         [{"candidate": "B1_COVERAGE_90", "eligible_videos": 1604},
          {"candidate": "B2_COVERAGE_95", "eligible_videos": 1520}])
    edges = [{"video_id": state.video_id, "split": state.split, "label": state.label,
              "context_missing_count": state.context_missing_count,
              "longest_context_missing_run": state.longest_context_missing_run,
              "C3_eligible": freeze.context_exclusion_reason(state, freeze.EXPECTED_CONTEXT) == "NONE"}
             for state in states]
    _csv(c1 / "context_policy_edge_cases.csv", list(edges[0]), edges)
    _csv(c1 / "branch_eligibility_cross_summary.csv", ["videos"], [{"videos": 1763}])
    manual = audit / "step4b_visual_review"
    manual.mkdir()
    (manual / "step4b_manual_review_summary.json").write_text(json.dumps({"manual_review_complete": True, "reviewed_candidates": 36}), encoding="utf-8")
    return canonical, audit


def test_prepare_and_write_freeze_preserves_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    states = _population()
    canonical, audit = _inputs(tmp_path, states)
    monkeypatch.setattr(freeze, "load_inputs", lambda unused_canonical, unused_audit: states)
    c1_path = audit / "step4c_policy_analysis/step4c_policy_analysis_summary.json"
    old = c1_path.read_bytes()
    prepared = freeze.prepare_freeze(_config(), canonical, audit)
    assert len(prepared["manifest_rows"]) == 1763
    assert prepared["summary"]["context_eligible"] == 1677
    assert prepared["summary"]["behavior_eligible"] == 1520
    n246 = next(row for row in prepared["manifest_rows"] if row["video_id"] == "n_246")
    assert n246["context_exclusion_reason"] == "NO_VALID_CONTEXT"
    assert n246["behavior_exclusion_reason"] == "NO_VALID_BEHAVIOR"
    output = tmp_path / "frozen"
    summary = freeze.write_freeze(prepared, output)
    assert len(list(output.iterdir())) == 5
    assert summary["step4_fully_closed"] is True
    assert c1_path.read_bytes() == old
    with (output / "sequence_missing_policy_eligibility.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1763
    assert (output / "sequence_missing_policy_hash.txt").read_text(encoding="ascii").strip() == summary["sequence_missing_policy_hash"]
    with pytest.raises(FileExistsError):
        freeze.write_freeze(prepared, output)


def test_count_mismatch_blocks_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    states = _population()
    canonical, audit = _inputs(tmp_path, states)
    monkeypatch.setattr(freeze, "load_inputs", lambda unused_canonical, unused_audit: states)
    path = audit / "step4c_policy_analysis/step4c_policy_analysis_summary.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values["behavior_candidates"][0]["eligible_videos"] = 1521
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="STEP_4C2_FREEZE_BLOCKED_COUNT_MISMATCH"):
        freeze.prepare_freeze(_config(), canonical, audit)
