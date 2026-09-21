"""STEP 4-C1 후보 자격·대체 시뮬레이션·집단 영향·test 보호를 검증한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from drowsiness_detection.evaluation_v2 import missing_policy_analysis as policy


def _state(video_id: str = "v0", missing: tuple[int, ...] = (), valid_behavior: int = 100,
           longest_behavior_run: int = 0, split: str = "train", label: str = "drowsy") -> policy.VideoState:
    available = tuple(index not in missing for index in range(32))
    return policy.VideoState(video_id, split, label, available, tuple(range(32)),
                             tuple(index * 0.3 for index in range(32)), valid_behavior,
                             100 - valid_behavior, longest_behavior_run)


def _population() -> list[policy.VideoState]:
    groups = (("train", "drowsy", 683), ("train", "not_drowsy", 769),
              ("val", "drowsy", 146), ("val", "not_drowsy", 165))
    states = []
    index = 0
    for split, label, count in groups:
        for _ in range(count):
            states.append(_state(f"v{index}", split=split, label=label))
            index += 1
    states[0] = _state("v0", missing=(5,), valid_behavior=99, longest_behavior_run=1)
    states[1] = _state("v1", missing=(5, 6), valid_behavior=98, longest_behavior_run=2)
    states[683] = _state("n_246", missing=tuple(range(32)), valid_behavior=0,
                         longest_behavior_run=100, split="train", label="not_drowsy")
    return states


def test_context_runs_isolated_consecutive_and_timestamp_span() -> None:
    isolated = _state(missing=(5, 10))
    consecutive = _state(missing=(5, 6, 7))
    assert isolated.context_runs == ((5, 5), (10, 10))
    assert isolated.longest_context_missing_run == 1
    assert isolated.context_run_spans_sec == (0.0, 0.0)
    assert consecutive.context_runs == ((5, 7),)
    assert consecutive.longest_context_missing_run == 3
    assert consecutive.context_run_spans_sec[0] == pytest.approx(0.6)


def test_context_c0_to_c4_and_all_missing() -> None:
    clean = _state()
    isolated = _state(missing=(5, 10))
    double = _state(missing=(5, 6))
    long = _state(missing=(5, 6, 7, 8, 9))
    assert [policy.context_eligible(clean, c) for c in policy.CONTEXT_CANDIDATES] == [True] * 5
    assert [policy.context_eligible(isolated, c) for c in policy.CONTEXT_CANDIDATES] == [False, True, True, True, True]
    assert [policy.context_eligible(double, c) for c in policy.CONTEXT_CANDIDATES] == [False, False, True, True, True]
    assert [policy.context_eligible(long, c) for c in policy.CONTEXT_CANDIDATES] == [False, False, False, False, True]
    assert [policy.context_eligible(_state(missing=tuple(range(32))), c) for c in policy.CONTEXT_CANDIDATES] == [False] * 5


def test_nearest_valid_tie_boundary_and_no_source() -> None:
    valid = (True, False, False, True)
    assert policy.nearest_valid_slot(valid, 1) == 0
    assert policy.nearest_valid_slot(valid, 2) == 3
    assert policy.nearest_valid_slot((False, True, True), 0) == 1
    assert policy.nearest_valid_slot((True, True, False), 2) == 1
    assert policy.nearest_valid_slot((False, False), 0) is None
    assert policy.nearest_valid_slot((True, False, True), 1) == 0


def test_behavior_b0_to_b3_mask_only() -> None:
    clean = _state()
    short = _state(valid_behavior=94, longest_behavior_run=2)
    long = _state(valid_behavior=80, longest_behavior_run=21)
    empty = _state(valid_behavior=0, longest_behavior_run=100)
    assert [policy.behavior_eligible(clean, c) for c in policy.BEHAVIOR_CANDIDATES] == [True] * 4
    assert [policy.behavior_eligible(short, c) for c in policy.BEHAVIOR_CANDIDATES] == [True, True, False, True]
    assert [policy.behavior_eligible(long, c) for c in policy.BEHAVIOR_CANDIDATES] == [True, False, False, False]
    assert [policy.behavior_eligible(empty, c) for c in policy.BEHAVIOR_CANDIDATES] == [False] * 4


def test_analysis_strata_distribution_cross_and_n246(tmp_path: Path) -> None:
    states = _population()
    before = tuple(states)
    result = policy.analyze(states)
    assert tuple(states) == before
    context = {row["candidate"]: row for row in result["context_rows"]}
    behavior = {row["candidate"]: row for row in result["behavior_rows"]}
    assert [context[name]["eligible_videos"] for name, _, _ in policy.CONTEXT_CANDIDATES] == [1760, 1761, 1762, 1762, 1762]
    assert context["C2_SHORT_GAP"]["total_substituted_slots"] == 3
    assert context["C2_SHORT_GAP"]["max_missing_run_retained"] == 2
    assert context["C2_SHORT_GAP"]["max_missing_time_span_sec"] == pytest.approx(0.3)
    assert context["C0_COMPLETE_ONLY"]["substituted_slot_fraction"] == 0
    assert behavior["B0_MASK_ONLY"]["eligible_videos"] == 1762
    assert behavior["B2_COVERAGE_95"]["eligible_videos"] == 1762
    assert len(result["stratum_rows"]) == 9 * 9
    assert len(result["cross_rows"]) == 4 * 3 * 4
    assert all(sum(row["videos"] for row in result["cross_rows"][index:index + 4]) == 1763
               for index in range(0, len(result["cross_rows"]), 4))
    edge = next(row for row in result["edge_rows"] if row["video_id"] == "n_246")
    assert all(edge[f"{name[:2]}_eligible"] is False for name, _, _ in policy.CONTEXT_CANDIDATES)
    assert edge["reason_excluded_C4"] == "NO_VALID_CONTEXT_SOURCE"
    assert result["summary"]["policy_selected"] is False
    output = tmp_path / "analysis"
    policy.write_analysis(result, output)
    assert len(list((output / "plots").glob("*.png"))) == 4
    assert (output / "context_policy_candidate_impact.csv").is_file()
    assert (output / "step4c_policy_analysis_report.txt").is_file()
    assert not (output / "sequence_missing_policy_hash").exists()
    with pytest.raises(FileExistsError):
        policy.write_analysis(result, output)


def test_no_label_dependent_threshold_and_shift() -> None:
    drowsy = _state("d", missing=(4,), split="train", label="drowsy")
    alert = _state("n", missing=(4,), split="val", label="not_drowsy")
    assert [policy.context_eligible(drowsy, c) for c in policy.CONTEXT_CANDIDATES] == [policy.context_eligible(alert, c) for c in policy.CONTEXT_CANDIDATES]
    assert [policy.behavior_eligible(drowsy, c) for c in policy.BEHAVIOR_CANDIDATES] == [policy.behavior_eligible(alert, c) for c in policy.BEHAVIOR_CANDIDATES]
    assert policy._shift([drowsy, alert], [drowsy]) == (50.0, 50.0)


def _csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_input_test_split_rejected_before_frames_read(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    audit = tmp_path / "audit"
    (canonical / "metadata").mkdir(parents=True)
    (canonical / "metadata/run_metadata.json").write_text(json.dumps({"policy_hash": policy.POLICY_HASH}), encoding="utf-8")
    videos = [{"video_id": f"v{i}", "split": "test" if i == 0 else "train", "label": "drowsy",
               "policy_hash": policy.POLICY_HASH} for i in range(1763)]
    quality = [dict(row) for row in videos]
    _csv(canonical / "metadata/canonical_videos.csv", list(videos[0]), videos)
    _csv(audit / "per_video_quality.csv", list(quality[0]), quality)
    _csv(audit / "missing_runs.csv", ["video_id", "split"], [])
    _csv(audit / "split_label_quality_summary.csv", ["group_name"], [])
    _csv(audit / "context_missing_cumulative.csv", ["max_missing_allowed"], [{"max_missing_allowed": "0"}])
    _csv(audit / "behavior_missing_cumulative.csv", ["max_missing_allowed"], [{"max_missing_allowed": "0"}])
    (audit / "quality_missing_summary.json").write_text(json.dumps({"policy_hash": policy.POLICY_HASH, "test_rows": 0}), encoding="utf-8")
    (audit / "step4b_visual_review").mkdir()
    (audit / "step4b_visual_review/step4b_manual_review_summary.json").write_text(json.dumps({"manual_review_complete": True, "reviewed_candidates": 36, "test_used": False, "missing_policy_selected": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="STEP_4C1_BLOCKED_TEST_LEAKAGE"):
        policy.load_inputs(canonical, audit)
