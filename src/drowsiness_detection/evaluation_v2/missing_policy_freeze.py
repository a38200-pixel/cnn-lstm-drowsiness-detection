"""STEP 4-C2 C3/B2 sequence 결측 정책을 동결하고 독립 적격성을 검증한다."""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from drowsiness_detection.preprocessing_v2.preprocessing_provenance import file_sha256, stable_hash

from .missing_policy_analysis import POLICY_HASH, VideoState, load_inputs


CONTEXT_POLICY_ID = "CONTEXT_C3_SHORT_GAP"
BEHAVIOR_POLICY_ID = "BEHAVIOR_B2_COVERAGE_95"
EXPECTED_CONTEXT = {
    "policy_id": CONTEXT_POLICY_ID, "total_slots": 32, "max_missing_count": 8,
    "max_consecutive_missing": 4, "handling": "nearest_valid_context_slot_substitution",
    "tie_break": "earlier", "source_scope": "same_video_existing_valid_context_slot",
    "preserve_original_valid_mask": True, "preserve_imputed_mask": True,
    "required_slot_provenance": [
        "context_index", "target_canonical_index", "original_available", "used_available",
        "imputed", "source_context_index", "source_canonical_index",
        "imputation_distance_context_slots", "imputation_distance_canonical_slots",
        "imputation_distance_sec",
    ],
    "required_sequence_masks": ["context_original_valid_mask", "context_imputed_mask"],
}
EXPECTED_BEHAVIOR = {
    "policy_id": BEHAVIOR_POLICY_ID, "total_slots": 100, "min_valid_count": 95,
    "max_missing_count": 5, "max_consecutive_missing": 5, "handling": "masked_nan",
    "interpolation": "none", "preserve_valid_mask": True,
}
EXPECTED_GLOBAL = {
    "branch_eligibility_independent": True, "label_dependent_policy": False,
    "test_allowed": False,
}
MANIFEST_COLUMNS = (
    "video_id", "split", "label", "context_missing_count", "longest_context_missing_run",
    "context_eligible", "context_exclusion_reason", "behavior_valid_count",
    "behavior_missing_count", "longest_yunet_missing_run", "behavior_eligible",
    "behavior_exclusion_reason", "canonical_policy_hash", "sequence_missing_policy_hash",
)


def load_config(path: Path) -> dict[str, Any]:
    """선택된 C3/B2 의미와 branch/test 보호 설정을 정확히 검증한다."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("sequence missing config는 mapping이어야 합니다")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """다른 threshold나 숨은 label별 분기를 동결하지 못하게 한다."""
    if config.get("version") != 1 or config.get("context") != EXPECTED_CONTEXT:
        raise ValueError("선택된 Context C3 정책 또는 version과 다릅니다")
    if config.get("behavior") != EXPECTED_BEHAVIOR:
        raise ValueError("선택된 Behavior B2 정책과 다릅니다")
    if config.get("global") != EXPECTED_GLOBAL:
        raise ValueError("branch 독립·label 독립·test 봉인 규칙과 다릅니다")
    if set(config) - {"version", "context", "behavior", "global", "metadata"}:
        raise ValueError("알 수 없는 최상위 config key가 있습니다")


def semantic_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """시각·경로·기계 정보를 빼고 정책의 의미만 직렬화한다."""
    return {"version": config["version"], "context": config["context"],
            "behavior": config["behavior"], "global": config["global"]}


def sequence_missing_policy_hash(config: Mapping[str, Any]) -> str:
    """기존 canonical helper와 같은 sorted canonical JSON SHA-256이다."""
    return stable_hash(semantic_payload(config))


def context_exclusion_reason(state: VideoState, config: Mapping[str, Any]) -> str:
    """원본 Context 결측 count와 32-slot run으로만 제외 이유를 계산한다."""
    if not any(state.context_available):
        return "NO_VALID_CONTEXT"
    excessive_count = state.context_missing_count > config["max_missing_count"]
    excessive_run = state.longest_context_missing_run > config["max_consecutive_missing"]
    if excessive_count and excessive_run:
        return "BOTH_CONTEXT_LIMITS_EXCEEDED"
    if excessive_count:
        return "TOO_MANY_CONTEXT_MISSING"
    if excessive_run:
        return "CONTEXT_RUN_TOO_LONG"
    return "NONE"


def behavior_exclusion_reason(state: VideoState, config: Mapping[str, Any]) -> str:
    """Behavior 유효 관측 수와 YuNet 연속 결측만 사용한다."""
    if state.behavior_valid_count == 0:
        return "NO_VALID_BEHAVIOR"
    insufficient = (state.behavior_valid_count < config["min_valid_count"]
                    or state.behavior_missing_count > config["max_missing_count"])
    excessive_run = state.longest_behavior_missing_run > config["max_consecutive_missing"]
    if insufficient and excessive_run:
        return "BOTH_BEHAVIOR_LIMITS_EXCEEDED"
    if insufficient:
        return "INSUFFICIENT_VALID_BEHAVIOR"
    if excessive_run:
        return "BEHAVIOR_RUN_TOO_LONG"
    return "NONE"


def manifest_row(state: VideoState, config: Mapping[str, Any], policy_hash: str) -> dict[str, Any]:
    """Label은 복사만 하며 Context·Behavior 적격성을 독립적으로 계산한다."""
    if state.split not in {"train", "val"}:
        raise ValueError("STEP_4C2_BLOCKED_TEST_LEAKAGE: eligibility 대상 test/unknown split")
    context_reason = context_exclusion_reason(state, config["context"])
    behavior_reason = behavior_exclusion_reason(state, config["behavior"])
    return {
        "video_id": state.video_id, "split": state.split, "label": state.label,
        "context_missing_count": state.context_missing_count,
        "longest_context_missing_run": state.longest_context_missing_run,
        "context_eligible": context_reason == "NONE", "context_exclusion_reason": context_reason,
        "behavior_valid_count": state.behavior_valid_count,
        "behavior_missing_count": state.behavior_missing_count,
        "longest_yunet_missing_run": state.longest_behavior_missing_run,
        "behavior_eligible": behavior_reason == "NONE", "behavior_exclusion_reason": behavior_reason,
        "canonical_policy_hash": POLICY_HASH, "sequence_missing_policy_hash": policy_hash,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def prepare_freeze(config: Mapping[str, Any], canonical_root: Path,
                   audit_root: Path) -> dict[str, Any]:
    """STEP 4-C1 실제 결과와 새 C3/B2 적격성을 영상별·총량별 대조한다."""
    validate_config(config)
    states = load_inputs(canonical_root, audit_root)
    c1_root = audit_root / "step4c_policy_analysis"
    c1_summary_path = c1_root / "step4c_policy_analysis_summary.json"
    c1_summary = json.loads(c1_summary_path.read_text(encoding="utf-8"))
    if c1_summary.get("input_videos") != 1763 or c1_summary.get("test_rows") != 0:
        raise ValueError("STEP_4C2_BLOCKED_TEST_LEAKAGE: C1 summary")
    if c1_summary.get("canonical_policy_hash") != POLICY_HASH or c1_summary.get("policy_selected"):
        raise ValueError("STEP 4-C1 정책·policy hash 보호 상태 불일치")
    c1_context = {row["candidate"]: row for row in _read_csv(c1_root / "context_policy_candidate_impact.csv")}
    c1_behavior = {row["candidate"]: row for row in _read_csv(c1_root / "behavior_policy_candidate_impact.csv")}
    c1_edge = _read_csv(c1_root / "context_policy_edge_cases.csv")
    c1_edge_map = {row["video_id"]: row for row in c1_edge}
    if len(c1_edge_map) != 1763 or len(c1_edge) != 1763:
        raise ValueError("STEP 4-C1 영상별 edge case 수/ID 불일치")
    if "C3_MODERATE_GAP" not in c1_context or "B2_COVERAGE_95" not in c1_behavior:
        raise ValueError("STEP 4-C1 선택 후보 결과가 없습니다")
    policy_hash = sequence_missing_policy_hash(config)
    rows = [manifest_row(state, config, policy_hash) for state in states]
    if len(rows) != 1763 or len({row["video_id"] for row in rows}) != 1763:
        raise ValueError("eligibility manifest는 영상별 고유 1763행이어야 합니다")
    for row in rows:
        old = c1_edge_map.get(row["video_id"])
        if old is None or old["split"] != row["split"] or old["label"] != row["label"]:
            raise ValueError(f"STEP 4-C1 edge/video 불일치: {row['video_id']}")
        if (int(old["context_missing_count"]) != row["context_missing_count"]
                or int(old["longest_context_missing_run"]) != row["longest_context_missing_run"]
                or (old["C3_eligible"].casefold() == "true") != row["context_eligible"]):
            raise ValueError(f"STEP_4C2_FREEZE_BLOCKED_COUNT_MISMATCH: {row['video_id']}")
    context_count = sum(row["context_eligible"] for row in rows)
    behavior_count = sum(row["behavior_eligible"] for row in rows)
    expected_context = int(c1_context["C3_MODERATE_GAP"]["eligible_videos"])
    expected_behavior = int(c1_behavior["B2_COVERAGE_95"]["eligible_videos"])
    summary_context = next(row for row in c1_summary["context_candidates"] if row["candidate"] == "C3_MODERATE_GAP")
    summary_behavior = next(row for row in c1_summary["behavior_candidates"] if row["candidate"] == "B2_COVERAGE_95")
    if (context_count != expected_context or context_count != int(summary_context["eligible_videos"])
            or context_count != 1677 or behavior_count != expected_behavior
            or behavior_count != int(summary_behavior["eligible_videos"]) or behavior_count != 1520):
        raise ValueError("STEP_4C2_FREEZE_BLOCKED_COUNT_MISMATCH: C3/B2 총량")
    n246 = next((row for row in rows if row["video_id"] == "n_246"), None)
    if (n246 is None or n246["context_missing_count"] != 32 or n246["behavior_missing_count"] != 100
            or n246["context_exclusion_reason"] != "NO_VALID_CONTEXT"
            or n246["behavior_exclusion_reason"] != "NO_VALID_BEHAVIOR"):
        raise ValueError("n_246 두 branch 결측 sanity check 불일치")
    branch_states = Counter((row["context_eligible"], row["behavior_eligible"]) for row in rows)
    summary = {
        "status": "STEP_4C2_MISSING_POLICY_SELECTION_AND_FREEZE_COMPLETE",
        "input_videos": 1763, "train_videos": 1452, "val_videos": 311, "test_rows": 0,
        "context_policy_id": CONTEXT_POLICY_ID, "behavior_policy_id": BEHAVIOR_POLICY_ID,
        "context_eligible": context_count, "context_ineligible": 1763 - context_count,
        "behavior_eligible": behavior_count, "behavior_ineligible": 1763 - behavior_count,
        "context_retention": context_count / 1763, "behavior_retention": behavior_count / 1763,
        "branch_eligibility_independent": True,
        "branch_eligibility_diagnostic": [
            {"context_eligible": context_ok, "behavior_eligible": behavior_ok,
             "videos": branch_states[(context_ok, behavior_ok)]}
            for context_ok in (True, False) for behavior_ok in (True, False)
        ],
        "n_246_context_eligible": False, "n_246_behavior_eligible": False,
        "canonical_policy_hash": POLICY_HASH,
        "sequence_missing_policy_hash": policy_hash,
        "step4c1_context_mismatch": 0, "step4c1_behavior_mismatch": 0,
        "canonical_artifacts_modified": False, "actual_sequences_created": False,
        "context_imputation_applied": False, "behavior_interpolation_applied": False,
        "raw_video_opened": False, "test_sealed": True,
        "step4_fully_closed": True, "step5_started": False,
    }
    manual_summary_path = audit_root / "step4b_visual_review/step4b_manual_review_summary.json"
    sources = {
        "step4c1_summary": "step4c_policy_analysis/step4c_policy_analysis_summary.json",
        "step4c1_context_impact": "step4c_policy_analysis/context_policy_candidate_impact.csv",
        "step4c1_behavior_impact": "step4c_policy_analysis/behavior_policy_candidate_impact.csv",
        "step4c1_edge_cases": "step4c_policy_analysis/context_policy_edge_cases.csv",
        "step4c1_cross_summary": "step4c_policy_analysis/branch_eligibility_cross_summary.csv",
    }
    frozen = {
        "policy_version": config["version"],
        "context_policy": config["context"], "behavior_policy": config["behavior"],
        "global_rules": config["global"],
        "source_step4c1_artifacts": sources,
        "source_step4c1_summary_sha256": file_sha256(c1_summary_path),
        "source_step4b_manual_review": "step4b_visual_review/step4b_manual_review_summary.json",
        "source_step4b_summary_sha256": file_sha256(manual_summary_path),
        "canonical_policy_hash": POLICY_HASH,
        "sequence_missing_policy_hash": policy_hash,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "test_sealed": True,
    }
    return {"manifest_rows": rows, "summary": summary, "frozen": frozen,
            "c1_context": c1_context, "c1_behavior": c1_behavior}


def _report(prepared: Mapping[str, Any]) -> str:
    summary = prepared["summary"]
    c3 = prepared["c1_context"]["C3_MODERATE_GAP"]
    c4 = prepared["c1_context"]["C4_COVERAGE_ONLY"]
    c2 = prepared["c1_context"]["C2_SHORT_GAP"]
    b1 = prepared["c1_behavior"]["B1_COVERAGE_90"]
    b2 = prepared["c1_behavior"]["B2_COVERAGE_95"]
    return "\n".join([
        "STEP 4-C2 Sequence Missing Policy Selection & Freeze", "",
        "[Scope]", "사용자가 지정한 Context C3·Behavior B2를 STEP 4-C1 train/val 1763개 metadata와 대조해 동결했다. Test는 열지 않았고 sequence/JPEG를 만들지 않았다.",
        "", "[Frozen Context Policy]", "CONTEXT_C3_SHORT_GAP: 원본 32-slot Context missing<=8 AND 최장 연속 missing<=4. 적격 1677, 부적격 86, 보존율 95.12%.",
        "", "[Context Handling Semantics]", "STEP 5에서만 같은 영상의 가장 가까운 기존 valid Context slot을 복제한다. 동거리면 이전 slot 우선. 이미지 재생성·interpolation·타 영상 복제는 금지한다. 적격성은 원본 결측 pattern으로 판단한다.",
        "", "[Required Context Provenance]", "context_index, target_canonical_index, original_available, used_available, imputed, source_context_index, source_canonical_index, imputation_distance_context_slots, imputation_distance_canonical_slots, imputation_distance_sec; sequence-level original/imputed masks. 이번에는 mapping을 생성하지 않았다.",
        "", "[Why C3]", f"STEP 4-C1: 최저 split×label 보존율 {float(c3['min_stratum_retention']):.2%}, 가상 대체 slot 비율 {float(c3['substituted_slot_fraction']):.3%}, 최대 유지 run {c3['max_missing_run_retained']} slot, 최대 실제 timestamp span {float(c3['max_missing_time_span_sec']):.3f}s. C2 적격 {c2['eligible_videos']}보다 영상 보존량이 높으면서 run guard가 있다.",
        "", "[Why Not C4]", f"C4는 C3보다 {int(c4['eligible_videos']) - int(c3['eligible_videos'])}개 더 유지하지만 run guard가 없어 최대 유지 run {c4['max_missing_run_retained']} slot, timestamp span {float(c4['max_missing_time_span_sec']):.3f}s까지 허용한다. 목적 선정 36개 수동 검토에서도 긴 결측은 다양한 pose/가시성 상태와 연관됐다. C0/C1/C2는 각각 더 많은 영상을 제외한다.",
        "", "[Frozen Behavior Policy]", "BEHAVIOR_B2_COVERAGE_95: 원본 100-slot behavior_valid>=95 AND 최장 YuNet missing run<=5. 적격 1520, 부적격 243, 보존율 86.22%.",
        "", "[Behavior Handling Semantics]", "EAR/MAR/pitch/yaw/roll의 결측은 NaN으로 유지하고 behavior_valid_mask를 보존한다. nearest/forward/backward/linear/spline fill과 zero-as-measurement 금지. 실제 mask sequence는 STEP 5에서 생성한다.",
        "", "[Why B2]", f"B1은 적격 {b1['eligible_videos']}개이나 최대 10-frame gap을 허용한다. B2는 {b2['eligible_videos']}개를 유지하면서 최장 5-frame gap·최소 95 valid observation을 요구한다. B0의 최소 1 valid frame과 B3의 최대 20-frame gap보다 시간 이벤트 분석의 미관측 범위를 줄인다.",
        "", "[Branch Independence]", "Context와 Behavior 적격 flag를 따로 보존한다. 영상 전체 삭제·global_video_eligible·label별 threshold·relabel은 수행하지 않는다.",
        "", "[Validation]", f"STEP 4-C1 C3/B2와 영상별·총량 불일치 0. n_246은 Context 32/32, Behavior 100/100 결측으로 두 branch 모두 부적격이나 원본 영상은 그대로 보존한다. Canonical policy hash {POLICY_HASH}. Sequence missing policy hash {summary['sequence_missing_policy_hash']}.",
        "", "[Limitations]", "C3/B2는 data-quality·보존율·집단 구성 변화·대체 부담·시간 연속성·수동 정성 근거에 따른 선택이다. CNN-LSTM 정확도 최적 정책임을 보인 실험은 아니다. 최종 model-specific 학습 표본 수도 아직 결정하지 않았다.",
        "", "[Decision]", "STEP 4-C2: COMPLETE", "MISSING POLICY: FROZEN", "STEP 4: FULLY CLOSED", "STEP 5: NOT STARTED", "TEST SPLIT: SEALED", "",
    ])


def write_freeze(prepared: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """기존 artifact를 덮어쓰지 않고 동결 정책·manifest만 새로 발행한다."""
    if output_dir.exists():
        raise FileExistsError(f"기존 STEP 4-C2 freeze를 덮어쓰지 않습니다: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step4c2_staging_", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        (stage / "sequence_missing_policy_frozen.json").write_text(json.dumps(prepared["frozen"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "sequence_missing_policy_summary.json").write_text(json.dumps(prepared["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "sequence_missing_policy_report.txt").write_text(_report(prepared), encoding="utf-8")
        (stage / "sequence_missing_policy_hash.txt").write_text(prepared["summary"]["sequence_missing_policy_hash"] + "\n", encoding="ascii")
        _write_csv(stage / "sequence_missing_policy_eligibility.csv", MANIFEST_COLUMNS, prepared["manifest_rows"])
        stage.rename(output_dir)
    return prepared["summary"]


def dry_run(prepared: Mapping[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Config·hash·예상 적격 수만 반환하고 freeze 파일을 쓰지 않는다."""
    summary = prepared["summary"]
    return {
        "mode": "DRY_RUN", "config_path": str(config_path),
        "sequence_missing_policy_hash": summary["sequence_missing_policy_hash"],
        "expected_context_eligible": summary["context_eligible"],
        "expected_behavior_eligible": summary["behavior_eligible"],
        "input_videos": summary["input_videos"], "test_rows": 0,
        "planned_output_dir": str(output_dir),
        "planned_files": ["sequence_missing_policy_frozen.json", "sequence_missing_policy_summary.json",
                          "sequence_missing_policy_report.txt", "sequence_missing_policy_hash.txt",
                          "sequence_missing_policy_eligibility.csv"],
        "output_written": False, "raw_video_opened": False, "actual_sequences_created": False,
    }
