"""STEP 5-C Context·Behavior sequence의 관계와 무결성을 읽기 전용으로 감사한다."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drowsiness_detection.preprocessing_v2.preprocessing_provenance import stable_hash
from drowsiness_detection.sequences_v2.behavior_sequence_builder import (
    FEATURE_NAMES, FEATURE_SCHEMA_HASH, BehaviorPlan,
    load_plan as load_behavior_plan, validate_bundle as validate_behavior_bundle,
)
from drowsiness_detection.sequences_v2.context_sequence_builder import (
    POLICY_HASH, SEQUENCE_POLICY_HASH, VideoPlan,
    load_plan as load_context_plan, validate_bundle as validate_context_bundle,
)


STATUS_COLUMNS = (
    "video_id", "split", "label", "context_eligible_frozen", "behavior_eligible_frozen",
    "context_bundle_exists", "behavior_bundle_exists", "context_sequence_rows",
    "behavior_sequence_rows", "context_status", "behavior_status", "cross_branch_status",
    "canonical_policy_hash", "sequence_missing_policy_hash",
)
MISMATCH_COLUMNS = ("video_id", "branch", "frozen_eligible", "bundle_exists", "detail")
METADATA_MISMATCH_COLUMNS = ("video_id", "artifact", "field", "expected", "actual")
ANOMALY_COLUMNS = ("video_id", "branch", "anomaly_type", "detail")
SUMMARY_COLUMNS = ("scope", "group_name", "status", "video_count", "fraction")
CONTEXT_ONLY_COLUMNS = (
    "video_id", "split", "label", "context_imputed_count", "context_missing_original_count",
    "behavior_valid_count", "behavior_missing_count", "longest_behavior_missing_run",
    "behavior_exclusion_reason",
)
NEITHER_COLUMNS = (
    "video_id", "split", "label", "context_exclusion_reason", "behavior_exclusion_reason",
    "context_missing_count", "context_longest_run", "behavior_valid_count",
    "behavior_longest_run",
)
EXPECTED = {"input_videos": 1763, "context_bundles": 1677, "behavior_bundles": 1520,
            "both": 1520, "context_only": 157, "behavior_only": 0, "neither": 86,
            "context_rows": 53664, "behavior_rows": 152000,
            "context_imputed_rows": 699, "behavior_valid_rows": 151655,
            "behavior_missing_rows": 345}


def _yes(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def classify_branch(context_exists: bool, behavior_exists: bool) -> str:
    """두 bundle의 실제 존재 여부만으로 네 branch 상태를 분류한다."""
    if context_exists and behavior_exists:
        return "BOTH"
    if context_exists:
        return "CONTEXT_ONLY"
    if behavior_exists:
        return "BEHAVIOR_ONLY"
    return "NEITHER"


def validate_context_pattern(plans: Sequence[VideoPlan]) -> tuple[int, ...]:
    """모든 Context 영상의 32개 target canonical index pattern이 같은지 확인한다."""
    patterns = {tuple(int(row["target_canonical_index"]) for row in plan.rows) for plan in plans}
    if len(patterns) != 1:
        raise ValueError("STEP_5C_REVIEW_REQUIRED: Context target pattern mismatch")
    pattern = next(iter(patterns))
    if (len(pattern) != 32 or len(set(pattern)) != 32
            or not all(0 <= value <= 99 for value in pattern)
            or any(left >= right for left, right in zip(pattern, pattern[1:]))):
        raise ValueError("STEP_5C_REVIEW_REQUIRED: Context target pattern invalid")
    return pattern


def behavior_semantic_anomalies(plan: BehaviorPlan) -> dict[str, int]:
    """Pose와 별개인 B2 EAR/MAR mask, feature-valid mask, infinity를 검사한다."""
    false_finite = sum(
        (not bool(plan.behavior_valid_mask[index]))
        and bool(np.isfinite(plan.features[index, :2]).all()) for index in range(100))
    true_nan = sum(
        bool(plan.behavior_valid_mask[index])
        and not bool(np.isfinite(plan.features[index, :2]).all()) for index in range(100))
    feature_mask = int(np.count_nonzero(np.isfinite(plan.features) != plan.feature_valid_mask))
    return {"false_mask_with_required_finite": false_finite,
            "required_valid_nan": true_nan,
            "feature_valid_mask_mismatch": feature_mask,
            "infinity": int(np.isinf(plan.features).sum())}


def _bundle_ids(root: Path, branch: str) -> set[str]:
    per_video = root / "per_video"
    if not per_video.is_dir():
        raise ValueError(f"STEP_5C_REVIEW_REQUIRED: {branch} per_video 없음")
    split_dirs = {path.name for path in per_video.iterdir() if path.is_dir()}
    if split_dirs - {"train", "val"}:
        raise ValueError(f"STEP_5C_BLOCKED_TEST_LEAKAGE: {branch} split directory")
    identifiers: set[str] = set()
    for split in ("train", "val"):
        directory = per_video / split
        if not directory.exists():
            continue
        for bundle in directory.iterdir():
            if not bundle.is_dir() or not (bundle / "COMPLETE.json").is_file():
                raise ValueError(f"STEP_5C_REVIEW_REQUIRED: incomplete {branch} bundle {bundle}")
            if bundle.name in identifiers:
                raise ValueError(f"STEP_5C_REVIEW_REQUIRED: duplicate {branch} bundle {bundle.name}")
            identifiers.add(bundle.name)
    return identifiers


def _tree_fingerprint(root: Path) -> str:
    """내용을 쓰지 않고 상대경로·크기·mtime으로 입력 tree 불변성을 추적한다."""
    entries = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()),
                       key=lambda item: item.relative_to(root).as_posix()):
        stat = path.stat()
        entries.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return stable_hash(entries)


def _validate_hash_document(path: Path, branch: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("canonical_policy_hash") != POLICY_HASH
            or data.get("sequence_missing_policy_hash") != SEQUENCE_POLICY_HASH
            or data.get("test_rows") != 0):
        raise ValueError(f"STEP_5C_REVIEW_REQUIRED: {branch} summary hash/test mismatch")


def prepare_dry_run(canonical_root: Path, freeze_root: Path, context_root: Path,
                    behavior_root: Path, output_root: Path) -> dict[str, Any]:
    """Bundle set·frozen relation·hash만 확인하며 bundle 내용은 열지 않는다."""
    canonical = _read_csv(canonical_root / "metadata/canonical_videos.csv")
    frozen = _read_csv(freeze_root / "sequence_missing_policy_eligibility.csv")
    if (any(row["split"] not in {"train", "val"} for row in canonical + frozen)
            or len(canonical) != EXPECTED["input_videos"] or len(frozen) != EXPECTED["input_videos"]):
        raise ValueError("STEP_5C_BLOCKED_TEST_LEAKAGE: canonical/frozen universe")
    context_ids = _bundle_ids(context_root, "Context")
    behavior_ids = _bundle_ids(behavior_root, "Behavior")
    universe = {row["video_id"] for row in canonical}
    if context_ids - universe or behavior_ids - universe:
        raise ValueError("STEP_5C_REVIEW_REQUIRED: bundle outside canonical universe")
    counts = Counter(classify_branch(video_id in context_ids, video_id in behavior_ids)
                     for video_id in universe)
    actual = {"input_videos": len(universe), "context_bundles": len(context_ids),
              "behavior_bundles": len(behavior_ids), "both": counts["BOTH"],
              "context_only": counts["CONTEXT_ONLY"], "behavior_only": counts["BEHAVIOR_ONLY"],
              "neither": counts["NEITHER"]}
    for key, value in actual.items():
        if value != EXPECTED[key]:
            raise ValueError(f"STEP_5C_REVIEW_REQUIRED: dry-run {key}: {value} != {EXPECTED[key]}")
    return {"status": "STEP_5C_DRY_RUN_PASS", **actual,
            "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
            "test_rows": 0, "planned_output_root": output_root.as_posix(),
            "deep_audit": False, "raw_video_opened": False, "cnn_started": False}


def _assert_global_manifest(rows: Sequence[Mapping[str, str]], plans: Mapping[str, Any],
                            branch: str, slots: int) -> None:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"STEP_5C_BLOCKED_TEST_LEAKAGE: {branch} global manifest")
        grouped[row["video_id"]].append(row)
    if set(grouped) != set(plans):
        raise ValueError(f"STEP_5C_REVIEW_REQUIRED: {branch} global video set mismatch")
    index_name = "target_context_index" if branch == "Context" else "canonical_index"
    for video_id, actual in grouped.items():
        actual.sort(key=lambda row: int(row[index_name]))
        if len(actual) != slots or [int(row[index_name]) for row in actual] != list(range(slots)):
            raise ValueError(f"STEP_5C_REVIEW_REQUIRED: {branch} global slots {video_id}")
        expected_rows = plans[video_id].rows
        if actual != [{key: str(value) for key, value in row.items()} for row in expected_rows]:
            raise ValueError(f"STEP_5C_REVIEW_REQUIRED: {branch} global content {video_id}")


def _cross_summary(status_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    group_specs = [("overall", "all", list(status_rows))]
    for field, scope in (("split", "split"), ("label", "label")):
        for name in sorted({str(row[field]) for row in status_rows}):
            group_specs.append((scope, name, [row for row in status_rows if row[field] == name]))
    for split, label in sorted({(str(row["split"]), str(row["label"])) for row in status_rows}):
        group_specs.append(("split_x_label", f"{split}/{label}",
                            [row for row in status_rows if row["split"] == split and row["label"] == label]))
    for scope, name, members in group_specs:
        counts = Counter(row["cross_branch_status"] for row in members)
        for status in ("BOTH", "CONTEXT_ONLY", "BEHAVIOR_ONLY", "NEITHER"):
            output.append({"scope": scope, "group_name": name, "status": status,
                           "video_count": counts[status],
                           "fraction": counts[status] / len(members) if members else 0.0})
    return output


def run_audit(project_root: Path, canonical_root: Path, freeze_root: Path,
              context_root: Path, context_report_root: Path, behavior_root: Path,
              behavior_report_root: Path, config_path: Path,
              output_root: Path) -> dict[str, Any]:
    """모든 실제 bundle과 global manifest를 canonical/frozen 기대값과 대조한다."""
    if output_root.exists():
        raise FileExistsError(f"STEP_5C_REVIEW_REQUIRED: audit output exists {output_root}")
    input_roots = {"canonical": canonical_root, "context": context_root, "behavior": behavior_root,
                   "freeze": freeze_root, "context_report": context_report_root,
                   "behavior_report": behavior_report_root}
    before = {name: _tree_fingerprint(root) for name, root in input_roots.items()}
    _validate_hash_document(context_report_root / "context_sequence_summary.json", "Context")
    _validate_hash_document(behavior_report_root / "behavior_sequence_summary.json", "Behavior")
    canonical_rows = _read_csv(canonical_root / "metadata/canonical_videos.csv")
    frozen_rows = _read_csv(freeze_root / "sequence_missing_policy_eligibility.csv")
    if any(row["split"] not in {"train", "val"} for row in canonical_rows + frozen_rows):
        raise ValueError("STEP_5C_BLOCKED_TEST_LEAKAGE: canonical/frozen")
    canonical = {row["video_id"]: row for row in canonical_rows}
    frozen = {row["video_id"]: row for row in frozen_rows}
    if len(canonical) != EXPECTED["input_videos"] or set(canonical) != set(frozen):
        raise ValueError("STEP_5C_REVIEW_REQUIRED: canonical/frozen universe mismatch")
    context_plans_list, context_excluded = load_context_plan(
        canonical_root, freeze_root, config_path, project_root, check_images=True)
    behavior_plans_list, behavior_excluded = load_behavior_plan(canonical_root, freeze_root, config_path)
    context_plans = {plan.video_id: plan for plan in context_plans_list}
    behavior_plans = {plan.video_id: plan for plan in behavior_plans_list}
    context_ids = _bundle_ids(context_root, "Context")
    behavior_ids = _bundle_ids(behavior_root, "Behavior")
    if context_ids != set(context_plans) or behavior_ids != set(behavior_plans):
        raise ValueError("STEP_5C_REVIEW_REQUIRED: actual bundle/expected plan mismatch")
    for plan in context_plans_list:
        validate_context_bundle(context_root / "per_video" / plan.split / plan.video_id, plan)
    for plan in behavior_plans_list:
        validate_behavior_bundle(behavior_root / "per_video" / plan.split / plan.video_id, plan)
    context_global = _read_csv(context_root / "metadata/context_sequence_frames.csv")
    behavior_global = _read_csv(behavior_root / "metadata/behavior_sequence_frames.csv")
    _assert_global_manifest(context_global, context_plans, "Context", 32)
    _assert_global_manifest(behavior_global, behavior_plans, "Behavior", 100)
    frozen_context_excluded = {row["video_id"]: row for row in context_excluded}
    frozen_behavior_excluded = {row["video_id"]: row for row in behavior_excluded}
    eligibility_mismatches: list[dict[str, Any]] = []
    metadata_mismatches: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    status_rows = []
    for video_id in sorted(canonical):
        base, policy = canonical[video_id], frozen[video_id]
        context_exists, behavior_exists = video_id in context_ids, video_id in behavior_ids
        for branch, expected_eligible, exists in (
                ("Context", _yes(policy["context_eligible"]), context_exists),
                ("Behavior", _yes(policy["behavior_eligible"]), behavior_exists)):
            if expected_eligible != exists:
                eligibility_mismatches.append({"video_id": video_id, "branch": branch,
                                               "frozen_eligible": expected_eligible,
                                               "bundle_exists": exists, "detail": "frozen/bundle mismatch"})
        artifacts = [("frozen", policy)]
        if context_exists:
            artifacts.append(("context", context_plans[video_id].rows[0]))
        if behavior_exists:
            artifacts.append(("behavior", behavior_plans[video_id].rows[0]))
        for artifact, row in artifacts:
            for field in ("split", "label"):
                if row[field] != base[field]:
                    metadata_mismatches.append({"video_id": video_id, "artifact": artifact,
                                                "field": field, "expected": base[field],
                                                "actual": row[field]})
        status = classify_branch(context_exists, behavior_exists)
        status_rows.append({
            "video_id": video_id, "split": base["split"], "label": base["label"],
            "context_eligible_frozen": _yes(policy["context_eligible"]),
            "behavior_eligible_frozen": _yes(policy["behavior_eligible"]),
            "context_bundle_exists": context_exists, "behavior_bundle_exists": behavior_exists,
            "context_sequence_rows": 32 if context_exists else 0,
            "behavior_sequence_rows": 100 if behavior_exists else 0,
            "context_status": "COMPLETE" if context_exists else "EXCLUDED",
            "behavior_status": "COMPLETE" if behavior_exists else "EXCLUDED",
            "cross_branch_status": status, "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        })
    if eligibility_mismatches or metadata_mismatches:
        raise ValueError("STEP_5C_REVIEW_REQUIRED: eligibility/metadata mismatch")
    status_counts = Counter(row["cross_branch_status"] for row in status_rows)
    expected_status = {"BOTH": 1520, "CONTEXT_ONLY": 157, "BEHAVIOR_ONLY": 0, "NEITHER": 86}
    if any(status_counts[key] != value for key, value in expected_status.items()):
        raise ValueError("STEP_5C_REVIEW_REQUIRED: unexpected cross-branch relation")
    pattern = validate_context_pattern(context_plans_list)
    context_imputed = sum(sum(plan.imputed_mask) for plan in context_plans_list)
    context_original = sum(sum(plan.original_mask) for plan in context_plans_list)
    temporal_overwrite = sum(
        bool(row["imputed"]) and float(row["target_timestamp_sec"]) == float(row["source_timestamp_sec"])
        for plan in context_plans_list for row in plan.rows)
    if temporal_overwrite:
        raise ValueError("STEP_5C_REVIEW_REQUIRED: Context target timestamp overwritten")
    behavior_valid = sum(int(plan.behavior_valid_mask.sum()) for plan in behavior_plans_list)
    behavior_missing = sum(int((~plan.behavior_valid_mask).sum()) for plan in behavior_plans_list)
    behavior_checks = [behavior_semantic_anomalies(plan) for plan in behavior_plans_list]
    false_with_required_values = sum(item["false_mask_with_required_finite"] for item in behavior_checks)
    true_with_required_nan = sum(item["required_valid_nan"] for item in behavior_checks)
    feature_mask_mismatch = sum(item["feature_valid_mask_mismatch"] for item in behavior_checks)
    infinity_count = sum(item["infinity"] for item in behavior_checks)
    if false_with_required_values or true_with_required_nan or feature_mask_mismatch or infinity_count:
        raise ValueError("STEP_5C_REVIEW_REQUIRED: Behavior NaN/mask/infinity anomaly")
    context_only_rows = []
    neither_rows = []
    for row in status_rows:
        video_id, policy = row["video_id"], frozen[row["video_id"]]
        if row["cross_branch_status"] == "CONTEXT_ONLY":
            plan = context_plans[video_id]
            context_only_rows.append({
                "video_id": video_id, "split": row["split"], "label": row["label"],
                "context_imputed_count": sum(plan.imputed_mask),
                "context_missing_original_count": int(policy["context_missing_count"]),
                "behavior_valid_count": int(policy["behavior_valid_count"]),
                "behavior_missing_count": int(policy["behavior_missing_count"]),
                "longest_behavior_missing_run": int(policy["longest_yunet_missing_run"]),
                "behavior_exclusion_reason": policy["behavior_exclusion_reason"],
            })
        elif row["cross_branch_status"] == "NEITHER":
            neither_rows.append({
                "video_id": video_id, "split": row["split"], "label": row["label"],
                "context_exclusion_reason": policy["context_exclusion_reason"],
                "behavior_exclusion_reason": policy["behavior_exclusion_reason"],
                "context_missing_count": int(policy["context_missing_count"]),
                "context_longest_run": int(policy["longest_context_missing_run"]),
                "behavior_valid_count": int(policy["behavior_valid_count"]),
                "behavior_longest_run": int(policy["longest_yunet_missing_run"]),
            })
    n246 = next(row for row in status_rows if row["video_id"] == "n_246")
    n246_policy = frozen["n_246"]
    if (n246["cross_branch_status"] != "NEITHER"
            or n246_policy["context_exclusion_reason"] != "NO_VALID_CONTEXT"
            or n246_policy["behavior_exclusion_reason"] != "NO_VALID_BEHAVIOR"):
        raise ValueError("STEP_5C_REVIEW_REQUIRED: n_246 sanity")
    after = {name: _tree_fingerprint(root) for name, root in input_roots.items()}
    mutation = {name: before[name] != after[name] for name in input_roots}
    if any(mutation.values()):
        raise ValueError("STEP_5C_REVIEW_REQUIRED: input artifact mutation")
    summary = {
        "status": "STEP_5C_CROSS_BRANCH_SEQUENCE_INTEGRITY_AUDIT_COMPLETE",
        "input_videos": len(canonical), "context_bundles": len(context_ids),
        "behavior_bundles": len(behavior_ids), "both": status_counts["BOTH"],
        "context_only": status_counts["CONTEXT_ONLY"],
        "behavior_only": status_counts["BEHAVIOR_ONLY"], "neither": status_counts["NEITHER"],
        "context_rows": len(context_global), "behavior_rows": len(behavior_global),
        "context_original_rows": context_original, "context_imputed_rows": context_imputed,
        "behavior_valid_rows": behavior_valid, "behavior_missing_rows": behavior_missing,
        "eligibility_mismatch_count": 0, "metadata_mismatch_count": 0,
        "context_integrity_anomaly_count": 0, "behavior_integrity_anomaly_count": 0,
        "cross_branch_alignment_error_count": 0, "total_anomaly_count": 0,
        "context_target_pattern_consistent": True,
        "context_target_timestamp_overwrite_count": temporal_overwrite,
        "context_chained_imputation_violation_count": 0,
        "context_source_jpeg_error_count": 0,
        "behavior_valid_count_mismatch": 0, "behavior_missing_run_mismatch": 0,
        "behavior_false_mask_with_required_finite_count": false_with_required_values,
        "behavior_required_valid_nan_contradiction": true_with_required_nan,
        "behavior_feature_valid_mask_mismatch": feature_mask_mismatch,
        "behavior_infinite_value_count": infinity_count, "canonical_numeric_mismatch": 0,
        "canonical_policy_hash": POLICY_HASH,
        "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
        "feature_names": list(FEATURE_NAMES), "test_rows": 0, "test_access": 0,
        "cnn_started": False, "raw_video_opened": False,
        "canonical_artifacts_modified": mutation["canonical"],
        "context_artifacts_modified": mutation["context"] or mutation["context_report"],
        "behavior_artifacts_modified": mutation["behavior"] or mutation["behavior_report"],
        "frozen_artifacts_modified": mutation["freeze"],
        "context_target_source_provenance": "PASS",
        "behavior_nan_mask_semantics": "PASS",
        "head_pose_physical_sign": "UNRESOLVED",
    }
    for key, value in EXPECTED.items():
        if summary[key] != value:
            raise ValueError(f"STEP_5C_REVIEW_REQUIRED: final {key}: {summary[key]} != {value}")
    reference = context_plans_list[0]
    pattern_json = {
        "reference_video_id": reference.video_id,
        "pattern_consistent_across_context_videos": True,
        "timestamp_semantics": "target_timestamp_sec is video-local canonical actual timestamp",
        "slots": [{"context_index": index, "target_canonical_index": pattern[index],
                   "target_timestamp_sec": float(reference.rows[index]["target_timestamp_sec"])}
                  for index in range(32)],
    }
    report = "\n".join([
        "STEP 5-C Cross-Branch Sequence Integrity Audit", "", "[Scope]",
        "Canonical, frozen eligibility, Context 및 Behavior sequence를 읽기 전용으로 교차 검증했다.",
        "", "[Frozen Policies]", f"Canonical: {POLICY_HASH}",
        f"Sequence missing: {SEQUENCE_POLICY_HASH}", "", "[Input Artifact Counts]",
        "Canonical 1763; Context 1677/53664 rows; Behavior 1520/152000 rows.",
        "", "[Eligibility Consistency]", "Frozen eligibility와 실제 bundle mismatch 0.",
        "", "[Cross-Branch Video Sets]", "BOTH 1520; CONTEXT_ONLY 157; BEHAVIOR_ONLY 0; NEITHER 86.",
        "", "[Context Sequence Integrity]", "32-slot, mask, source JPEG, no chained imputation: PASS.",
        "", "[Behavior Sequence Integrity]", "100-slot, (100,6) float32, NaN/mask, canonical numeric lineage: PASS.",
        "", "[Target Timeline Alignment]", "Context target canonical indices are one deterministic 32-slot pattern.",
        "Context target timeline and image-source provenance are intentionally separated.",
        "", "[Context Imputation Provenance]", "Original 52965; imputed references 699; timestamp overwrite 0.",
        "", "[Behavior NaN/Mask Semantics]", "Valid 151655; missing 345; interpolation 0; anomalies 0.",
        "", "[Context-Only Videos]", "157 videos; retained only in the independent Context branch.",
        "", "[Neither-Branch Videos]", "86 videos; no sequence bundle in either branch.",
        "", "[n_246 Sanity Check]", "NEITHER; NO_VALID_CONTEXT; NO_VALID_BEHAVIOR.",
        "", "[Test Protection]", "Test rows/access 0; SEALED.",
        "", "[Artifact Immutability]", "Canonical, Context, Behavior, frozen inputs unchanged.",
        "", "[Limitations]", "Head Pose physical sign is unresolved; no head-down/head-up meaning is assigned.",
        "", "[Decision]", "STEP 5-C: COMPLETE", "STEP 5-D CNN FEATURE EXTRACTION: NOT STARTED", "",
    ])
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step5c_audit_", dir=output_root.parent) as temporary:
        stage = Path(temporary)
        (stage / "cross_branch_audit_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "cross_branch_audit_report.txt").write_text(report, encoding="utf-8")
        (stage / "context_target_pattern.json").write_text(
            json.dumps(pattern_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_csv(stage / "cross_branch_video_status.csv", STATUS_COLUMNS, status_rows)
        _write_csv(stage / "cross_branch_summary.csv", SUMMARY_COLUMNS, _cross_summary(status_rows))
        _write_csv(stage / "branch_eligibility_mismatches.csv", MISMATCH_COLUMNS, eligibility_mismatches)
        _write_csv(stage / "cross_branch_metadata_mismatches.csv", METADATA_MISMATCH_COLUMNS,
                   metadata_mismatches)
        _write_csv(stage / "context_only_videos.csv", CONTEXT_ONLY_COLUMNS, context_only_rows)
        _write_csv(stage / "neither_branch_videos.csv", NEITHER_COLUMNS, neither_rows)
        _write_csv(stage / "sequence_integrity_anomalies.csv", ANOMALY_COLUMNS, anomalies)
        stage.rename(output_root)
    return summary
