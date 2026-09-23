"""STEP 6-E3 frozen Context feature의 temporal variation을 read-only로 감사한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drowsiness_detection.evaluation_v2.context_feature_distribution_audit import (
    BACKBONES,
    EXPECTED_COUNTS,
    EXPECTED_SHAPE,
    LABELS,
    SPLITS,
    _file_fingerprint,
    _read_index,
    _verify_unchanged,
    distribution_summary,
)


COSINE_THRESHOLDS = (0.95, 0.98, 0.99, 0.995)
TRANSITION_TYPES = (
    "original_to_original",
    "original_to_imputed",
    "imputed_to_original",
    "imputed_to_imputed",
)
VIDEO_METRICS = (
    "mean_temporal_l2_delta",
    "median_temporal_l2_delta",
    "max_temporal_l2_delta",
    "mean_adjacent_cosine",
    "median_adjacent_cosine",
    "min_adjacent_cosine",
    "start_end_l2",
    "start_end_cosine",
)
LABEL_METRICS = (
    "mean_temporal_l2_delta",
    "median_temporal_l2_delta",
    "mean_adjacent_cosine",
    "start_end_l2",
    "start_end_cosine",
)
CLASSIFICATIONS = {
    "NO_OBVIOUS_TEMPORAL_SHIFT",
    "POSSIBLE_TEMPORAL_REDUNDANCY",
    "POSSIBLE_TEMPORAL_SPLIT_SHIFT",
    "LABEL_CONDITIONAL_TEMPORAL_PATTERN",
    "MIXED",
}


class ContextTemporalVariationAuditError(RuntimeError):
    """Temporal audit artifact 계약 또는 보호 조건 위반."""


def _cosine_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = (
        np.linalg.norm(left.astype(np.float64), axis=1)
        * np.linalg.norm(right.astype(np.float64), axis=1)
    )
    if np.any(denominator == 0):
        raise ContextTemporalVariationAuditError(
            "zero-norm feature가 있어 cosine similarity를 계산할 수 없습니다")
    return np.clip(numerator / denominator, -1.0, 1.0)


def transition_type(previous_imputed: bool, current_imputed: bool) -> str:
    if not previous_imputed and not current_imputed:
        return "original_to_original"
    if not previous_imputed and current_imputed:
        return "original_to_imputed"
    if previous_imputed and not current_imputed:
        return "imputed_to_original"
    return "imputed_to_imputed"


def temporal_statistics(
    features: np.ndarray,
    imputed_mask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    if features.shape != EXPECTED_SHAPE or features.dtype != np.float32:
        raise ContextTemporalVariationAuditError(
            f"feature 계약 위반: shape={features.shape}, dtype={features.dtype}")
    if not np.isfinite(features).all():
        raise ContextTemporalVariationAuditError("feature에 NaN/Inf가 있습니다")
    if imputed_mask.shape != (32,) or imputed_mask.dtype != np.bool_:
        raise ContextTemporalVariationAuditError(
            f"imputed mask 계약 위반: shape={imputed_mask.shape}, dtype={imputed_mask.dtype}")

    previous = features[:-1].astype(np.float64)
    current = features[1:].astype(np.float64)
    l2_delta = np.linalg.norm(current - previous, axis=1)
    cosine = _cosine_pairs(previous, current)
    start_end_l2 = float(np.linalg.norm(features[-1].astype(np.float64)
                                        - features[0].astype(np.float64)))
    start_end_cosine = float(_cosine_pairs(
        features[0:1].astype(np.float64),
        features[-1:].astype(np.float64),
    )[0])
    types = [
        transition_type(bool(imputed_mask[index]), bool(imputed_mask[index + 1]))
        for index in range(31)
    ]
    return ({
        "mean_temporal_l2_delta": float(l2_delta.mean()),
        "median_temporal_l2_delta": float(np.median(l2_delta)),
        "max_temporal_l2_delta": float(l2_delta.max()),
        "mean_adjacent_cosine": float(cosine.mean()),
        "median_adjacent_cosine": float(np.median(cosine)),
        "min_adjacent_cosine": float(cosine.min()),
        "start_end_l2": start_end_l2,
        "start_end_cosine": start_end_cosine,
    }, l2_delta, cosine, types)


def _transition_distribution(values: Sequence[float], include_min: bool = False) -> dict[str, float | int]:
    summary = distribution_summary(values)
    keys = ("count", "mean", "median", "p10", "p90", "p95", "p99", "min")
    if not include_min:
        keys = ("count", "mean", "median", "p10", "p90", "p95", "p99", "max")
    return {key: summary[key] for key in keys}


def summarize_split_transitions(
    l2_values: Sequence[float],
    cosine_values: Sequence[float],
) -> dict[str, Any]:
    if len(l2_values) != len(cosine_values) or not l2_values:
        raise ContextTemporalVariationAuditError("transition 집계가 비었거나 길이가 다릅니다")
    cosine_array = np.asarray(cosine_values, dtype=np.float64)
    return {
        "transition_count": len(l2_values),
        "adjacent_l2_delta": _transition_distribution(l2_values),
        "adjacent_cosine_similarity": _transition_distribution(
            cosine_values, include_min=True),
        "near_duplicate_transition_rate": {
            f"cosine_gte_{threshold}": float(np.mean(cosine_array >= threshold))
            for threshold in COSINE_THRESHOLDS
        },
    }


def _label_rows(backbone: str, video_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split in SPLITS:
        for label in LABELS:
            group = [row for row in video_rows
                     if row["split"] == split and row["label"] == label]
            if not group:
                raise ContextTemporalVariationAuditError(
                    f"empty temporal label group: {backbone}/{split}/{label}")
            for metric in LABEL_METRICS:
                output.append({
                    "backbone": backbone,
                    "split": split,
                    "label": label,
                    "metric": metric,
                    **distribution_summary([float(row[metric]) for row in group]),
                })
    return output


def _slot_rows(
    backbone: str,
    slot_values: Mapping[str, Sequence[Sequence[float]]],
) -> list[dict[str, Any]]:
    output = []
    for split in SPLITS:
        for index in range(31):
            l2_values = slot_values[split][index][0]
            cosine_values = slot_values[split][index][1]
            output.append({
                "backbone": backbone,
                "split": split,
                "transition_index": index,
                "from_slot": index,
                "to_slot": index + 1,
                "count": len(l2_values),
                "mean_l2_delta": float(np.mean(l2_values)),
                "median_l2_delta": float(np.median(l2_values)),
                "mean_cosine_similarity": float(np.mean(cosine_values)),
            })
    return output


def _imputation_rows(
    backbone: str,
    type_values: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
) -> list[dict[str, Any]]:
    output = []
    for split in SPLITS:
        total = sum(len(type_values[split][kind][0]) for kind in TRANSITION_TYPES)
        for kind in TRANSITION_TYPES:
            l2_values = type_values[split][kind][0]
            cosine_values = type_values[split][kind][1]
            if not l2_values:
                output.append({
                    "backbone": backbone, "split": split,
                    "transition_type": kind, "count": 0, "rate": 0.0,
                    "mean_l2_delta": None, "median_l2_delta": None,
                    "mean_cosine_similarity": None, "median_cosine_similarity": None,
                })
                continue
            output.append({
                "backbone": backbone,
                "split": split,
                "transition_type": kind,
                "count": len(l2_values),
                "rate": len(l2_values) / total,
                "mean_l2_delta": float(np.mean(l2_values)),
                "median_l2_delta": float(np.median(l2_values)),
                "mean_cosine_similarity": float(np.mean(cosine_values)),
                "median_cosine_similarity": float(np.median(cosine_values)),
            })
    return output


def _outlier_rows(backbone: str, video_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    specifications = (
        ("mean_temporal_l2_delta_high", "mean_temporal_l2_delta", True),
        ("max_temporal_l2_delta_high", "max_temporal_l2_delta", True),
        ("mean_adjacent_cosine_high", "mean_adjacent_cosine", True),
        ("mean_adjacent_cosine_low", "mean_adjacent_cosine", False),
    )
    for split in SPLITS:
        split_rows = [row for row in video_rows if row["split"] == split]
        for candidate_type, metric, reverse in specifications:
            ordered = sorted(split_rows, key=lambda row: float(row[metric]), reverse=reverse)[:10]
            for rank, row in enumerate(ordered, start=1):
                output.append({
                    "backbone": backbone, "split": split,
                    "candidate_type": candidate_type, "rank": rank,
                    "video_id": row["video_id"], "label": row["label"],
                    "metric_value": row[metric],
                })
    return output


def audit_backbone_temporal(
    artifact_root: Path,
    backbone: str,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]]]:
    if backbone not in BACKBONES:
        raise ContextTemporalVariationAuditError(f"unsupported backbone: {backbone}")
    backbone_root = artifact_root / backbone
    index_rows, index_fingerprint = _read_index(backbone_root / "full_feature_videos.csv")
    video_rows: list[dict[str, Any]] = []
    fingerprints = [index_fingerprint]
    split_l2 = {split: [] for split in SPLITS}
    split_cosine = {split: [] for split in SPLITS}
    slot_values = {
        split: [[[], []] for _ in range(31)] for split in SPLITS
    }
    type_values = {
        split: {kind: [[], []] for kind in TRANSITION_TYPES} for split in SPLITS
    }

    for row in index_rows:
        split = row["split"]
        video_id = row["video_id"]
        bundle = backbone_root / "per_video" / split / video_id
        feature_path = bundle / "features.npy"
        mask_path = bundle / "masks.npz"
        for path in (feature_path, mask_path):
            if path.is_symlink() or not path.is_file():
                raise ContextTemporalVariationAuditError(f"필수 artifact가 없습니다: {path}")
            fingerprints.append(_file_fingerprint(path))
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        with np.load(mask_path, allow_pickle=False) as masks:
            if "context_imputed_mask" not in masks.files:
                raise ContextTemporalVariationAuditError(
                    f"context_imputed_mask가 없습니다: {mask_path}")
            imputed_mask = np.asarray(masks["context_imputed_mask"])
        if int(np.count_nonzero(imputed_mask)) != int(row["imputed_count"]):
            raise ContextTemporalVariationAuditError(
                f"index/mask imputed_count mismatch: {backbone}/{split}/{video_id}")
        stats, l2_delta, cosine, kinds = temporal_statistics(features, imputed_mask)
        video_rows.append({
            "backbone": backbone, "video_id": video_id,
            "split": split, "label": row["label"], **stats,
        })
        split_l2[split].extend(float(value) for value in l2_delta)
        split_cosine[split].extend(float(value) for value in cosine)
        for index, (l2_value, cosine_value, kind) in enumerate(
                zip(l2_delta, cosine, kinds, strict=True)):
            slot_values[split][index][0].append(float(l2_value))
            slot_values[split][index][1].append(float(cosine_value))
            type_values[split][kind][0].append(float(l2_value))
            type_values[split][kind][1].append(float(cosine_value))
        del features

    actual_counts = {split: sum(row["split"] == split for row in index_rows) for split in SPLITS}
    for split in SPLITS:
        if actual_counts[split] != expected_counts[split]:
            raise ContextTemporalVariationAuditError(
                f"{backbone}/{split} count {actual_counts[split]} != {expected_counts[split]}")
    summary = {
        "backbone": backbone,
        "splits": {
            split: {
                "video_count": actual_counts[split],
                **summarize_split_transitions(split_l2[split], split_cosine[split]),
                "video_level": {
                    metric: distribution_summary([
                        float(row[metric]) for row in video_rows if row["split"] == split
                    ]) for metric in VIDEO_METRICS
                },
            } for split in SPLITS
        },
    }
    return (
        summary,
        video_rows,
        _slot_rows(backbone, slot_values),
        _label_rows(backbone, video_rows),
        _imputation_rows(backbone, type_values),
        _outlier_rows(backbone, video_rows),
        fingerprints,
    )


def classify_temporal_pattern(
    backbone_summaries: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """사전에 고정한 descriptive range/rate 규칙으로 결과를 요약한다."""

    split_evidence = {}
    redundancy_backbones = []
    for summary in backbone_summaries:
        train = summary["splits"]["train"]
        val = summary["splits"]["val"]
        outside = [
            metric for metric in LABEL_METRICS
            if not train["video_level"][metric]["p10"]
            <= val["video_level"][metric]["median"]
            <= train["video_level"][metric]["p90"]
        ]
        split_evidence[summary["backbone"]] = {
            "val_median_outside_train_p10_p90": outside,
            "shift_flag": len(outside) >= 2,
        }
        if all(
            summary["splits"][split]["near_duplicate_transition_rate"]["cosine_gte_0.99"]
            >= 0.80 for split in SPLITS
        ):
            redundancy_backbones.append(summary["backbone"])

    lookup = {
        (row["backbone"], row["split"], row["label"], row["metric"]): row
        for row in label_rows
    }
    label_evidence = []
    for backbone in BACKBONES:
        for split in SPLITS:
            not_drowsy_outside = []
            drowsy_outside = []
            for metric in LABEL_METRICS:
                alert = lookup[(backbone, split, "not_drowsy", metric)]
                drowsy = lookup[(backbone, split, "drowsy", metric)]
                if not alert["p10"] <= drowsy["median"] <= alert["p90"]:
                    drowsy_outside.append(metric)
                if not drowsy["p10"] <= alert["median"] <= drowsy["p90"]:
                    not_drowsy_outside.append(metric)
            if len(set(drowsy_outside + not_drowsy_outside)) >= 2:
                label_evidence.append({
                    "backbone": backbone, "split": split,
                    "drowsy_median_outside_not_drowsy_range": drowsy_outside,
                    "not_drowsy_median_outside_drowsy_range": not_drowsy_outside,
                })

    split_shift = any(item["shift_flag"] for item in split_evidence.values())
    redundancy = bool(redundancy_backbones)
    label_pattern = bool(label_evidence)
    pattern_count = sum((split_shift, redundancy, label_pattern))
    if pattern_count > 1:
        classification = "MIXED"
    elif split_shift:
        classification = "POSSIBLE_TEMPORAL_SPLIT_SHIFT"
    elif redundancy:
        classification = "POSSIBLE_TEMPORAL_REDUNDANCY"
    elif label_pattern:
        classification = "LABEL_CONDITIONAL_TEMPORAL_PATTERN"
    else:
        classification = "NO_OBVIOUS_TEMPORAL_SHIFT"
    if classification not in CLASSIFICATIONS:
        raise AssertionError(f"unexpected classification: {classification}")
    return classification, {
        "split_evidence": split_evidence,
        "redundancy_backbones": redundancy_backbones,
        "label_evidence": label_evidence,
        "criteria_note": (
            "Descriptive rule fixed before full audit: validation medians versus train "
            "p10-p90, label medians versus opposite-label p10-p90, and >=80% of both "
            "splits' transitions at cosine >=0.99. Not significance or causality."
        ),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    video_rows: Sequence[Mapping[str, Any]],
    slot_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    imputation_rows: Sequence[Mapping[str, Any]],
    outlier_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_dir / "video_temporal_statistics.csv",
               ("backbone", "video_id", "split", "label", *VIDEO_METRICS), video_rows)
    _write_csv(output_dir / "transition_slot_summary.csv",
               ("backbone", "split", "transition_index", "from_slot", "to_slot", "count",
                "mean_l2_delta", "median_l2_delta", "mean_cosine_similarity"), slot_rows)
    _write_csv(output_dir / "label_temporal_summary.csv",
               ("backbone", "split", "label", "metric", "count", "mean", "median",
                "p10", "p90", "p95", "p99", "min", "max"), label_rows)
    _write_csv(output_dir / "imputation_transition_summary.csv",
               ("backbone", "split", "transition_type", "count", "rate",
                "mean_l2_delta", "median_l2_delta", "mean_cosine_similarity",
                "median_cosine_similarity"), imputation_rows)
    _write_csv(output_dir / "temporal_outlier_candidates.csv",
               ("backbone", "split", "candidate_type", "rank", "video_id", "label",
                "metric_value"), outlier_rows)


def run_context_temporal_variation_audit(
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    artifact_resolved = artifact_root.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == artifact_resolved or artifact_resolved in output_resolved.parents:
        raise ContextTemporalVariationAuditError(
            "audit output은 read-only feature artifact 내부에 둘 수 없습니다")

    backbone_summaries = []
    video_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    imputation_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        result = audit_backbone_temporal(artifact_root, backbone)
        summary, videos, slots, labels, imputation, outliers, source_fingerprints = result
        backbone_summaries.append(summary)
        video_rows.extend(videos)
        slot_rows.extend(slots)
        label_rows.extend(labels)
        imputation_rows.extend(imputation)
        outlier_rows.extend(outliers)
        fingerprints.extend(source_fingerprints)

    classification, evidence = classify_temporal_pattern(backbone_summaries, label_rows)
    _verify_unchanged(fingerprints)
    summary = {
        "step": "STEP_6_E3_TEMPORAL_FEATURE_VARIATION_AUDIT",
        "scope": "TRAIN_VS_VAL_TEMPORAL_VARIATION_AND_REDUNDANCY",
        "previous": {
            "STEP_6_E1": "NO_OBVIOUS_SEQUENCE_QUALITY_SHIFT",
            "STEP_6_E2": "NO_OBVIOUS_FEATURE_DISTRIBUTION_SHIFT",
        },
        "classification": classification,
        "classification_evidence": evidence,
        "backbones": {item["backbone"]: item for item in backbone_summaries},
        "interpretation": (
            "Descriptive temporal audit only; similarity, variation, label patterns, or "
            "imputation transitions do not establish classification utility or causality."
        ),
        "protection": {
            "allowed_splits": list(SPLITS),
            "test_access_count": 0,
            "source_hash_mtime_size_unchanged": True,
            "jpeg_read": False,
            "cnn_inference": False,
            "model_inference": False,
            "checkpoint_access": False,
            "training_run": False,
            "sequence_policy_changed": False,
        },
    }
    write_outputs(
        output_dir, summary, video_rows, slot_rows, label_rows,
        imputation_rows, outlier_rows)
    return summary
