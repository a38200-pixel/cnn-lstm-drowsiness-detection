"""STEP 6-E2 frozen Context CNN feature 분포를 read-only로 감사한다."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BACKBONES = ("resnet18", "vgg16")
SPLITS = ("train", "val")
LABELS = ("not_drowsy", "drowsy")
EXPECTED_COUNTS = {"train": 1382, "val": 295}
EXPECTED_SHAPE = (32, 512)
EPSILON = 1e-12
REQUIRED_INDEX_COLUMNS = {
    "feature_video_index", "video_id", "split", "label",
    "imputed_count", "context_sequence_path",
}
CLASSIFICATIONS = {
    "NO_OBVIOUS_FEATURE_DISTRIBUTION_SHIFT",
    "POSSIBLE_FEATURE_DISTRIBUTION_SHIFT",
    "BACKBONE_SPECIFIC_SHIFT",
    "LABEL_CONDITIONAL_SHIFT",
    "MIXED",
}
VIDEO_METRICS = (
    "feature_mean", "feature_std", "mean_frame_l2_norm",
    "median_frame_l2_norm", "min_frame_l2_norm", "max_frame_l2_norm",
)
LABEL_METRICS = ("feature_mean", "feature_std", "mean_frame_l2_norm")


class ContextFeatureDistributionAuditError(RuntimeError):
    """Frozen feature 계약 또는 leakage/read-only 보호 위반."""


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ContextFeatureDistributionAuditError("분포 요약 대상이 비어 있습니다")
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile * 100))


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ContextFeatureDistributionAuditError("유한한 분포 값이 필요합니다")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": _percentile(values, 0.50),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def video_feature_statistics(features: np.ndarray) -> dict[str, float]:
    if features.shape != EXPECTED_SHAPE:
        raise ContextFeatureDistributionAuditError(
            f"feature shape {features.shape} != {EXPECTED_SHAPE}")
    if features.dtype != np.float32:
        raise ContextFeatureDistributionAuditError(
            f"feature dtype {features.dtype} != float32")
    if not np.isfinite(features).all():
        raise ContextFeatureDistributionAuditError("feature에 NaN/Inf가 있습니다")
    frame_norms = np.linalg.norm(features.astype(np.float64), axis=1)
    return {
        "feature_mean": float(features.mean(dtype=np.float64)),
        "feature_std": float(features.std(dtype=np.float64)),
        "mean_frame_l2_norm": float(frame_norms.mean()),
        "median_frame_l2_norm": float(np.median(frame_norms)),
        "min_frame_l2_norm": float(frame_norms.min()),
        "max_frame_l2_norm": float(frame_norms.max()),
    }


def dimension_shift_rows(
    backbone: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    epsilon: float = EPSILON,
) -> list[dict[str, float | int | str]]:
    for split, array in (("train", train_features), ("val", val_features)):
        if array.ndim != 2 or array.shape[1] != EXPECTED_SHAPE[1]:
            raise ContextFeatureDistributionAuditError(
                f"{split} dimension array shape가 [N,512]가 아닙니다: {array.shape}")
        if not np.isfinite(array).all():
            raise ContextFeatureDistributionAuditError(f"{split} dimension array에 NaN/Inf")
    train_mean = train_features.mean(axis=0, dtype=np.float64)
    val_mean = val_features.mean(axis=0, dtype=np.float64)
    train_std = train_features.std(axis=0, dtype=np.float64)
    val_std = val_features.std(axis=0, dtype=np.float64)
    absolute = np.abs(val_mean - train_mean)
    standardized = absolute / (train_std + epsilon)
    return [{
        "backbone": backbone,
        "dimension": dimension,
        "train_mean": float(train_mean[dimension]),
        "val_mean": float(val_mean[dimension]),
        "train_std": float(train_std[dimension]),
        "val_std": float(val_std[dimension]),
        "absolute_mean_difference": float(absolute[dimension]),
        "standardized_shift": float(standardized[dimension]),
    } for dimension in range(EXPECTED_SHAPE[1])]


def split_mean_vector_comparison(
    train_mean: np.ndarray,
    val_mean: np.ndarray,
) -> dict[str, float | None]:
    if train_mean.shape != (512,) or val_mean.shape != (512,):
        raise ContextFeatureDistributionAuditError("mean vector shape가 [512]가 아닙니다")
    denominator = float(np.linalg.norm(train_mean) * np.linalg.norm(val_mean))
    cosine = None if denominator == 0 else float(np.dot(train_mean, val_mean) / denominator)
    return {
        "cosine_similarity": cosine,
        "l2_distance": float(np.linalg.norm(val_mean - train_mean)),
    }


def _new_state() -> dict[str, Any]:
    return {
        "video_count": 0,
        "element_count": 0,
        "sum": 0.0,
        "sum_squares": 0.0,
        "min": math.inf,
        "max": -math.inf,
        "nan_count": 0,
        "inf_count": 0,
        "zero_element_count": 0,
        "all_zero_frame_count": 0,
        "all_zero_video_count": 0,
        "dimension_sum": np.zeros(512, dtype=np.float64),
        "dimension_sum_squares": np.zeros(512, dtype=np.float64),
        "dimension_count": 0,
    }


def _update_state(state: dict[str, Any], features: np.ndarray) -> None:
    finite = np.isfinite(features)
    state["nan_count"] += int(np.count_nonzero(np.isnan(features)))
    state["inf_count"] += int(np.count_nonzero(np.isinf(features)))
    state["zero_element_count"] += int(np.count_nonzero(features == 0))
    if not finite.all():
        raise ContextFeatureDistributionAuditError("feature에 NaN/Inf가 있습니다")
    values = features.astype(np.float64)
    state["video_count"] += 1
    state["element_count"] += int(values.size)
    state["sum"] += float(values.sum())
    state["sum_squares"] += float(np.square(values).sum())
    state["min"] = min(state["min"], float(values.min()))
    state["max"] = max(state["max"], float(values.max()))
    zero_frames = np.all(features == 0, axis=1)
    state["all_zero_frame_count"] += int(np.count_nonzero(zero_frames))
    state["all_zero_video_count"] += int(np.all(zero_frames))
    state["dimension_sum"] += values.sum(axis=0)
    state["dimension_sum_squares"] += np.square(values).sum(axis=0)
    state["dimension_count"] += features.shape[0]


def _finalize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    count = int(state["element_count"])
    if count == 0:
        raise ContextFeatureDistributionAuditError("feature state가 비어 있습니다")
    mean = float(state["sum"] / count)
    variance = max(0.0, float(state["sum_squares"] / count - mean * mean))
    dim_count = int(state["dimension_count"])
    dim_mean = state["dimension_sum"] / dim_count
    dim_variance = np.maximum(
        0.0, state["dimension_sum_squares"] / dim_count - np.square(dim_mean))
    return {
        "integrity": {
            "video_count": int(state["video_count"]),
            "sample_shape": list(EXPECTED_SHAPE),
            "dtype": "float32",
            "nan_count": int(state["nan_count"]),
            "inf_count": int(state["inf_count"]),
            "zero_element_count": int(state["zero_element_count"]),
            "all_zero_frame_feature_count": int(state["all_zero_frame_count"]),
            "all_zero_video_count": int(state["all_zero_video_count"]),
        },
        "global": {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": float(state["min"]),
            "max": float(state["max"]),
        },
        "dimension_mean": dim_mean,
        "dimension_std": np.sqrt(dim_variance),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": path,
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _verify_unchanged(fingerprints: Sequence[Mapping[str, Any]]) -> None:
    for before in fingerprints:
        after = _file_fingerprint(Path(before["path"]))
        for key in ("sha256", "size", "mtime_ns"):
            if after[key] != before[key]:
                raise ContextFeatureDistributionAuditError(
                    f"read-only verification failed: {before['path']} ({key})")


def _read_index(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ContextFeatureDistributionAuditError(f"feature index가 없습니다: {path}")
    fingerprint = _file_fingerprint(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_INDEX_COLUMNS - columns
        if missing:
            raise ContextFeatureDistributionAuditError(
                f"feature index columns missing: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    for position, row in enumerate(rows):
        if row["split"] not in SPLITS:
            raise ContextFeatureDistributionAuditError(
                f"test/unsupported split 접근 차단: {row['split']}")
        if "test" in Path(row["context_sequence_path"]).parts:
            raise ContextFeatureDistributionAuditError(
                "feature index의 test context path 접근을 차단합니다")
        if row["label"] not in LABELS:
            raise ContextFeatureDistributionAuditError(f"invalid label: {row['label']}")
        if int(row["feature_video_index"]) != position:
            raise ContextFeatureDistributionAuditError("feature_video_index order mismatch")
        video_id = row["video_id"]
        if not video_id or Path(video_id).name != video_id:
            raise ContextFeatureDistributionAuditError(f"unsafe video_id: {video_id!r}")
    return rows, fingerprint


def _dimension_rows_from_states(
    backbone: str,
    train: Mapping[str, Any],
    val: Mapping[str, Any],
) -> list[dict[str, Any]]:
    absolute = np.abs(val["dimension_mean"] - train["dimension_mean"])
    standardized = absolute / (train["dimension_std"] + EPSILON)
    return [{
        "backbone": backbone,
        "dimension": dimension,
        "train_mean": float(train["dimension_mean"][dimension]),
        "val_mean": float(val["dimension_mean"][dimension]),
        "train_std": float(train["dimension_std"][dimension]),
        "val_std": float(val["dimension_std"][dimension]),
        "absolute_mean_difference": float(absolute[dimension]),
        "standardized_shift": float(standardized[dimension]),
    } for dimension in range(512)]


def _summarize_video_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        metric: distribution_summary([float(row[metric]) for row in rows])
        for metric in VIDEO_METRICS
    }


def _label_group_rows(backbone: str, video_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split in SPLITS:
        for label in LABELS:
            group = [row for row in video_rows
                     if row["split"] == split and row["label"] == label]
            if not group:
                raise ContextFeatureDistributionAuditError(
                    f"empty label group: {backbone}/{split}/{label}")
            for metric in LABEL_METRICS:
                output.append({
                    "backbone": backbone,
                    "split": split,
                    "label": label,
                    "metric": metric,
                    **distribution_summary([float(row[metric]) for row in group]),
                })
    return output


def _outlier_rows(backbone: str, video_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    specifications = (
        ("mean_frame_l2_norm_high", "mean_frame_l2_norm", False),
        ("feature_std_high", "feature_std", False),
        ("absolute_feature_mean_high", "feature_mean", True),
    )
    for split in SPLITS:
        split_rows = [row for row in video_rows if row["split"] == split]
        for candidate_type, metric, absolute in specifications:
            ordered = sorted(
                split_rows,
                key=lambda row: abs(float(row[metric])) if absolute else float(row[metric]),
                reverse=True,
            )[:10]
            for rank, row in enumerate(ordered, start=1):
                output.append({
                    "backbone": backbone,
                    "split": split,
                    "candidate_type": candidate_type,
                    "rank": rank,
                    "video_id": row["video_id"],
                    "label": row["label"],
                    "metric_value": row[metric],
                    "ranking_value": abs(float(row[metric])) if absolute else row[metric],
                })
    return output


def audit_backbone(
    artifact_root: Path,
    backbone: str,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if backbone not in BACKBONES:
        raise ContextFeatureDistributionAuditError(f"unsupported backbone: {backbone}")
    backbone_root = artifact_root / backbone
    index_rows, index_fingerprint = _read_index(backbone_root / "full_feature_videos.csv")
    states = {split: _new_state() for split in SPLITS}
    video_rows: list[dict[str, Any]] = []
    fingerprints = [index_fingerprint]

    for row in index_rows:
        split = row["split"]
        video_id = row["video_id"]
        feature_path = backbone_root / "per_video" / split / video_id / "features.npy"
        if feature_path.is_symlink() or not feature_path.is_file():
            raise ContextFeatureDistributionAuditError(f"features.npy가 없습니다: {feature_path}")
        fingerprints.append(_file_fingerprint(feature_path))
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        stats = video_feature_statistics(features)
        _update_state(states[split], features)
        video_rows.append({
            "backbone": backbone,
            "video_id": video_id,
            "split": split,
            "label": row["label"],
            **stats,
        })
        del features

    actual_counts = {split: sum(row["split"] == split for row in index_rows) for split in SPLITS}
    for split in SPLITS:
        if actual_counts[split] != expected_counts[split]:
            raise ContextFeatureDistributionAuditError(
                f"{backbone}/{split} count {actual_counts[split]} != {expected_counts[split]}")

    finalized = {split: _finalize_state(states[split]) for split in SPLITS}
    dimension_rows = _dimension_rows_from_states(
        backbone, finalized["train"], finalized["val"])
    standardized = [float(row["standardized_shift"]) for row in dimension_rows]
    ordered_dimensions = sorted(
        dimension_rows, key=lambda row: float(row["standardized_shift"]), reverse=True)
    train_video_rows = [row for row in video_rows if row["split"] == "train"]
    val_video_rows = [row for row in video_rows if row["split"] == "val"]
    center = split_mean_vector_comparison(
        finalized["train"]["dimension_mean"], finalized["val"]["dimension_mean"])
    summary = {
        "backbone": backbone,
        "splits": {
            split: {
                "integrity": finalized[split]["integrity"],
                "global": finalized[split]["global"],
                "video_level": _summarize_video_metrics(
                    train_video_rows if split == "train" else val_video_rows),
            } for split in SPLITS
        },
        "dimension_standardized_shift": {
            "median": _percentile(standardized, 0.50),
            "p90": _percentile(standardized, 0.90),
            "p95": _percentile(standardized, 0.95),
            "p99": _percentile(standardized, 0.99),
            "max": max(standardized),
            "top10": ordered_dimensions[:10],
            "description": "abs(val_mean-train_mean)/(train_std+epsilon); descriptive, not a hypothesis test",
        },
        "split_mean_vector": center,
    }
    return (summary, video_rows, dimension_rows, _label_group_rows(backbone, video_rows),
            _outlier_rows(backbone, video_rows), fingerprints)


def classify_feature_shift(
    backbone_summaries: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Train reference ranges를 이용한 보수적 descriptive classification."""

    backbone_evidence: dict[str, dict[str, Any]] = {}
    for summary in backbone_summaries:
        train = summary["splits"]["train"]["video_level"]
        val = summary["splits"]["val"]["video_level"]
        outside_metrics = [
            metric for metric in LABEL_METRICS
            if not train[metric]["p10"] <= val[metric]["median"] <= train[metric]["p90"]
        ]
        dimension_large = summary["dimension_standardized_shift"]["p95"] >= 0.5
        cosine = summary["split_mean_vector"]["cosine_similarity"]
        center_large = cosine is not None and cosine < 0.99
        signals = len(outside_metrics) + int(dimension_large) + int(center_large)
        backbone_evidence[summary["backbone"]] = {
            "val_median_outside_train_p10_p90": outside_metrics,
            "dimension_p95_at_least_0_5": dimension_large,
            "mean_vector_cosine_below_0_99": center_large,
            "shift_flag": signals >= 2,
        }

    label_lookup = {
        (row["backbone"], row["split"], row["label"], row["metric"]): row
        for row in label_rows
    }
    conditional_evidence = []
    for backbone in BACKBONES:
        for label in LABELS:
            outside = []
            for metric in LABEL_METRICS:
                train = label_lookup[(backbone, "train", label, metric)]
                val = label_lookup[(backbone, "val", label, metric)]
                if not train["p10"] <= val["median"] <= train["p90"]:
                    outside.append(metric)
            if len(outside) >= 2:
                conditional_evidence.append({
                    "backbone": backbone, "label": label,
                    "val_median_outside_train_p10_p90": outside,
                })

    flagged = [name for name, evidence in backbone_evidence.items() if evidence["shift_flag"]]
    if len(flagged) == len(BACKBONES):
        classification = "POSSIBLE_FEATURE_DISTRIBUTION_SHIFT"
    elif len(flagged) == 1:
        classification = "BACKBONE_SPECIFIC_SHIFT"
    elif conditional_evidence:
        classification = "LABEL_CONDITIONAL_SHIFT"
    elif any(
        evidence["val_median_outside_train_p10_p90"]
        or evidence["dimension_p95_at_least_0_5"]
        or evidence["mean_vector_cosine_below_0_99"]
        for evidence in backbone_evidence.values()
    ):
        classification = "MIXED"
    else:
        classification = "NO_OBVIOUS_FEATURE_DISTRIBUTION_SHIFT"
    if classification not in CLASSIFICATIONS:
        raise AssertionError(f"unexpected classification: {classification}")
    return classification, {
        "backbone_evidence": backbone_evidence,
        "label_conditional_evidence": conditional_evidence,
        "criteria_note": (
            "Descriptive rule: validation video-metric medians versus train p10-p90, "
            "dimension-shift p95, and split mean-vector cosine; not significance testing."
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
    dimension_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    outlier_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_dir / "video_feature_statistics.csv",
               ("backbone", "video_id", "split", "label", *VIDEO_METRICS), video_rows)
    _write_csv(output_dir / "dimension_shift.csv",
               ("backbone", "dimension", "train_mean", "val_mean", "train_std",
                "val_std", "absolute_mean_difference", "standardized_shift"),
               dimension_rows)
    _write_csv(output_dir / "label_group_summary.csv",
               ("backbone", "split", "label", "metric", "count", "mean", "median",
                "p10", "p90", "p95", "p99", "min", "max"), label_rows)
    _write_csv(output_dir / "outlier_candidates.csv",
               ("backbone", "split", "candidate_type", "rank", "video_id", "label",
                "metric_value", "ranking_value"), outlier_rows)


def run_context_feature_distribution_audit(
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    artifact_resolved = artifact_root.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == artifact_resolved or artifact_resolved in output_resolved.parents:
        raise ContextFeatureDistributionAuditError(
            "audit output은 read-only feature artifact 내부에 둘 수 없습니다")

    backbone_summaries = []
    video_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        result = audit_backbone(artifact_root, backbone)
        summary, videos, dimensions, labels, outliers, source_fingerprints = result
        backbone_summaries.append(summary)
        video_rows.extend(videos)
        dimension_rows.extend(dimensions)
        label_rows.extend(labels)
        outlier_rows.extend(outliers)
        fingerprints.extend(source_fingerprints)

    classification, evidence = classify_feature_shift(backbone_summaries, label_rows)
    _verify_unchanged(fingerprints)
    summary = {
        "step": "STEP_6_E2_FROZEN_FEATURE_DISTRIBUTION_AUDIT",
        "scope": "TRAIN_VS_VAL_STATIC_FROZEN_FEATURE_DISTRIBUTION",
        "previous": {
            "step": "STEP_6_E1",
            "classification": "NO_OBVIOUS_SEQUENCE_QUALITY_SHIFT",
            "sequence_imputation_policy_changed": False,
        },
        "classification": classification,
        "classification_evidence": evidence,
        "backbones": {item["backbone"]: item for item in backbone_summaries},
        "interpretation": (
            "Descriptive audit only; observed feature differences are not established "
            "causes of validation-loss behavior or statistical significance."
        ),
        "protection": {
            "allowed_splits": list(SPLITS),
            "test_access_count": 0,
            "source_hash_mtime_size_unchanged": True,
            "jpeg_read": False,
            "cnn_inference": False,
            "model_used": False,
            "training_run": False,
            "temporal_delta_analyzed": False,
        },
    }
    write_outputs(
        output_dir, summary, video_rows, dimension_rows, label_rows, outlier_rows)
    return summary
