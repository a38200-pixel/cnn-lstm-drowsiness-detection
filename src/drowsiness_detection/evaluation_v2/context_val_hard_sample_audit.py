"""STEP 6-E4 validation per-sample loss 및 hard-sample 감사."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from drowsiness_detection.data_v2 import (
    ContextFeatureDataset,
    build_context_feature_dataloader,
)
from drowsiness_detection.models_v2 import build_context_lstm


BACKBONES = ("resnet18", "vgg16")
EXPECTED_VAL_COUNT = 295
LABEL_NAMES = {0: "not_drowsy", 1: "drowsy"}
METRIC_TOLERANCE = 1.0e-6
BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "resnet18": {
        "checkpoint": "outputs/training_v2/context_lstm/resnet18/seed_42/best_model.pt",
        "best_epoch": 16,
        "val_loss": 0.4905628689264847,
        "accuracy": 0.7796610169491526,
        "macro_f1": 0.7784287216463872,
        "confusion_matrix": [[126, 27], [38, 104]],
    },
    "vgg16": {
        "checkpoint": "outputs/training_v2/context_lstm/vgg16/seed_42/best_model.pt",
        "best_epoch": 7,
        "val_loss": 0.46366154884887956,
        "accuracy": 0.8067796610169492,
        "macro_f1": 0.8060576002583593,
        "confusion_matrix": [[128, 25], [32, 110]],
    },
}

E1_FIELDS = ("imputed_count", "max_imputed_run")
E2_FIELDS = ("feature_mean", "feature_std", "mean_frame_l2_norm")
E3_FIELDS = (
    "mean_temporal_l2_delta",
    "mean_adjacent_cosine",
    "start_end_l2",
    "start_end_cosine",
)


class ContextValHardSampleAuditError(RuntimeError):
    """감사 계약, metric 재현 또는 artifact 무결성 위반."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContextValHardSampleAuditError(f"필수 CSV가 없습니다: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprints(paths: Iterable[Path]) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ContextValHardSampleAuditError(f"읽기 전용 artifact가 없습니다: {resolved}")
        stat = resolved.stat()
        result[str(resolved)] = (stat.st_size, stat.st_mtime_ns, _sha256(resolved))
    return result


def _verify_unchanged(
    before: Mapping[str, tuple[int, int, str]],
    description: str,
) -> None:
    after = _fingerprints(Path(path) for path in before)
    if dict(before) != after:
        raise ContextValHardSampleAuditError(
            f"{description} artifact가 감사 중 변경되었습니다")


def _safe_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ContextValHardSampleAuditError(f"{name}에 NaN/Inf가 있습니다")
    return number


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> dict[str, Any]:
    """Binary accuracy, macro F1, confusion matrix를 계산한다."""

    if len(labels) != len(predictions) or not labels:
        raise ContextValHardSampleAuditError("metric 입력 길이가 잘못되었습니다")
    confusion = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predictions):
        if label not in LABEL_NAMES or prediction not in LABEL_NAMES:
            raise ContextValHardSampleAuditError("binary label 범위를 벗어났습니다")
        confusion[label][prediction] += 1
    accuracy = sum(confusion[index][index] for index in range(2)) / len(labels)
    f1_scores: list[float] = []
    for class_index in range(2):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(confusion[row][class_index] for row in range(2)) - true_positive
        false_negative = sum(confusion[class_index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / 2,
        "confusion_matrix": confusion,
    }


def per_sample_rows_from_logits(
    backbone: str,
    video_ids: Sequence[str],
    labels: Tensor,
    logits: Tensor,
) -> list[dict[str, Any]]:
    """Unsmoothed CE(reduction=none)로 video별 결과를 만든다."""

    if backbone not in BACKBONES:
        raise ContextValHardSampleAuditError(f"지원하지 않는 backbone입니다: {backbone}")
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.ndim != 1:
        raise ContextValHardSampleAuditError("logits/label shape가 잘못되었습니다")
    if len(video_ids) != logits.shape[0] or labels.shape[0] != logits.shape[0]:
        raise ContextValHardSampleAuditError("batch field 길이가 일치하지 않습니다")
    if logits.dtype != torch.float32 or labels.dtype != torch.long:
        raise ContextValHardSampleAuditError("logits는 float32, label은 long이어야 합니다")
    if not torch.isfinite(logits).all():
        raise ContextValHardSampleAuditError("logits에 NaN/Inf가 있습니다")

    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.0)
    losses = criterion(logits, labels)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)
    rows: list[dict[str, Any]] = []
    for index, video_id in enumerate(video_ids):
        label = int(labels[index].item())
        prediction = int(predictions[index].item())
        rows.append({
            "backbone": backbone,
            "video_id": str(video_id),
            "label": LABEL_NAMES[label],
            "label_index": label,
            "predicted_label": LABEL_NAMES[prediction],
            "predicted_label_index": prediction,
            "correct": label == prediction,
            "logit_not_drowsy": float(logits[index, 0].item()),
            "logit_drowsy": float(logits[index, 1].item()),
            "probability_not_drowsy": float(probabilities[index, 0].item()),
            "probability_drowsy": float(probabilities[index, 1].item()),
            "true_class_probability": float(probabilities[index, label].item()),
            "predicted_class_probability": float(probabilities[index, prediction].item()),
            "per_sample_ce_loss": float(losses[index].item()),
        })
    return rows


def aggregate_inference_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label_index"]) for row in rows]
    predictions = [int(row["predicted_label_index"]) for row in rows]
    metrics = classification_metrics(labels, predictions)
    metrics["val_loss"] = float(np.mean(
        [float(row["per_sample_ce_loss"]) for row in rows], dtype=np.float64))
    metrics["video_count"] = len(rows)
    return metrics


def verify_metric_reproduction(
    backbone: str,
    metrics: Mapping[str, Any],
    *,
    tolerance: float = METRIC_TOLERANCE,
) -> dict[str, Any]:
    """Baseline best metric과 일치하지 않으면 즉시 중단한다."""

    expected = BASELINE_SPECS[backbone]
    mismatches: dict[str, Any] = {}
    for name in ("val_loss", "accuracy", "macro_f1"):
        difference = abs(float(metrics[name]) - float(expected[name]))
        if difference > tolerance:
            mismatches[name] = {
                "expected": expected[name], "actual": metrics[name], "difference": difference}
    if metrics["confusion_matrix"] != expected["confusion_matrix"]:
        mismatches["confusion_matrix"] = {
            "expected": expected["confusion_matrix"],
            "actual": metrics["confusion_matrix"],
        }
    if int(metrics["video_count"]) != EXPECTED_VAL_COUNT:
        mismatches["video_count"] = {
            "expected": EXPECTED_VAL_COUNT, "actual": metrics["video_count"]}
    if mismatches:
        raise ContextValHardSampleAuditError(
            f"{backbone} baseline metric 재현 실패: {mismatches}. "
            "hard-sample 분석을 중단합니다.")
    return {
        "passed": True,
        "tolerance": tolerance,
        "expected": {key: expected[key] for key in (
            "val_loss", "accuracy", "macro_f1", "confusion_matrix")},
        "actual": dict(metrics),
    }


def _validate_checkpoint(checkpoint: Mapping[str, Any], backbone: str) -> None:
    spec = BASELINE_SPECS[backbone]
    if int(checkpoint.get("epoch", -1)) != spec["best_epoch"]:
        raise ContextValHardSampleAuditError(f"{backbone} checkpoint epoch가 baseline과 다릅니다")
    if abs(float(checkpoint.get("val_loss", math.inf)) - spec["val_loss"]) > METRIC_TOLERANCE:
        raise ContextValHardSampleAuditError(f"{backbone} checkpoint val_loss가 baseline과 다릅니다")
    if checkpoint.get("test_used") is not False:
        raise ContextValHardSampleAuditError("test_used=False checkpoint만 허용합니다")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ContextValHardSampleAuditError("checkpoint metadata가 없습니다")
    required = {"backbone": backbone, "seed": 42}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ContextValHardSampleAuditError(
                f"baseline checkpoint metadata 불일치: {key}={metadata.get(key)!r}")
    if metadata.get("run_tag") not in (None, ""):
        raise ContextValHardSampleAuditError("tuning checkpoint는 E4에서 사용할 수 없습니다")
    # 초기 baseline checkpoint에는 아래 field가 아직 저장되지 않았다. 존재할 때는
    # baseline 값만 허용하고, 구조는 strict state_dict 및 고정 config로 검증한다.
    optional_baseline = {
        "classifier_dropout": 0.0,
        "lstm_dropout": 0.0,
        "input_layer_norm": False,
        "label_smoothing": 0.0,
    }
    for key, expected in optional_baseline.items():
        if key in metadata and metadata[key] != expected:
            raise ContextValHardSampleAuditError(
                f"tuning checkpoint metadata가 감지되었습니다: {key}={metadata[key]!r}")


def _run_validation_inference(
    project_root: Path,
    config: Mapping[str, Any],
    backbone: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], ContextFeatureDataset]:
    dataset = ContextFeatureDataset(project_root, backbone, "val")
    feature_paths = [dataset.backbone_root / "full_feature_videos.csv"]
    for record in dataset.records:
        bundle = dataset.backbone_root / "per_video" / "val" / record.video_id
        feature_paths.extend((bundle / "features.npy", bundle / "masks.npz"))
    feature_before = _fingerprints(feature_paths)
    loader = build_context_feature_dataloader(
        dataset,
        int(config["data"]["val_batch_size"]),
        num_workers=0,
    )
    model = build_context_lstm(config).to(device)
    checkpoint_path = project_root / BASELINE_SPECS[backbone]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ContextValHardSampleAuditError("checkpoint 형식이 잘못되었습니다")
    _validate_checkpoint(checkpoint, backbone)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            logits = model(features)
            rows.extend(per_sample_rows_from_logits(
                backbone,
                list(batch["video_id"]),
                labels.cpu(),
                logits.cpu(),
            ))
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ContextValHardSampleAuditError(f"{backbone} validation video_id가 중복됩니다")
    _verify_unchanged(feature_before, f"{backbone} frozen feature")
    return rows, aggregate_inference_metrics(rows), dataset


def loss_concentration_rows(
    backbone: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["per_sample_ce_loss"]), reverse=True)
    total = sum(float(row["per_sample_ce_loss"]) for row in ordered)
    if total <= 0 or not math.isfinite(total):
        raise ContextValHardSampleAuditError("전체 CE 합이 유효하지 않습니다")
    segments = (
        ("top_5", 5),
        ("top_10", 10),
        ("top_20", 20),
        ("top_10_percent", math.ceil(len(ordered) * 0.10)),
        ("top_20_percent", math.ceil(len(ordered) * 0.20)),
    )
    result = []
    for name, count in segments:
        ce_sum = sum(float(row["per_sample_ce_loss"]) for row in ordered[:count])
        result.append({
            "backbone": backbone,
            "segment": name,
            "sample_count": count,
            "validation_count": len(ordered),
            "ce_sum": ce_sum,
            "total_ce_sum": total,
            "share": ce_sum / total,
        })
    return result


def hard_sample_rows(
    backbone: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_loss = sorted(rows, key=lambda row: float(row["per_sample_ce_loss"]), reverse=True)[:20]
    errors = [row for row in rows if not bool(row["correct"])]
    by_confidence = sorted(
        errors,
        key=lambda row: float(row["predicted_class_probability"]),
        reverse=True,
    )[:20]
    result: list[dict[str, Any]] = []
    for candidate_type, candidates in (
        ("loss_top20", by_loss), ("confident_error_top20", by_confidence)):
        for rank, row in enumerate(candidates, start=1):
            result.append({"candidate_type": candidate_type, "rank": rank, **dict(row)})
    return result


def backbone_overlap_rows(
    resnet_rows: Sequence[Mapping[str, Any]],
    vgg_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    resnet = {str(row["video_id"]): row for row in resnet_rows}
    vgg = {str(row["video_id"]): row for row in vgg_rows}
    if list(resnet) != list(vgg):
        raise ContextValHardSampleAuditError("backbone validation universe/order가 다릅니다")
    resnet_top = {
        row["video_id"] for row in sorted(
            resnet_rows, key=lambda item: float(item["per_sample_ce_loss"]), reverse=True)[:20]}
    vgg_top = {
        row["video_id"] for row in sorted(
            vgg_rows, key=lambda item: float(item["per_sample_ce_loss"]), reverse=True)[:20]}
    result: list[dict[str, Any]] = []
    for video_id in resnet:
        left, right = resnet[video_id], vgg[video_id]
        categories: list[str] = []
        if not left["correct"] and not right["correct"]:
            categories.append("both_incorrect")
        elif not left["correct"]:
            categories.append("resnet18_only_incorrect")
        elif not right["correct"]:
            categories.append("vgg16_only_incorrect")
        if video_id in resnet_top and video_id in vgg_top:
            categories.append("both_high_loss_top20")
        for category in categories:
            result.append({
                "category": category,
                "video_id": video_id,
                "label": left["label"],
                "resnet18_predicted_label": left["predicted_label"],
                "vgg16_predicted_label": right["predicted_label"],
                "resnet18_correct": left["correct"],
                "vgg16_correct": right["correct"],
                "resnet18_ce_loss": left["per_sample_ce_loss"],
                "vgg16_ce_loss": right["per_sample_ce_loss"],
            })
    return result


def _keyed_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    backbone: str | None,
    source_name: str,
) -> dict[str, Mapping[str, str]]:
    selected: dict[str, Mapping[str, str]] = {}
    for row in rows:
        if row.get("split") != "val":
            continue
        if backbone is not None and row.get("backbone") != backbone:
            continue
        video_id = str(row.get("video_id", ""))
        if not video_id or video_id in selected:
            raise ContextValHardSampleAuditError(f"{source_name} video_id가 없거나 중복됩니다")
        selected[video_id] = row
    if len(selected) != EXPECTED_VAL_COUNT:
        raise ContextValHardSampleAuditError(
            f"{source_name} val count가 {EXPECTED_VAL_COUNT}가 아닙니다: {len(selected)}")
    return selected


def join_audit_fields(
    rows: Sequence[Mapping[str, Any]],
    e1: Mapping[str, Mapping[str, str]],
    e2: Mapping[str, Mapping[str, str]],
    e3: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        video_id = str(source["video_id"])
        if video_id not in e1 or video_id not in e2 or video_id not in e3:
            raise ContextValHardSampleAuditError(f"audit join 누락: {video_id}")
        enriched = dict(source)
        enriched["imputed_count"] = int(e1[video_id]["imputed_count"])
        enriched["max_imputed_run"] = int(e1[video_id]["max_imputed_run"])
        for field in E2_FIELDS:
            enriched[field] = _safe_float(e2[video_id][field], field)
        for field in E3_FIELDS:
            enriched[field] = _safe_float(e3[video_id][field], field)
        result.append(enriched)
    return result


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if math.isfinite(value) else None


def descriptive_relations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        values = [float(row["per_sample_ce_loss"]) for row in group]
        return {
            "count": len(values),
            "mean_ce": None if not values else float(np.mean(values, dtype=np.float64)),
            "median_ce": None if not values else float(median(values)),
        }

    groups = {
        "prediction": {
            "correct": summarize([row for row in rows if row["correct"]]),
            "incorrect": summarize([row for row in rows if not row["correct"]]),
        },
        "imputation": {
            "zero": summarize([row for row in rows if int(row["imputed_count"]) == 0]),
            "positive": summarize([row for row in rows if int(row["imputed_count"]) > 0]),
        },
        "label": {
            label: summarize([row for row in rows if row["label"] == label])
            for label in ("not_drowsy", "drowsy")
        },
    }
    losses = [float(row["per_sample_ce_loss"]) for row in rows]
    correlations = {
        field: spearman_correlation(losses, [float(row[field]) for row in rows])
        for field in (
            "imputed_count",
            "mean_frame_l2_norm",
            "mean_temporal_l2_delta",
            "mean_adjacent_cosine",
        )
    }
    return {"groups": groups, "spearman_with_per_sample_ce": correlations}


def classify_result(
    concentration: Sequence[Mapping[str, Any]],
    overlap: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """사전에 고정한 descriptive threshold로 결과 유형을 분류한다."""

    top_10_percent = {
        str(row["backbone"]): float(row["share"])
        for row in concentration if row["segment"] == "top_10_percent"
    }
    counts = {
        category: sum(row["category"] == category for row in overlap)
        for category in (
            "both_incorrect", "resnet18_only_incorrect",
            "vgg16_only_incorrect", "both_high_loss_top20")
    }
    concentrated = any(share >= 0.50 for share in top_10_percent.values())
    shared = counts["both_high_loss_top20"] >= 10
    specific = (
        counts["resnet18_only_incorrect"] + counts["vgg16_only_incorrect"]
        > counts["both_incorrect"]
    )
    if concentrated:
        classification = "LOSS_CONCENTRATED_IN_FEW_SAMPLES"
    elif shared and specific:
        classification = "MIXED"
    elif shared:
        classification = "SHARED_BACKBONE_HARD_SAMPLES"
    elif specific:
        classification = "BACKBONE_SPECIFIC_HARD_SAMPLES"
    else:
        classification = "BROAD_VALIDATION_DIFFICULTY"
    evidence = {
        "top_10_percent_ce_share": top_10_percent,
        "overlap_counts": counts,
        "criteria": {
            "loss_concentrated": "any backbone top 10% CE share >= 0.50",
            "shared": "both-high-loss top20 overlap >= 10",
            "backbone_specific": "one-backbone-only errors > both-backbone errors",
            "mixed": "shared and backbone-specific criteria both true",
        },
        "interpretation_limit": "descriptive association only; no causal conclusion",
    }
    return classification, evidence


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ContextValHardSampleAuditError(f"빈 CSV는 저장하지 않습니다: {path.name}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _source_paths(project_root: Path) -> dict[str, Path]:
    return {
        "config": project_root / "configs/context_lstm_baseline.yaml",
        "e1": project_root / "outputs/audits_v2/context_sequence_quality/video_sequence_quality.csv",
        "e2": project_root / "outputs/audits_v2/context_feature_distribution/video_feature_statistics.csv",
        "e3": project_root / "outputs/audits_v2/context_temporal_variation/video_temporal_statistics.csv",
    }


def run_context_val_hard_sample_audit(
    project_root: Path | str,
    output_dir: Path | str,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """두 baseline의 재현 gate를 통과한 뒤에만 E4 산출물을 저장한다."""

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name in ("cpu", "cuda"):
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ContextValHardSampleAuditError("CUDA를 사용할 수 없습니다")
        device = torch.device(device_name)
    else:
        raise ContextValHardSampleAuditError("device는 auto, cpu, cuda 중 하나여야 합니다")

    sources = _source_paths(root)
    config = yaml.safe_load(sources["config"].read_text(encoding="utf-8"))
    if config["test_policy"]["sealed"] is not True:
        raise ContextValHardSampleAuditError("test_policy.sealed=true가 필요합니다")
    if config["model_baseline_future"].get("input_layer_norm") is not False:
        raise ContextValHardSampleAuditError("baseline input LayerNorm은 off여야 합니다")
    if float(config["model_baseline_future"]["classifier_dropout"]) != 0.0:
        raise ContextValHardSampleAuditError("baseline classifier dropout은 0.0이어야 합니다")

    checkpoint_paths = [root / BASELINE_SPECS[name]["checkpoint"] for name in BACKBONES]
    checkpoint_before = _fingerprints(checkpoint_paths)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    reproductions: dict[str, Any] = {}
    datasets: dict[str, ContextFeatureDataset] = {}
    for backbone in BACKBONES:
        rows, metrics, dataset = _run_validation_inference(root, config, backbone, device)
        reproductions[backbone] = verify_metric_reproduction(backbone, metrics)
        all_rows[backbone] = rows
        datasets[backbone] = dataset
    if datasets["resnet18"].ordered_signature != datasets["vgg16"].ordered_signature:
        raise ContextValHardSampleAuditError("두 backbone의 val universe/order가 다릅니다")

    _verify_unchanged(checkpoint_before, "checkpoint")

    audit_paths = [sources["e1"], sources["e2"], sources["e3"]]
    audit_before = _fingerprints(audit_paths)
    e1_rows, e2_rows, e3_rows = (_read_csv(path) for path in audit_paths)
    e1 = _keyed_rows(e1_rows, backbone=None, source_name="E1")
    enriched: dict[str, list[dict[str, Any]]] = {}
    for backbone in BACKBONES:
        e2 = _keyed_rows(e2_rows, backbone=backbone, source_name=f"E2/{backbone}")
        e3 = _keyed_rows(e3_rows, backbone=backbone, source_name=f"E3/{backbone}")
        enriched[backbone] = join_audit_fields(all_rows[backbone], e1, e2, e3)
    _verify_unchanged(audit_before, "E1-E3 audit")

    per_sample = [row for backbone in BACKBONES for row in enriched[backbone]]
    concentration = [
        row for backbone in BACKBONES
        for row in loss_concentration_rows(backbone, enriched[backbone])]
    hard_samples = [
        row for backbone in BACKBONES
        for row in hard_sample_rows(backbone, enriched[backbone])]
    overlap = backbone_overlap_rows(enriched["resnet18"], enriched["vgg16"])
    classification, evidence = classify_result(concentration, overlap)
    summary = {
        "step": "6-E4",
        "status": "COMPLETE",
        "classification": classification,
        "classification_evidence": evidence,
        "metric_reproduction": reproductions,
        "descriptive_relations": {
            backbone: descriptive_relations(enriched[backbone]) for backbone in BACKBONES},
        "checkpoint_paths": {
            backbone: BASELINE_SPECS[backbone]["checkpoint"] for backbone in BACKBONES},
        "loss": {"name": "CrossEntropyLoss", "reduction": "none", "label_smoothing": 0.0},
        "protections": {
            "test_sealed": True,
            "test_access_count": 0,
            "training_count": 0,
            "jpeg_image_access_count": 0,
            "checkpoint_read_only_verified": True,
            "frozen_feature_read_only_verified": True,
            "existing_audit_read_only_verified": True,
        },
    }

    # 재현과 모든 read-only 검증이 끝난 뒤에만 결과 디렉터리를 쓴다.
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_sample_loss.csv", per_sample)
    _write_csv(output / "loss_concentration.csv", concentration)
    _write_csv(output / "hard_sample_top20.csv", hard_samples)
    _write_csv(output / "backbone_error_overlap.csv", overlap)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
