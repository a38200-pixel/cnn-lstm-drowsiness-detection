"""STEP 6-E1 train/validation Context sequence 품질을 read-only로 감사한다."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ALLOWED_SPLITS = ("train", "val")
EXPECTED_VIDEO_COUNTS = {"train": 1382, "val": 295}
SEQUENCE_LENGTH = 32
MAX_IMPUTED_SLOTS = 8
MAX_CONSECUTIVE_RUN = 4
SUMMARY_LABELS = {
    "NO_OBVIOUS_SEQUENCE_QUALITY_SHIFT",
    "POSSIBLE_SEQUENCE_QUALITY_SHIFT",
    "LOCALIZED_SLOT_SHIFT",
    "MIXED",
}
REQUIRED_COLUMNS = {"video_id", "split", "target_context_index", "imputed"}


class ContextSequenceQualityAuditError(RuntimeError):
    """Sequence artifact 구조 또는 보호 조건 위반."""


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized not in {"true", "false", "1", "0"}:
        raise ContextSequenceQualityAuditError(f"invalid imputed boolean: {value!r}")
    return normalized in {"true", "1"}


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        raise ContextSequenceQualityAuditError("percentile input이 비어 있습니다")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def max_consecutive_true(mask: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def analyze_video(video_id: str, split: str, mask: Sequence[bool]) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ContextSequenceQualityAuditError(f"test/unsupported split 접근 차단: {split}")
    if len(mask) != SEQUENCE_LENGTH:
        raise ContextSequenceQualityAuditError(
            f"{split}/{video_id}: sequence length {len(mask)} != {SEQUENCE_LENGTH}")
    normalized = [bool(value) for value in mask]
    imputed_slots = [index for index, value in enumerate(normalized) if value]
    return {
        "video_id": video_id,
        "split": split,
        "imputed_count": len(imputed_slots),
        "imputed_rate": len(imputed_slots) / SEQUENCE_LENGTH,
        "max_imputed_run": max_consecutive_true(normalized),
        "imputed_slots": imputed_slots,
        "mask": normalized,
    }


def _fingerprint(path: Path, payload: bytes) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_sequence_csv(path: Path, split: str, expected_video_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = path.read_bytes()
    fingerprint = _fingerprint(path, payload)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    columns = set(reader.fieldnames or ())
    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        raise ContextSequenceQualityAuditError(
            f"{path}: required columns missing: {sorted(missing_columns)}")
    rows = list(reader)
    if len(rows) != SEQUENCE_LENGTH:
        raise ContextSequenceQualityAuditError(
            f"{path}: row count {len(rows)} != {SEQUENCE_LENGTH}")

    indices = [int(row["target_context_index"]) for row in rows]
    if indices != list(range(SEQUENCE_LENGTH)):
        raise ContextSequenceQualityAuditError(f"{path}: target slot order가 0..31이 아닙니다")
    if any(row["video_id"] != expected_video_id for row in rows):
        raise ContextSequenceQualityAuditError(f"{path}: video_id mismatch")
    if any(row["split"] != split for row in rows):
        raise ContextSequenceQualityAuditError(f"{path}: split mismatch")
    mask = [_parse_bool(row["imputed"]) for row in rows]
    return analyze_video(expected_video_id, split, mask), fingerprint


def load_split_sequences(
    sequence_root: Path,
    split: str,
    expected_video_count: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """허용된 split의 sequence.csv만 읽는다. 이미지와 feature 파일은 열지 않는다."""

    if split not in ALLOWED_SPLITS:
        raise ContextSequenceQualityAuditError(f"test/unsupported split 접근 차단: {split}")
    split_root = sequence_root / "per_video" / split
    if not split_root.is_dir():
        raise ContextSequenceQualityAuditError(f"split directory가 없습니다: {split_root}")

    records: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    for video_dir in sorted(split_root.iterdir(), key=lambda path: path.name):
        if video_dir.is_symlink() or not video_dir.is_dir():
            raise ContextSequenceQualityAuditError(f"unexpected split entry: {video_dir}")
        sequence_path = video_dir / "sequence.csv"
        if sequence_path.is_symlink() or not sequence_path.is_file():
            raise ContextSequenceQualityAuditError(f"sequence.csv가 없습니다: {sequence_path}")
        record, fingerprint = _read_sequence_csv(sequence_path, split, video_dir.name)
        records.append(record)
        fingerprints.append(fingerprint)

    if expected_video_count is not None and len(records) != expected_video_count:
        raise ContextSequenceQualityAuditError(
            f"{split} video count {len(records)} != {expected_video_count}")
    return records, fingerprints


def verify_artifacts_unchanged(fingerprints: Sequence[Mapping[str, Any]]) -> None:
    for before in fingerprints:
        path = Path(before["path"])
        payload = path.read_bytes()
        after = _fingerprint(path, payload)
        for field in ("sha256", "size", "mtime_ns"):
            if after[field] != before[field]:
                raise ContextSequenceQualityAuditError(
                    f"read-only artifact verification failed: {path} ({field})")


def _bucket_counts(counts: Sequence[int]) -> dict[str, int]:
    return {
        "0": sum(count == 0 for count in counts),
        "1-2": sum(1 <= count <= 2 for count in counts),
        "3-4": sum(3 <= count <= 4 for count in counts),
        "5-8": sum(5 <= count <= 8 for count in counts),
        "over_8": sum(count > 8 for count in counts),
    }


def summarize_split(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ContextSequenceQualityAuditError("split record가 비어 있습니다")
    counts = [int(record["imputed_count"]) for record in records]
    runs = [int(record["max_imputed_run"]) for record in records]
    video_count = len(records)
    total_slots = video_count * SEQUENCE_LENGTH
    total_imputed = sum(counts)
    return {
        "video_count": video_count,
        "sequence_length": SEQUENCE_LENGTH,
        "all_sequence_lengths_valid": True,
        "total_target_slots": total_slots,
        "total_imputed_slots": total_imputed,
        "original_slots": total_slots - total_imputed,
        "imputed_video_count": sum(count > 0 for count in counts),
        "zero_imputation_video_count": sum(count == 0 for count in counts),
        "any_imputation_video_rate": sum(count > 0 for count in counts) / video_count,
        "total_imputation_rate": total_imputed / total_slots,
        "imputed_slots_per_video": {
            "mean": sum(counts) / video_count,
            "median": _percentile(counts, 0.50),
            "p90": _percentile(counts, 0.90),
            "p95": _percentile(counts, 0.95),
            "p99": _percentile(counts, 0.99),
            "max": max(counts),
            "buckets": _bucket_counts(counts),
        },
        "max_consecutive_imputation_run": {
            "mean": sum(runs) / video_count,
            "median": _percentile(runs, 0.50),
            "p90": _percentile(runs, 0.90),
            "p95": _percentile(runs, 0.95),
            "max": max(runs),
        },
        "anomalies": {
            "imputed_count_over_8": [
                record["video_id"] for record in records
                if int(record["imputed_count"]) > MAX_IMPUTED_SLOTS
            ],
            "max_run_over_4": [
                record["video_id"] for record in records
                if int(record["max_imputed_run"]) > MAX_CONSECUTIVE_RUN
            ],
        },
    }


def slot_frequency_rows(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for slot in range(SEQUENCE_LENGTH):
        train_count = sum(bool(record["mask"][slot]) for record in train_records)
        val_count = sum(bool(record["mask"][slot]) for record in val_records)
        train_rate = train_count / len(train_records)
        val_rate = val_count / len(val_records)
        rows.append({
            "slot": slot,
            "train_imputed_count": train_count,
            "train_imputed_rate": train_rate,
            "val_imputed_count": val_count,
            "val_imputed_rate": val_rate,
            "val_minus_train_rate": val_rate - train_rate,
            "val_train_rate_ratio": None if train_rate == 0 else val_rate / train_rate,
        })
    return rows


def _ratio(val_value: float, train_value: float) -> float | None:
    return None if train_value == 0 else val_value / train_value


def classify_shift(
    train_summary: Mapping[str, Any],
    val_summary: Mapping[str, Any],
    slot_rows: Sequence[Mapping[str, Any]],
) -> str:
    total_gap = val_summary["total_imputation_rate"] - train_summary["total_imputation_rate"]
    any_gap = (val_summary["any_imputation_video_rate"]
               - train_summary["any_imputation_video_rate"])
    train_counts = train_summary["imputed_slots_per_video"]
    val_counts = val_summary["imputed_slots_per_video"]
    train_runs = train_summary["max_consecutive_imputation_run"]
    val_runs = val_summary["max_consecutive_imputation_run"]
    slot_gap = max(float(row["val_minus_train_rate"]) for row in slot_rows)

    overall_signals = sum((
        total_gap >= 0.01,
        any_gap >= 0.05,
        val_counts["p95"] >= train_counts["p95"] + 1,
        val_runs["max"] >= train_runs["max"] + 1,
    ))
    if overall_signals >= 3:
        return "POSSIBLE_SEQUENCE_QUALITY_SHIFT"
    if abs(total_gap) < 0.01 and slot_gap >= 0.05:
        return "LOCALIZED_SLOT_SHIFT"
    if (overall_signals > 0 or abs(total_gap) >= 0.005 or abs(any_gap) >= 0.03
            or slot_gap >= 0.03):
        return "MIXED"
    return "NO_OBVIOUS_SEQUENCE_QUALITY_SHIFT"


def build_audit_result(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_summary = summarize_split(train_records)
    val_summary = summarize_split(val_records)
    slots = slot_frequency_rows(train_records, val_records)
    comparison = {
        "any_imputation_video_rate": {
            "train": train_summary["any_imputation_video_rate"],
            "val": val_summary["any_imputation_video_rate"],
            "val_train_ratio": _ratio(
                val_summary["any_imputation_video_rate"],
                train_summary["any_imputation_video_rate"]),
        },
        "total_imputation_rate": {
            "train": train_summary["total_imputation_rate"],
            "val": val_summary["total_imputation_rate"],
            "val_train_ratio": _ratio(
                val_summary["total_imputation_rate"],
                train_summary["total_imputation_rate"]),
        },
        "mean_imputed_slots_per_video": {
            "train": train_summary["imputed_slots_per_video"]["mean"],
            "val": val_summary["imputed_slots_per_video"]["mean"],
            "val_train_ratio": _ratio(
                val_summary["imputed_slots_per_video"]["mean"],
                train_summary["imputed_slots_per_video"]["mean"]),
        },
        "p95_imputed_slots_per_video": {
            "train": train_summary["imputed_slots_per_video"]["p95"],
            "val": val_summary["imputed_slots_per_video"]["p95"],
        },
        "max_imputed_slots_per_video": {
            "train": train_summary["imputed_slots_per_video"]["max"],
            "val": val_summary["imputed_slots_per_video"]["max"],
        },
        "max_consecutive_imputation_run": {
            "train": train_summary["max_consecutive_imputation_run"]["max"],
            "val": val_summary["max_consecutive_imputation_run"]["max"],
        },
    }
    classification = classify_shift(train_summary, val_summary, slots)
    if classification not in SUMMARY_LABELS:
        raise AssertionError(f"unexpected classification: {classification}")
    return ({
        "step": "STEP_6_E1_CONTEXT_SEQUENCE_QUALITY_AUDIT",
        "scope": "TRAIN_VS_VAL_32_SLOT_CONTEXT_SEQUENCE",
        "classification": classification,
        "interpretation": (
            "Descriptive audit only; observed differences are not established causes "
            "of validation-loss behavior."
        ),
        "splits": {"train": train_summary, "val": val_summary},
        "comparison": comparison,
        "protection": {
            "allowed_splits": list(ALLOWED_SPLITS),
            "test_access_count": 0,
            "cnn_feature_read": False,
            "image_read": False,
            "model_used": False,
            "training_run": False,
        },
    }, slots)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_outputs(
    output_dir: Path,
    result: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    video_rows = [{
        "video_id": record["video_id"],
        "split": record["split"],
        "imputed_count": record["imputed_count"],
        "imputed_rate": record["imputed_rate"],
        "max_imputed_run": record["max_imputed_run"],
        "imputed_slots": "|".join(str(slot) for slot in record["imputed_slots"]),
    } for record in records]
    _write_csv(
        output_dir / "video_sequence_quality.csv",
        ("video_id", "split", "imputed_count", "imputed_rate",
         "max_imputed_run", "imputed_slots"),
        video_rows,
    )
    _write_csv(
        output_dir / "slot_imputation_frequency.csv",
        ("slot", "train_imputed_count", "train_imputed_rate",
         "val_imputed_count", "val_imputed_rate", "val_minus_train_rate",
         "val_train_rate_ratio"),
        slots,
    )


def run_context_sequence_quality_audit(
    sequence_root: Path,
    output_dir: Path,
    expected_video_counts: Mapping[str, int] = EXPECTED_VIDEO_COUNTS,
) -> dict[str, Any]:
    sequence_resolved = sequence_root.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == sequence_resolved or sequence_resolved in output_resolved.parents:
        raise ContextSequenceQualityAuditError(
            "audit output은 read-only sequence artifact 내부에 둘 수 없습니다")
    train_records, train_fingerprints = load_split_sequences(
        sequence_root, "train", expected_video_counts.get("train"))
    val_records, val_fingerprints = load_split_sequences(
        sequence_root, "val", expected_video_counts.get("val"))
    result, slots = build_audit_result(train_records, val_records)
    verify_artifacts_unchanged([*train_fingerprints, *val_fingerprints])
    result["protection"]["source_hash_mtime_size_unchanged"] = True
    result["top10"] = {
        split: [
            {key: record[key] for key in (
                "video_id", "imputed_count", "imputed_rate",
                "max_imputed_run", "imputed_slots")}
            for record in sorted(
                (item for item in [*train_records, *val_records] if item["split"] == split),
                key=lambda item: (-item["imputed_count"], -item["max_imputed_run"], item["video_id"]),
            )[:10]
        ]
        for split in ALLOWED_SPLITS
    }
    write_audit_outputs(output_dir, result, [*train_records, *val_records], slots)
    return result
