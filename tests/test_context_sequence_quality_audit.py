"""STEP 6-E1 Context sequence quality audit의 synthetic 테스트."""

from __future__ import annotations

import pytest

from drowsiness_detection.evaluation_v2.context_sequence_quality_audit import (
    ContextSequenceQualityAuditError,
    analyze_video,
    build_audit_result,
    classify_shift,
    load_split_sequences,
    max_consecutive_true,
    slot_frequency_rows,
    summarize_split,
)


def _mask(*slots: int) -> list[bool]:
    return [index in slots for index in range(32)]


def _records(split: str, masks: list[list[bool]]) -> list[dict]:
    return [analyze_video(f"{split}_{index}", split, mask)
            for index, mask in enumerate(masks)]


def test_video_and_split_metrics_cover_counts_runs_buckets_and_percentiles() -> None:
    records = _records("train", [
        _mask(),
        _mask(0, 1),
        _mask(4, 5, 6, 10, 12),
    ])
    summary = summarize_split(records)
    assert summary["video_count"] == 3
    assert summary["total_target_slots"] == 96
    assert summary["total_imputed_slots"] == 7
    assert summary["original_slots"] == 89
    assert summary["imputed_video_count"] == 2
    assert summary["zero_imputation_video_count"] == 1
    assert summary["imputed_slots_per_video"]["mean"] == pytest.approx(7 / 3)
    assert summary["imputed_slots_per_video"]["median"] == 2.0
    assert summary["imputed_slots_per_video"]["max"] == 5
    assert summary["imputed_slots_per_video"]["buckets"] == {
        "0": 1, "1-2": 1, "3-4": 0, "5-8": 1, "over_8": 0,
    }
    assert summary["max_consecutive_imputation_run"]["max"] == 3
    assert max_consecutive_true(_mask(3, 4, 5, 8, 9)) == 3


def test_anomalies_are_reported_without_modifying_records() -> None:
    mask = _mask(*range(9))
    record = analyze_video("train_bad", "train", mask)
    before = list(record["mask"])
    summary = summarize_split([record])
    assert summary["anomalies"]["imputed_count_over_8"] == ["train_bad"]
    assert summary["anomalies"]["max_run_over_4"] == ["train_bad"]
    assert record["mask"] == before


def test_slot_frequency_and_integrated_result_are_train_val_only() -> None:
    train = _records("train", [_mask(0), _mask()])
    val = _records("val", [_mask(0, 31), _mask(31)])
    result, slots = build_audit_result(train, val)
    assert len(slots) == 32
    assert slots[0]["train_imputed_rate"] == 0.5
    assert slots[0]["val_imputed_rate"] == 0.5
    assert slots[31]["train_imputed_count"] == 0
    assert slots[31]["val_imputed_rate"] == 1.0
    assert result["classification"] in {
        "NO_OBVIOUS_SEQUENCE_QUALITY_SHIFT", "POSSIBLE_SEQUENCE_QUALITY_SHIFT",
        "LOCALIZED_SLOT_SHIFT", "MIXED",
    }
    assert result["protection"]["test_access_count"] == 0
    assert result["protection"]["cnn_feature_read"] is False
    assert result["protection"]["image_read"] is False
    assert result["protection"]["model_used"] is False
    assert result["protection"]["training_run"] is False


def test_classification_labels_cover_similar_localized_and_possible_shift() -> None:
    train = _records("train", [_mask() for _ in range(20)])
    similar_val = _records("val", [_mask() for _ in range(10)])
    train_summary = summarize_split(train)
    similar_summary = summarize_split(similar_val)
    similar_slots = slot_frequency_rows(train, similar_val)
    assert classify_shift(train_summary, similar_summary, similar_slots) == (
        "NO_OBVIOUS_SEQUENCE_QUALITY_SHIFT")

    localized_train = _records("train", [_mask(0) for _ in range(20)])
    localized_val = _records("val", [_mask(31) for _ in range(10)])
    localized_train_summary = summarize_split(localized_train)
    localized_summary = summarize_split(localized_val)
    localized_slots = slot_frequency_rows(localized_train, localized_val)
    assert classify_shift(
        localized_train_summary, localized_summary, localized_slots) == (
        "LOCALIZED_SLOT_SHIFT")

    shifted_val = _records("val", [_mask(0, 1, 2, 3, 4) for _ in range(10)])
    shifted_summary = summarize_split(shifted_val)
    shifted_slots = slot_frequency_rows(train, shifted_val)
    assert classify_shift(train_summary, shifted_summary, shifted_slots) == (
        "POSSIBLE_SEQUENCE_QUALITY_SHIFT")


def test_test_split_and_invalid_sequence_length_are_blocked() -> None:
    with pytest.raises(ContextSequenceQualityAuditError, match="접근 차단"):
        analyze_video("test_1", "test", _mask())
    with pytest.raises(ContextSequenceQualityAuditError, match="접근 차단"):
        load_split_sequences(object(), "test")  # type: ignore[arg-type]
    with pytest.raises(ContextSequenceQualityAuditError, match="sequence length"):
        analyze_video("train_1", "train", [False] * 31)
