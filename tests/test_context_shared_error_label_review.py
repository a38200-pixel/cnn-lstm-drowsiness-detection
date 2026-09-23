"""STEP 6-E6 후보 선정, prefill, split 보호 및 manual validation 테스트."""

from __future__ import annotations

from copy import deepcopy

import pytest

from drowsiness_detection.evaluation_v2.context_shared_error_label_review import (
    ContextSharedErrorLabelReviewError,
    KNOWN_MISMATCH_CANDIDATES,
    build_manual_review_template,
    join_validation_metadata,
    select_both_incorrect_candidates,
    validate_and_summarize_manual_review,
    validate_known_prefill,
)


def _overlap_rows() -> list[dict[str, str]]:
    ids = [*KNOWN_MISMATCH_CANDIDATES, *(f"v_{index:02d}" for index in range(38))]
    return [{
        "category": "both_incorrect",
        "video_id": video_id,
        "label": "drowsy" if index % 2 == 0 or video_id.startswith("d_") else "not_drowsy",
        "resnet18_predicted_label": "not_drowsy",
        "vgg16_predicted_label": "not_drowsy",
        "resnet18_correct": "False",
        "vgg16_correct": "False",
        "resnet18_ce_loss": str(1.0 + index / 100),
        "vgg16_ce_loss": str(1.1 + index / 100),
    } for index, video_id in enumerate(ids)]


def _completed_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected = select_both_incorrect_candidates(_overlap_rows())
    candidates = [{**row, "split": "val", "original_video_path": f"data/raw/{row['video_id']}.mp4"}
                  for row in selected]
    reviews = build_manual_review_template(candidates)
    for row in reviews:
        if row["video_id"] in KNOWN_MISMATCH_CANDIDATES:
            continue
        row.update({
            "frame32_review": "visually_consistent",
            "original_video_review": "drowsy_like" if row["dataset_label"] == "drowsy"
            else "not_drowsy_like",
            "label_consistency": "consistent",
            "audit_status": "reviewed_consistent",
        })
    return candidates, reviews


def test_exact_42_both_incorrect_and_known_four_are_selected() -> None:
    selected = select_both_incorrect_candidates(_overlap_rows())
    assert len(selected) == 42
    assert set(KNOWN_MISMATCH_CANDIDATES).issubset(
        {row["video_id"] for row in selected})


def test_known_four_prefill_is_fixed() -> None:
    candidates, _ = _completed_rows()
    reviews = build_manual_review_template(candidates)
    validate_known_prefill(reviews)
    keyed = {row["video_id"]: row for row in reviews}
    for video_id in KNOWN_MISMATCH_CANDIDATES:
        assert keyed[video_id]["original_video_review"] == "not_drowsy_like"
        assert keyed[video_id]["audit_status"] == "manual_label_mismatch_candidate"


def test_test_split_metadata_is_blocked_before_video_lookup() -> None:
    selected = select_both_incorrect_candidates(_overlap_rows())
    metadata = [{
        "video_id": row["video_id"], "split": "test", "label": row["dataset_label"],
        "video_path": f"data/raw/test/{row['video_id']}.mp4",
    } for row in selected]
    with pytest.raises(ContextSharedErrorLabelReviewError, match="test metadata"):
        join_validation_metadata(".", selected, metadata, require_video_exists=False)


def test_completed_manual_review_summary_and_limitation() -> None:
    candidates, reviews = _completed_rows()
    summary = validate_and_summarize_manual_review(candidates, reviews)
    assert summary["scope"]["validation_sample_count"] == 42
    assert summary["scope"]["representative_of_full_validation"] is False
    assert summary["overall"]["conflicting"]["count"] == 4
    assert summary["manual_label_mismatch_candidate_video_ids"] == sorted(
        KNOWN_MISMATCH_CANDIDATES)
    assert "not representative" in summary["limitations"]["en"]
    assert "대표하지" in summary["limitations"]["ko"]


def test_incomplete_or_inconsistent_manual_review_is_blocked() -> None:
    candidates, reviews = _completed_rows()
    incomplete = deepcopy(reviews)
    incomplete[10]["original_video_review"] = ""
    with pytest.raises(ContextSharedErrorLabelReviewError, match="original_video_review"):
        validate_and_summarize_manual_review(candidates, incomplete)
    inconsistent = deepcopy(reviews)
    inconsistent[10]["audit_status"] = "reviewed_ambiguous"
    with pytest.raises(ContextSharedErrorLabelReviewError, match="불일치"):
        validate_and_summarize_manual_review(candidates, inconsistent)
