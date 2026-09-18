"""STEP 2-B 수동 시각 검토 표본 선정 규칙을 검증한다."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drowsiness_detection.preprocessing_v2.review_pack import (
    _manual_rows,
    _outcome,
    _padding_summary,
    select_review_samples,
)


RESULT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_v2"
    / "detector_primary_comparison"
    / "paired_primary_results.csv"
)


def _rows() -> list[dict[str, str]]:
    """완료된 STEP 2-B 비교 행을 읽는다."""

    with RESULT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_selection_keeps_all_detector_failures_and_deduplicates() -> None:
    """필수 detector 결과를 모두 포함하고 sample 중복을 제거해야 한다."""

    rows = _rows()
    selected, _ = select_review_samples(rows)
    selected_ids = [item.row["sample_id"] for item in selected]
    required_ids = {row["sample_id"] for row in rows if _outcome(row) != "BOTH_SUCCESS"}

    assert 30 <= len(selected) <= 45
    assert len(selected_ids) == len(set(selected_ids))
    assert required_ids <= set(selected_ids)
    assert Counter(_outcome(item.row) for item in selected) >= Counter(
        {"YUNET_ONLY": 6, "HOG_ONLY": 1, "BOTH_FAILED": 2}
    )


def test_selection_has_balanced_normal_references_and_merged_reasons() -> None:
    """정상 참조를 균형 있게 넣고 겹치는 이상치 사유를 보존해야 한다."""

    selected, _ = select_review_samples(_rows())
    normal = [item for item in selected if "NORMAL_REFERENCE" in item.reasons]

    assert 8 <= len(normal) <= 12
    assert set((item.row["split"], item.row["label"]) for item in normal) == {
        ("train", "not_drowsy"), ("train", "drowsy"),
        ("val", "not_drowsy"), ("val", "drowsy"),
    }
    assert any(len(item.reasons) > 1 for item in selected)


def test_manual_template_leaves_judgment_fields_blank() -> None:
    """수동 검토 판단 칸은 자동으로 채우지 않아야 한다."""

    manifest = [{
        "review_order": 1,
        "sample_id": "sample",
        "video_id": "video",
        "timestamp_sec": "1.0",
        "comparison_image_path": "image.jpg",
        "selection_reasons": "BOTH_FAILED",
    }]
    row = _manual_rows(manifest)[0]

    assert row["sample_id"] == "sample"
    assert row["preferred_detector"] == ""
    assert row["review_decision"] == ""
    assert row["review_note"] == ""


def test_padding_summary_excludes_failed_detector_rows() -> None:
    """검출 실패로 값이 빈 행을 padding이 0인 crop으로 세지 않아야 한다."""

    summary = _padding_summary(
        [{"padding": ""}, {"padding": "0"}, {"padding": "0.25"}], "padding",
    )

    assert summary["frame_count"] == 2
    assert summary["padding_frame_count"] == 1
