"""STEP 2-C compact review pack 후처리 규칙을 검증한다."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drowsiness_detection.preprocessing_v2.landmark_roi_review_pack import (
    ReviewSample,
    assign_review_groups,
    compute_review_metrics,
)


RESULT_DIR = PROJECT_ROOT / "outputs" / "preprocessing_v2" / "landmark_roi_margin_audit"


def _selected_samples() -> list[ReviewSample]:
    """기존 수동 검토 35개와 결과 수치를 결합한다."""

    with (RESULT_DIR / "roi_margin_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        results = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with (RESULT_DIR / "roi_margin_manual_review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        selected = list(csv.DictReader(handle))
    return [ReviewSample(results[row["sample_id"]], compute_review_metrics(results[row["sample_id"]])) for row in selected]


def test_existing_selection_is_unique_and_has_matching_images() -> None:
    """새 sampling 없이 기존 35개와 해당 5-panel 이미지만 사용해야 한다."""

    samples = _selected_samples()
    sample_ids = [sample.row["sample_id"] for sample in samples]

    assert len(samples) == 35
    assert len(set(sample_ids)) == 35
    assert all((RESULT_DIR / "visual_samples" / f"{sample_id}.jpg").is_file() for sample_id in sample_ids)


def test_group_assignment_is_deduplicated_and_balanced() -> None:
    """다섯 review sheet에 중복 없이 7개씩 배정해야 한다."""

    assigned = assign_review_groups(_selected_samples())
    counts = Counter(sample.group for sample in assigned)

    assert len({sample.row["sample_id"] for sample in assigned}) == 35
    assert counts == Counter({
        "overview": 7, "eye": 7, "mouth": 7, "pose": 7, "difficult": 7,
    })
    overview = [sample for sample in assigned if sample.group == "overview"]
    assert len({(sample.row["split"], sample.row["label"]) for sample in overview}) == 4


def test_metrics_are_derived_from_existing_columns() -> None:
    """EAR/MAR/landmark/pose 변화량을 기존 CSV 값에서 계산해야 한다."""

    sample = _selected_samples()[0]
    metrics = sample.metrics

    assert metrics["raw_m15_ear_diff"] >= 0
    assert metrics["raw_m15_mar_diff"] >= 0
    assert metrics["raw_m15_landmark_diff"] >= 0
    assert 0 <= metrics["raw_m15_pitch_diff"] <= 180
