"""기존 157개 결과의 clipping 보정과 Dlib ROI 영향 여부를 검증한다."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.clipping_metadata_correction import calculate_correction


def test_original_anomaly_and_corrected_roi_coordinates() -> None:
    """원본 flag 불일치를 재현하고 저장된 모든 ROI 좌표가 동일한지 확인한다."""

    rows, summary = calculate_correction(
        PROJECT_ROOT, PROJECT_ROOT / "outputs" / "preprocessing_v2" / "landmark_roi_margin_audit",
    )

    assert len(rows) == 157 * 4
    assert {key: item["clipped_count"] for key, item in summary["original_statistics"].items()} == {
        "raw": 0, "m05": 157, "m10": 155, "m15": 157,
    }
    assert all(item["clipped_count"] == 0 for item in summary["corrected_statistics"].values())
    assert summary["stored_roi_coordinate_match_count"] == 628
    assert summary["flag_changed_count"] == 469
    assert summary["impact"] == "REPORTING_ONLY_BUG"
    assert summary["roi_audit_rerun_required"] is False
