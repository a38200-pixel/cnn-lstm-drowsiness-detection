"""STEP 2 config, dry-run 범위, CSV writer를 검증한다."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.detector_landmark_audit import (  # noqa: E402
    _render_report,
    _write_csv,
    describe_dry_run,
    load_audit_config,
)


class AuditConfigTest(unittest.TestCase):
    """실제 영상이나 detector를 열지 않는 설정 테스트."""

    def test_config_and_dry_run_exclude_test_split(self) -> None:
        """기본 config가 train/val metadata만 사용해야 한다."""

        config = load_audit_config(
            Path("configs/detector_landmark_audit.yaml"), PROJECT_ROOT
        )
        plan = describe_dry_run(config)
        self.assertEqual(plan["used_splits"], ["train", "val"])
        self.assertFalse(plan["test_split_used"])
        self.assertEqual(plan["train_metadata_rows"], 1452)
        self.assertEqual(plan["val_metadata_rows"], 311)

    def test_csv_writer(self) -> None:
        """결과 writer가 지정한 header와 값을 보존해야 한다."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "result.csv"
            _write_csv(output, ("sample_id", "value"), [{"sample_id": "s1", "value": 1}])
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"sample_id": "s1", "value": "1"}])

    def test_report_writer_sections(self) -> None:
        """audit report가 필수 section 제목을 모두 포함해야 한다."""

        summary = {
            "dataset_scope": {}, "hog": {}, "yunet_fallback": {},
            "combined_face_detection": {}, "dlib68": {}, "ear": {}, "mar": {},
            "head_pose": {}, "paired_detector_check": {}, "timing_ms": {},
            "visual_audit": {}, "decision": "WAITING_FOR_MANUAL_VISUAL_REVIEW",
            "limitations": [],
        }
        report = _render_report(summary)
        for section in (
            "[Dataset Scope]", "[HOG]", "[YuNet Fallback]", "[Dlib68]",
            "[EAR]", "[MAR]", "[Head Pose]", "[Visual Audit]", "[Decision]",
            "[Limitations]",
        ):
            self.assertIn(section, report)


if __name__ == "__main__":
    unittest.main()
