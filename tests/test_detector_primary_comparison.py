"""STEP 2-B config, 동일 표본 재사용, report/CSV utility를 검증한다."""

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

from drowsiness_detection.preprocessing_v2.detector_primary_comparison import (  # noqa: E402
    DECISION,
    _render_report,
    _write_csv,
    describe_comparison_dry_run,
    load_comparison_config,
)


class DetectorPrimaryComparisonTest(unittest.TestCase):
    """실제 detector 실행 없이 STEP 2-B 입출력 계약을 검사한다."""

    def test_dry_run_reuses_step2a_frames(self) -> None:
        """STEP 2-A의 160개 frame과 train/val만 재사용해야 한다."""

        config = load_comparison_config(Path("configs/detector_primary_comparison.yaml"), PROJECT_ROOT)
        plan = describe_comparison_dry_run(config)
        self.assertEqual(plan["source_sample_count"], 160)
        self.assertEqual(plan["unique_video_count"], 32)
        self.assertEqual(plan["split_counts"], {"train": 80, "val": 80})
        self.assertFalse(plan["test_split_used"])
        self.assertFalse(plan["new_random_sampling"])

    def test_report_and_csv_writer(self) -> None:
        """Decision을 보존하는 report와 고정 header CSV를 생성한다."""

        summary = {
            "scope": {}, "detector_detection": {}, "outcome_matrix": {}, "landmark": {},
            "ear": {}, "mar": {}, "head_pose": {}, "full_face_crop": {}, "latency": {},
            "visual_review": {}, "decision": DECISION, "limitations": [],
        }
        report = _render_report(summary)
        self.assertIn("[Outcome Matrix]", report)
        self.assertIn(DECISION, report)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            path = Path(temporary_directory) / "result.csv"
            _write_csv(path, ("sample_id", "value"), [{"sample_id": "s1", "value": 1}])
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["sample_id"], "s1")


if __name__ == "__main__":
    unittest.main()

