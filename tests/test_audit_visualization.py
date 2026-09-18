"""annotation과 contact sheet 저장 utility를 검증한다."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.audit_visualization import (  # noqa: E402
    annotate_frame,
    create_contact_sheets,
)
from drowsiness_detection.preprocessing_v2.detectors import BoundingBox  # noqa: E402


class AuditVisualizationTest(unittest.TestCase):
    """시각화 결과의 shape와 파일 생성을 검사한다."""

    def test_annotation_and_contact_sheet(self) -> None:
        """합성 이미지로 annotation과 한 페이지 contact sheet를 생성한다."""

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        landmarks = np.column_stack((np.linspace(80, 240, 68), np.linspace(60, 180, 68)))
        annotated = annotate_frame(
            frame, BoundingBox(60, 40, 260, 210), landmarks, None,
            {"sample_id": "sample", "split": "train", "label": "drowsy", "detector_source": "HOG"},
        )
        self.assertEqual(annotated.shape, frame.shape)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "sample.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), annotated))
            sheets = create_contact_sheets([image_path], root / "sheets", "hog", 2, (160, 120))
            self.assertEqual(len(sheets), 1)
            self.assertTrue(sheets[0].is_file())


if __name__ == "__main__":
    unittest.main()

