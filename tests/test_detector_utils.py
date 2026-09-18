"""bbox 공통 처리와 detector wrapper를 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.detectors import (  # noqa: E402
    BoundingBox,
    FaceCandidate,
    HogFaceDetector,
    YuNetFaceDetector,
    bbox_iou,
    clip_bbox,
    select_largest_face,
)


class _FakeYuNet:
    """asset 없이 YuNet 결과 변환을 검사하기 위한 최소 구현."""

    def setInputSize(self, _size: tuple[int, int]) -> None:
        """실제 wrapper와 같은 호출 표면을 제공한다."""

    def detect(self, _frame: np.ndarray) -> tuple[int, np.ndarray]:
        """크기가 다른 얼굴 두 개를 반환한다."""

        return 1, np.asarray([[1, 1, 10, 10, 0, 0, 0, 0, 0.8], [2, 2, 20, 20, 0, 0, 0, 0, 0.9]], dtype=np.float32)


class DetectorUtilityTest(unittest.TestCase):
    """경계 제한, 가장 큰 얼굴 선택, wrapper smoke test를 수행한다."""

    def test_clip_selection_and_iou(self) -> None:
        """경계를 벗어난 bbox를 제한하고 가장 큰 얼굴을 선택한다."""

        clipped = clip_bbox(BoundingBox(-5, -2, 120, 80), 100, 60)
        self.assertEqual(clipped.as_tuple(), (0, 0, 100, 60))
        selected = select_largest_face([FaceCandidate(BoundingBox(0, 0, 10, 10)), FaceCandidate(BoundingBox(0, 0, 20, 20))])
        self.assertEqual(selected.bbox.area, 400)  # type: ignore[union-attr]
        self.assertAlmostEqual(bbox_iou(clipped, clipped), 1.0)

    def test_hog_smoke(self) -> None:
        """합성 grayscale 입력 경로에서 HOG가 예외 없이 결과를 반환한다."""

        result = HogFaceDetector(upsample=0).detect(np.zeros((80, 80, 3), dtype=np.uint8))
        self.assertTrue(result.attempted)
        self.assertGreaterEqual(result.detection_count, 0)

    def test_yunet_wrapper_with_fake_detector(self) -> None:
        """YuNet wrapper가 가장 큰 face와 confidence를 보존한다."""

        detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
        detector._detector = _FakeYuNet()
        result = detector.detect(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertTrue(result.success)
        self.assertEqual(result.detection_count, 2)
        self.assertEqual(result.selected.bbox.as_tuple(), (2, 2, 22, 22))  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()

