"""EAR/MAR 수학 구현의 안전성을 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.geometric_features import (  # noqa: E402
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    RIGHT_EYE_INDICES,
    calculate_ear,
    calculate_eye_aspect_ratio,
    calculate_mar,
)


class GeometricFeatureTest(unittest.TestCase):
    """정상, 분모 0, NaN geometry를 검사한다."""

    def setUp(self) -> None:
        """표준 index에 단순하고 유효한 합성 좌표를 배치한다."""

        self.landmarks = np.zeros((68, 2), dtype=np.float64)
        eye = np.asarray([(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)], dtype=np.float64)
        self.landmarks[list(LEFT_EYE_INDICES)] = eye
        self.landmarks[list(RIGHT_EYE_INDICES)] = eye + (10, 0)
        mouth = np.asarray([(0, 0), (1, 1), (2, 2), (3, 1), (4, 0), (3, -1), (2, -2), (1, -1)], dtype=np.float64)
        self.landmarks[list(MOUTH_INDICES)] = mouth

    def test_finite_ear_and_mar(self) -> None:
        """정상 합성 geometry에서 finite ratio를 반환해야 한다."""

        left, right, average = calculate_ear(self.landmarks)
        mar = calculate_mar(self.landmarks)
        self.assertTrue(left.valid and right.valid and average.valid and mar.valid)
        self.assertTrue(np.isfinite([left.value, right.value, average.value, mar.value]).all())

    def test_zero_denominator_is_invalid(self) -> None:
        """가로 span이 0이면 예외 대신 명시적인 invalid를 반환해야 한다."""

        eye = np.zeros((6, 2), dtype=np.float64)
        self.assertEqual(calculate_eye_aspect_ratio(eye).failure_reason, "EAR_INVALID_GEOMETRY")
        self.landmarks[64] = self.landmarks[60]
        self.assertEqual(calculate_mar(self.landmarks).failure_reason, "MAR_INVALID_GEOMETRY")

    def test_nan_is_invalid(self) -> None:
        """NaN 좌표는 EAR/MAR 모두 invalid여야 한다."""

        self.landmarks[36, 0] = np.nan
        self.assertFalse(calculate_ear(self.landmarks)[2].valid)
        self.assertFalse(calculate_mar(self.landmarks).valid)


if __name__ == "__main__":
    unittest.main()

