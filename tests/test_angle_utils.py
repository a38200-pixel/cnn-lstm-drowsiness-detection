"""Euler angle wrap과 pitch centered 후보 표현을 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.angle_utils import (  # noqa: E402
    circular_angle_difference,
    pitch_centered_candidate,
    wrap_to_180,
)


class AngleUtilityTest(unittest.TestCase):
    """±180도 경계의 최소 원형 거리를 검사한다."""

    def test_circular_difference(self) -> None:
        """wrap 경계에서도 실제 최소 회전 차이를 반환해야 한다."""

        cases = ((179, -179, 2), (-179, 179, 2), (10, 20, 10), (-170, 170, 20), (0, 180, 180))
        for first, second, expected in cases:
            with self.subTest(first=first, second=second):
                self.assertAlmostEqual(circular_angle_difference(first, second), expected)

    def test_pitch_centered_candidate(self) -> None:
        """정면 중심 후보 변환의 수학적 convention을 고정한다."""

        self.assertAlmostEqual(pitch_centered_candidate(180), 0)
        self.assertAlmostEqual(pitch_centered_candidate(179), -1)
        self.assertAlmostEqual(pitch_centered_candidate(-179), 1)
        self.assertAlmostEqual(wrap_to_180(540), -180)


if __name__ == "__main__":
    unittest.main()

