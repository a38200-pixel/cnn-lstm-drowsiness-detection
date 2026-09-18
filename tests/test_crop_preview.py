"""공통 Full Face crop preview와 padding 정책을 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.crop_preview import (  # noqa: E402
    create_face_crop_preview,
    crop_mean_absolute_difference,
)
from drowsiness_detection.preprocessing_v2.detectors import BoundingBox  # noqa: E402


class CropPreviewTest(unittest.TestCase):
    """square crop 크기, 경계 padding, 차이 metric을 검사한다."""

    def test_square_preview_and_padding(self) -> None:
        """frame 경계 밖 margin은 padding하고 RGB 224 크기를 유지해야 한다."""

        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        preview = create_face_crop_preview(frame, BoundingBox(0, 0, 40, 60), 0.25, True, 224)
        self.assertEqual(preview.rgb.shape, (224, 224, 3))
        self.assertGreater(preview.padding_fraction, 0.0)
        self.assertAlmostEqual(crop_mean_absolute_difference(preview.rgb, preview.rgb), 0.0)


if __name__ == "__main__":
    unittest.main()

