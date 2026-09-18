"""Head Pose solvePnP 호출과 invalid 처리를 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.head_pose import (  # noqa: E402
    MODEL_POINTS,
    POSE_LANDMARK_INDICES,
    build_camera_matrix,
    estimate_head_pose,
)


class HeadPoseTest(unittest.TestCase):
    """합성 투영점에서 finite pose가 생성되는지 확인한다."""

    def test_synthetic_projection(self) -> None:
        """알려진 3D 투영점으로 solvePnP smoke test를 수행한다."""

        width, height = 640, 480
        camera = build_camera_matrix(width, height)
        projected, _ = cv2.projectPoints(
            MODEL_POINTS,
            np.asarray([[0.05], [0.1], [-0.02]], dtype=np.float64),
            np.asarray([[0.0], [0.0], [1000.0]], dtype=np.float64),
            camera,
            np.zeros((4, 1), dtype=np.float64),
        )
        landmarks = np.zeros((68, 2), dtype=np.float64)
        landmarks[list(POSE_LANDMARK_INDICES)] = projected.reshape(-1, 2)
        result = estimate_head_pose(landmarks, width, height)
        self.assertTrue(result.success, result.failure_reason)
        self.assertTrue(np.isfinite([result.pitch, result.yaw, result.roll]).all())

    def test_invalid_input(self) -> None:
        """잘못된 landmark shape는 예외 전파 대신 실패 결과를 반환한다."""

        result = estimate_head_pose(np.zeros((6, 2)), 640, 480)
        self.assertFalse(result.success)
        self.assertIn("HEAD_POSE_ERROR", result.failure_reason)


if __name__ == "__main__":
    unittest.main()

