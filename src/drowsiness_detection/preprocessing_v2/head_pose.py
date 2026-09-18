"""Dlib68 landmark와 OpenCV solvePnP로 근사 head pose를 계산한다."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


POSE_LANDMARK_INDICES = (30, 8, 36, 45, 48, 54)

# 일반적인 3D 얼굴 모형의 코끝, 턱, 좌·우 눈꼬리, 좌·우 입꼬리 좌표다.
# x는 얼굴의 오른쪽, y는 위쪽, z는 얼굴 앞쪽을 향하는 모형 좌표로 해석한다.
MODEL_POINTS = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPoseResult:
    """solvePnP 성공 여부, Euler angle, pose axis 투영 결과."""

    success: bool
    pitch: float | None
    yaw: float | None
    roll: float | None
    axis_points: np.ndarray | None
    elapsed_ms: float
    failure_reason: str = ""


def build_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    """
    프레임 크기만으로 근사 camera matrix를 구성한다.

    focal length는 frame width와 같다고 가정하고 optical center는 영상 중심으로
    둔다. 실제 camera calibration이 아니므로 결과 angle은 상대 변화 분석 후보이며
    절대적인 ground-truth pose로 해석하지 않는다.
    """

    focal_length = float(frame_width)
    return np.asarray(
        [
            (focal_length, 0.0, frame_width / 2.0),
            (0.0, focal_length, frame_height / 2.0),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )


def estimate_head_pose(landmarks: np.ndarray, frame_width: int, frame_height: int) -> HeadPoseResult:
    """
    Dlib index 30, 8, 36, 45, 48, 54를 이용해 pitch/yaw/roll을 추정한다.

    distortion은 0으로 가정한다. Euler angle은 OpenCV의
    decomposeProjectionMatrix convention을 따르며 축 부호는 camera와 모형 좌표
    정의에 의존하므로 visual audit의 pose axis와 함께 수동 검증해야 한다.
    """

    started = perf_counter()
    try:
        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (68, 2) or not np.isfinite(points).all():
            raise ValueError("유효한 (68, 2) landmark가 필요함")
        image_points = points[list(POSE_LANDMARK_INDICES)]
        camera_matrix = build_camera_matrix(frame_width, frame_height)
        distortion = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise ValueError("solvePnP가 False를 반환함")
        rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)
        projection = np.hstack((rotation_matrix, translation_vector))
        angles = cv2.decomposeProjectionMatrix(projection)[6].reshape(-1)
        if angles.size < 3 or not np.isfinite(angles[:3]).all():
            raise ValueError("Euler angle이 유효하지 않음")

        axis_3d = np.asarray(
            [(100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (0.0, 0.0, 100.0)],
            dtype=np.float64,
        )
        projected, _jacobian = cv2.projectPoints(
            axis_3d, rotation_vector, translation_vector, camera_matrix, distortion
        )
        return HeadPoseResult(
            True,
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
            projected.reshape(-1, 2),
            (perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return HeadPoseResult(
            False,
            None,
            None,
            None,
            None,
            (perf_counter() - started) * 1000.0,
            f"HEAD_POSE_ERROR:{type(exc).__name__}:{exc}",
        )

