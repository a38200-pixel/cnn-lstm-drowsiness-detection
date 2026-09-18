"""Dlib68 landmark에서 EAR/MAR와 구조적 유효성을 계산한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .detectors import BoundingBox


LEFT_EYE_INDICES = (36, 37, 38, 39, 40, 41)
RIGHT_EYE_INDICES = (42, 43, 44, 45, 46, 47)
MOUTH_INDICES = (60, 61, 62, 63, 64, 65, 66, 67)


@dataclass(frozen=True)
class RatioResult:
    """aspect ratio 값과 수학적 geometry 유효성."""

    value: float | None
    valid: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class GeometryValidation:
    """한 프레임의 bbox, landmark, feature 유효성과 의심 flag."""

    face_valid: bool
    landmark_valid: bool
    geometry_valid: bool
    suspicious: bool
    suspicious_reasons: tuple[str, ...]
    failure_reasons: tuple[str, ...]


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    """두 2차원 점 사이의 Euclidean 거리를 계산한다."""

    return float(np.linalg.norm(first - second))


def calculate_eye_aspect_ratio(
    eye_points: np.ndarray, epsilon: float = 1.0e-6
) -> RatioResult:
    """
    Dlib 눈 landmark 6개로 Eye Aspect Ratio를 계산한다.

    점을 p1부터 p6까지 눈 둘레 순서로 두면 공식은
    (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)이다.
    분모가 epsilon 이하이거나 입력에 NaN/Inf가 있으면 유효하지 않다.
    """

    points = np.asarray(eye_points, dtype=np.float64)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        return RatioResult(None, False, "EAR_INVALID_GEOMETRY")
    denominator = 2.0 * _distance(points[0], points[3])
    if denominator <= epsilon:
        return RatioResult(None, False, "EAR_INVALID_GEOMETRY")
    value = (_distance(points[1], points[5]) + _distance(points[2], points[4])) / denominator
    if not np.isfinite(value):
        return RatioResult(None, False, "EAR_INVALID_GEOMETRY")
    return RatioResult(float(value), True)


def calculate_ear(
    landmarks: np.ndarray, epsilon: float = 1.0e-6
) -> tuple[RatioResult, RatioResult, RatioResult]:
    """Dlib index 36–41과 42–47에서 좌·우 및 평균 EAR을 계산한다."""

    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (68, 2):
        invalid = RatioResult(None, False, "EAR_INVALID_GEOMETRY")
        return invalid, invalid, invalid
    left = calculate_eye_aspect_ratio(points[list(LEFT_EYE_INDICES)], epsilon)
    right = calculate_eye_aspect_ratio(points[list(RIGHT_EYE_INDICES)], epsilon)
    if not left.valid or not right.valid:
        return left, right, RatioResult(None, False, "EAR_INVALID_GEOMETRY")
    mean = (float(left.value) + float(right.value)) / 2.0
    return left, right, RatioResult(mean, True)


def calculate_mar(landmarks: np.ndarray, epsilon: float = 1.0e-6) -> RatioResult:
    """
    Dlib 내부 입 landmark index 60–67로 Mouth Aspect Ratio를 계산한다.

    입 둘레 점 p60..p67에 대해 세 세로 거리
    ||p61-p67||, ||p62-p66||, ||p63-p65||의 합을
    2 * ||p60-p64||로 나눈 표준 MAR 정의를 사용한다. 1차 참조 보고서에서
    동일한 세로 거리 합/가로 span 계열만 확인되어 calibration 값은 재사용하지 않는다.
    """

    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        return RatioResult(None, False, "MAR_INVALID_GEOMETRY")
    mouth = points[list(MOUTH_INDICES)]
    denominator = 2.0 * _distance(mouth[0], mouth[4])
    if denominator <= epsilon:
        return RatioResult(None, False, "MAR_INVALID_GEOMETRY")
    numerator = (
        _distance(mouth[1], mouth[7])
        + _distance(mouth[2], mouth[6])
        + _distance(mouth[3], mouth[5])
    )
    value = numerator / denominator
    if not np.isfinite(value):
        return RatioResult(None, False, "MAR_INVALID_GEOMETRY")
    return RatioResult(float(value), True)


def validate_geometry(
    bbox: BoundingBox,
    landmarks: np.ndarray | None,
    frame_shape: Sequence[int],
    ear: RatioResult,
    mar: RatioResult,
    head_pose_valid: bool,
    config: dict[str, object],
) -> GeometryValidation:
    """값의 의미를 분류하지 않고 수학적·구조적 유효성만 검사한다."""

    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    failures: list[str] = []
    suspicious: list[str] = []
    face_valid = bbox.valid and bbox.x1 < frame_width and bbox.y1 < frame_height
    if not face_valid:
        failures.append("INVALID_BBOX")

    landmark_valid = landmarks is not None and np.asarray(landmarks).shape == (68, 2)
    if landmark_valid:
        points = np.asarray(landmarks, dtype=np.float64)
        landmark_valid = bool(np.isfinite(points).all())
    if not landmark_valid:
        failures.append("INVALID_LANDMARKS")
    else:
        tolerance = float(config["landmark_outside_tolerance_ratio"])
        margin_x = frame_width * tolerance
        margin_y = frame_height * tolerance
        outside = (
            (points[:, 0] < -margin_x)
            | (points[:, 0] > frame_width + margin_x)
            | (points[:, 1] < -margin_y)
            | (points[:, 1] > frame_height + margin_y)
        )
        if bool(outside.any()):
            suspicious.append("LANDMARK_OUTSIDE_FRAME")
            landmark_valid = False
            failures.append("LANDMARK_OUTSIDE_FRAME_TOLERANCE")
        outside_bbox = (
            (points[:, 0] < bbox.x1)
            | (points[:, 0] > bbox.x2)
            | (points[:, 1] < bbox.y1)
            | (points[:, 1] > bbox.y2)
        )
        if bool(outside_bbox.any()):
            # Dlib landmark가 bbox 경계에 가까울 수 있으므로 즉시 제거하지 않고
            # 사람이 annotation을 확인할 수 있게 의심 사례로만 표시한다.
            suspicious.append("LANDMARK_OUTSIDE_BBOX")

    if min(bbox.width, bbox.height) < int(config["suspicious_min_bbox_side"]):
        suspicious.append("SUSPICIOUS_BBOX")
    if ear.valid and ear.value is not None:
        lower, upper = config["suspicious_ear_range"]  # type: ignore[misc]
        if not float(lower) <= ear.value <= float(upper):
            suspicious.append("SUSPICIOUS_EAR_EXTREME")
    if mar.valid and mar.value is not None:
        lower, upper = config["suspicious_mar_range"]  # type: ignore[misc]
        if not float(lower) <= mar.value <= float(upper):
            suspicious.append("SUSPICIOUS_MAR_EXTREME")
    if not ear.valid:
        failures.append(ear.failure_reason)
    if not mar.valid:
        failures.append(mar.failure_reason)
    if not head_pose_valid:
        failures.append("HEAD_POSE_INVALID")
    valid = face_valid and landmark_valid and ear.valid and mar.valid and head_pose_valid
    return GeometryValidation(
        face_valid,
        landmark_valid,
        valid,
        bool(suspicious),
        tuple(dict.fromkeys(suspicious)),
        tuple(dict.fromkeys(failures)),
    )
