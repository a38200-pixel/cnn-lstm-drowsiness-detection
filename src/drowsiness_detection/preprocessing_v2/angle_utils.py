"""Euler angle의 ±180도 wrap-around를 안전하게 비교한다."""

from __future__ import annotations

import math


def wrap_to_180(angle_degrees: float) -> float:
    """유한한 각도를 [-180, 180) 범위의 동등한 각도로 변환한다."""

    angle = float(angle_degrees)
    if not math.isfinite(angle):
        raise ValueError("angle은 유한한 값이어야 합니다")
    wrapped = (angle + 180.0) % 360.0 - 180.0
    # 부동소수점 계산에서 -0.0이 artifact에 남지 않게 정규화한다.
    return 0.0 if abs(wrapped) < 1.0e-12 else wrapped


def circular_angle_difference(first: float, second: float) -> float:
    """
    두 Euler angle 사이의 최소 원형 각도 차이를 [0, 180] 범위로 반환한다.

    단순 절댓값 차이는 +179도와 -179도를 358도 차이로 계산하지만,
    원형 거리를 사용하면 실제 최소 회전인 2도를 반환한다.
    """

    return abs(wrap_to_180(float(first) - float(second)))


def pitch_centered_candidate(pitch_raw: float) -> float:
    """
    raw pitch에서 180도를 뺀 뒤 wrap하여 정면 중심 후보 표현을 만든다.

    180→0, 179→-1, -179→1로 변환한다. 이는 visual audit을 위한 수학적
    후보일 뿐이며 양수/음수와 head-down/head-up의 관계를 확정하지 않는다.
    """

    return wrap_to_180(float(pitch_raw) - 180.0)

