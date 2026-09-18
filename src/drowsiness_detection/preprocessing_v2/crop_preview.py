"""두 detector에 동일하게 적용하는 Full Face crop preview를 생성한다."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detectors import BoundingBox


@dataclass(frozen=True)
class CropPreview:
    """RGB crop preview와 padding 및 원본 crop 좌표 정보."""

    rgb: np.ndarray
    requested_bbox: BoundingBox
    padding_fraction: float


def create_face_crop_preview(
    frame_bgr: np.ndarray,
    bbox: BoundingBox,
    margin_ratio: float,
    square_crop: bool,
    output_size: int,
    padding_value: int = 0,
) -> CropPreview:
    """
    bbox 중심을 유지하며 margin을 더하고 frame 밖 영역은 고정값으로 padding한다.

    square_crop=True이면 긴 변을 기준으로 정사각형을 만든다. HOG와 YuNet에
    완전히 같은 정책을 적용하며 결과는 학습 데이터가 아닌 RGB audit preview다.
    """

    if not bbox.valid or output_size <= 0 or margin_ratio < 0:
        raise ValueError("유효한 bbox, output_size, margin_ratio가 필요합니다")
    width = bbox.width * (1.0 + 2.0 * margin_ratio)
    height = bbox.height * (1.0 + 2.0 * margin_ratio)
    if square_crop:
        width = height = max(width, height)
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    requested = BoundingBox(
        int(np.floor(center_x - width / 2.0)), int(np.floor(center_y - height / 2.0)),
        int(np.ceil(center_x + width / 2.0)), int(np.ceil(center_y + height / 2.0)),
    )
    crop_width, crop_height = requested.width, requested.height
    canvas = np.full((crop_height, crop_width, 3), int(padding_value), dtype=np.uint8)
    frame_height, frame_width = frame_bgr.shape[:2]
    source_x1, source_y1 = max(0, requested.x1), max(0, requested.y1)
    source_x2, source_y2 = min(frame_width, requested.x2), min(frame_height, requested.y2)
    copied_area = 0
    if source_x2 > source_x1 and source_y2 > source_y1:
        target_x1, target_y1 = source_x1 - requested.x1, source_y1 - requested.y1
        target_x2 = target_x1 + source_x2 - source_x1
        target_y2 = target_y1 + source_y2 - source_y1
        canvas[target_y1:target_y2, target_x1:target_x2] = frame_bgr[source_y1:source_y2, source_x1:source_x2]
        copied_area = (source_x2 - source_x1) * (source_y2 - source_y1)
    resized = cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    total_area = max(1, crop_width * crop_height)
    return CropPreview(rgb, requested, 1.0 - copied_area / total_area)


def crop_mean_absolute_difference(left_rgb: np.ndarray, right_rgb: np.ndarray) -> float:
    """동일 크기 RGB preview의 [0, 1] 정규화 평균 절대 차이를 계산한다."""

    if left_rgb.shape != right_rgb.shape:
        raise ValueError("crop 비교에는 동일한 shape가 필요합니다")
    return float(np.mean(np.abs(left_rgb.astype(np.float32) - right_rgb.astype(np.float32))) / 255.0)

