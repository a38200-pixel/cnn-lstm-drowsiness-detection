"""STEP 2 수동 검토용 annotation과 contact sheet를 생성한다."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .detectors import BoundingBox
from .geometric_features import LEFT_EYE_INDICES, MOUTH_INDICES, RIGHT_EYE_INDICES


def annotate_frame(
    frame: np.ndarray,
    bbox: BoundingBox | None,
    landmarks: np.ndarray | None,
    axis_points: np.ndarray | None,
    values: Mapping[str, object],
) -> np.ndarray:
    """bbox, landmark, pose axis 및 주요 수치를 원본 frame 사본에 표시한다."""

    canvas = frame.copy()
    if bbox is not None:
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
    if landmarks is not None and np.asarray(landmarks).shape == (68, 2):
        points = np.asarray(landmarks, dtype=np.int32)
        eye_indices = set(LEFT_EYE_INDICES + RIGHT_EYE_INDICES)
        mouth_indices = set(MOUTH_INDICES)
        for index, point in enumerate(points):
            color = (0, 255, 255) if index in eye_indices else (255, 0, 255) if index in mouth_indices else (255, 255, 0)
            cv2.circle(canvas, tuple(point), 2, color, -1)
        if axis_points is not None and np.asarray(axis_points).shape == (3, 2):
            origin = tuple(points[30])
            colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
            for endpoint, color in zip(np.asarray(axis_points, dtype=np.int32), colors):
                cv2.line(canvas, origin, tuple(endpoint), color, 2)

    lines = [
        f"{values.get('sample_id', '')} | {values.get('split', '')} | {values.get('label', '')}",
        f"detector={values.get('detector_source', '')} t={values.get('timestamp_sec', '')}",
        f"EAR={_format_value(values.get('ear_mean'))} MAR={_format_value(values.get('mar'))}",
        f"P/Y/R={_format_value(values.get('pitch'))}/{_format_value(values.get('yaw'))}/{_format_value(values.get('roll'))}",
    ]
    for line_number, line in enumerate(lines):
        y = 25 + line_number * 24
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _format_value(value: object) -> str:
    """시각화용 숫자를 짧게 표시한다."""

    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "NA"


def create_contact_sheets(
    image_paths: Sequence[Path],
    output_dir: Path,
    category: str,
    columns: int,
    thumbnail_size: tuple[int, int],
) -> list[Path]:
    """landmark를 식별할 수 있는 크기의 여러 페이지 contact sheet를 만든다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_per_page = 3
    per_page = max(1, columns * rows_per_page)
    saved: list[Path] = []
    thumb_width, thumb_height = thumbnail_size
    for page_index in range(0, len(image_paths), per_page):
        page_paths = image_paths[page_index : page_index + per_page]
        canvas = np.zeros((rows_per_page * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
        for cell_index, path in enumerate(page_paths):
            image = cv2.imread(str(path))
            if image is None:
                continue
            thumbnail = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
            row, column = divmod(cell_index, columns)
            canvas[row * thumb_height : (row + 1) * thumb_height, column * thumb_width : (column + 1) * thumb_width] = thumbnail
        output_path = output_dir / f"{category}_{page_index // per_page + 1:02d}.jpg"
        if not cv2.imwrite(str(output_path), canvas):
            raise OSError(f"contact sheet 저장 실패: {output_path}")
        saved.append(output_path)
    return saved

