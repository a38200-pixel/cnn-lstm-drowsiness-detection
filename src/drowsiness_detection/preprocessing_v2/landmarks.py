"""검출기 종류와 무관하게 동일한 Dlib68 landmark를 생성한다."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .detectors import BoundingBox


@dataclass(frozen=True)
class LandmarkResult:
    """Dlib68 landmark 예측 결과."""

    success: bool
    points: np.ndarray | None
    elapsed_ms: float
    failure_reason: str = ""


class Dlib68Predictor:
    """모든 detector source에 공통으로 적용되는 Dlib68 predictor."""

    def __init__(self, model_path: str) -> None:
        """Dlib shape predictor asset을 한 번만 불러온다."""

        import dlib

        self._dlib = dlib
        self._predictor = dlib.shape_predictor(model_path)

    def predict(self, frame: np.ndarray, bbox: BoundingBox) -> LandmarkResult:
        """공통 bbox를 dlib.rectangle로 변환해 68개 좌표를 반환한다."""

        started = perf_counter()
        try:
            if not bbox.valid:
                raise ValueError("유효하지 않은 bbox")
            rectangle = self._dlib.rectangle(
                bbox.x1, bbox.y1, bbox.x2 - 1, bbox.y2 - 1
            )
            shape = self._predictor(frame, rectangle)
            points = np.asarray(
                [(shape.part(index).x, shape.part(index).y) for index in range(shape.num_parts)],
                dtype=np.float64,
            )
            if points.shape != (68, 2):
                raise ValueError(f"landmark shape가 (68, 2)가 아님: {points.shape}")
            if not np.isfinite(points).all():
                raise ValueError("landmark 좌표에 NaN 또는 Inf가 있음")
            return LandmarkResult(True, points, (perf_counter() - started) * 1000.0)
        except Exception as exc:
            return LandmarkResult(
                False,
                None,
                (perf_counter() - started) * 1000.0,
                f"LANDMARK_ERROR:{type(exc).__name__}:{exc}",
            )

