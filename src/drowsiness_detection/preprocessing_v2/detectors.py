"""HOG 및 YuNet 얼굴 검출기와 공통 bbox 처리를 제공한다."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """프레임 좌표계의 배타적 우·하단 좌표를 사용하는 얼굴 영역."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """bbox 너비를 반환한다."""

        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """bbox 높이를 반환한다."""

        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """음수가 되지 않도록 bbox 면적을 반환한다."""

        return max(0, self.width) * max(0, self.height)

    @property
    def valid(self) -> bool:
        """양의 너비와 높이를 가진 bbox인지 반환한다."""

        return self.width > 0 and self.height > 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        """CSV 및 OpenCV에 사용하기 쉬운 tuple로 변환한다."""

        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class FaceCandidate:
    """검출기가 반환한 얼굴 후보 하나."""

    bbox: BoundingBox
    confidence: float | None = None


@dataclass(frozen=True)
class DetectorResult:
    """검출 시도 결과와 선택된 주 얼굴 정보를 보존한다."""

    attempted: bool
    success: bool
    detection_count: int
    selected: FaceCandidate | None
    elapsed_ms: float
    failure_reason: str = ""


def clip_bbox(bbox: BoundingBox, frame_width: int, frame_height: int) -> BoundingBox:
    """bbox를 프레임 경계 안으로 제한한다."""

    return BoundingBox(
        x1=min(max(bbox.x1, 0), frame_width),
        y1=min(max(bbox.y1, 0), frame_height),
        x2=min(max(bbox.x2, 0), frame_width),
        y2=min(max(bbox.y2, 0), frame_height),
    )


def select_largest_face(candidates: Sequence[FaceCandidate]) -> FaceCandidate | None:
    """운전자 중심 영상에서 면적이 가장 큰 얼굴을 주 얼굴로 선택한다."""

    valid = [candidate for candidate in candidates if candidate.bbox.valid]
    if not valid:
        return None
    return max(valid, key=lambda candidate: candidate.bbox.area)


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    """두 bbox의 Intersection over Union을 계산한다."""

    intersection_width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return float(intersection / union) if union > 0 else 0.0


class HogFaceDetector:
    """grayscale 프레임에서 Dlib HOG frontal face detector를 실행한다."""

    def __init__(self, upsample: int) -> None:
        """설정된 pyramid upsample 횟수로 detector를 초기화한다."""

        import dlib

        self._detector = dlib.get_frontal_face_detector()
        self._upsample = upsample

    def detect(self, frame: np.ndarray) -> DetectorResult:
        """모든 얼굴을 검출하고 가장 큰 유효 bbox를 선택한다."""

        started = perf_counter()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rectangles = self._detector(gray, self._upsample)
            height, width = frame.shape[:2]
            candidates = [
                FaceCandidate(
                    clip_bbox(
                        BoundingBox(rect.left(), rect.top(), rect.right() + 1, rect.bottom() + 1),
                        width,
                        height,
                    )
                )
                for rect in rectangles
            ]
            selected = select_largest_face(candidates)
            reason = "" if selected else "HOG_FACE_NOT_FOUND"
            return DetectorResult(
                True,
                selected is not None,
                len(candidates),
                selected,
                (perf_counter() - started) * 1000.0,
                reason,
            )
        except Exception as exc:
            return DetectorResult(
                True,
                False,
                0,
                None,
                (perf_counter() - started) * 1000.0,
                f"HOG_ERROR:{type(exc).__name__}:{exc}",
            )


class YuNetFaceDetector:
    """OpenCV FaceDetectorYN을 이용하는 HOG 실패 전용 fallback detector."""

    def __init__(
        self,
        model_path: str,
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
    ) -> None:
        """YuNet model과 검출 파라미터를 보존한다."""

        self._detector: Any = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, frame: np.ndarray) -> DetectorResult:
        """현재 프레임 크기로 input size를 설정하고 가장 큰 얼굴을 선택한다."""

        started = perf_counter()
        try:
            height, width = frame.shape[:2]
            self._detector.setInputSize((width, height))
            _retval, faces = self._detector.detect(frame)
            candidates: list[FaceCandidate] = []
            if faces is not None:
                for face in faces:
                    x, y, box_width, box_height = face[:4]
                    bbox = clip_bbox(
                        BoundingBox(
                            int(round(x)),
                            int(round(y)),
                            int(round(x + box_width)),
                            int(round(y + box_height)),
                        ),
                        width,
                        height,
                    )
                    candidates.append(FaceCandidate(bbox, float(face[-1])))
            selected = select_largest_face(candidates)
            reason = "" if selected else "YUNET_FACE_NOT_FOUND"
            return DetectorResult(
                True,
                selected is not None,
                len(candidates),
                selected,
                (perf_counter() - started) * 1000.0,
                reason,
            )
        except Exception as exc:
            return DetectorResult(
                True,
                False,
                0,
                None,
                (perf_counter() - started) * 1000.0,
                f"YUNET_ERROR:{type(exc).__name__}:{exc}",
            )

