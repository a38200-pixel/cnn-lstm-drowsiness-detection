"""STEP 3 고정 시간축과 Context slot을 결정한다."""

from __future__ import annotations

import math


def canonical_timestamps(hz: float = 10.0, duration_sec: float = 10.0, slots: int = 100) -> list[float]:
    """0부터 시작하는 고정 slot을 만들며 실제 영상 길이로 개수를 바꾸지 않는다."""

    if not math.isfinite(hz) or hz <= 0 or not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("유효한 sampling rate와 clip 길이가 필요합니다")
    if slots != 100 or hz != 10.0 or duration_sec != 10.0:
        raise ValueError("STEP 3 정책은 10 Hz × 10초 = 100 slot으로 고정됩니다")
    return [index / hz for index in range(slots)]


def timestamp_to_nearest_frame_index(timestamp_sec: float, fps: float) -> int:
    """가장 가까운 source index를 floor(t×fps+0.5)로 계산한다; banker's rounding은 쓰지 않는다."""

    if not math.isfinite(timestamp_sec) or timestamp_sec < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("timestamp와 FPS가 유효해야 합니다")
    return math.floor(timestamp_sec * fps + 0.5)


def context_indices() -> list[int]:
    """검출 결과와 무관하게 0..99에서 끝점을 포함한 32개 slot을 고른다."""

    result = [math.floor(index * 99 / 31 + 0.5) for index in range(32)]
    assert len(result) == len(set(result)) == 32 and result[0] == 0 and result[-1] == 99
    return result


def evenly_spaced_positions(count: int, selected: int = 5) -> list[int]:
    """정렬된 stratum 전체에 걸쳐 중복 없는 위치를 고른다."""

    if count < selected or selected < 1:
        raise ValueError("stratum에 선택 가능한 영상이 부족합니다")
    if selected == 1:
        return [count // 2]
    return [math.floor(index * (count - 1) / (selected - 1) + 0.5) for index in range(selected)]
