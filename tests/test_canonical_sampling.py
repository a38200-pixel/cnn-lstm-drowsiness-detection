"""실제 영상을 열지 않고 canonical 시간축을 검증한다."""

import math

import pytest

from drowsiness_detection.preprocessing_v2.canonical_sampling import (
    canonical_timestamps, context_indices, evenly_spaced_positions, timestamp_to_nearest_frame_index,
)


def test_100_slots_and_rounding() -> None:
    times = canonical_timestamps()
    assert len(times) == 100
    assert times[0] == 0.0 and times[-1] == 9.9
    assert timestamp_to_nearest_frame_index(0.05, 10) == 1
    assert timestamp_to_nearest_frame_index(9.9, 30) == 297
    assert (timestamp_to_nearest_frame_index(0.1, 24) / 24 - 0.1) * 1000 < 0


@pytest.mark.parametrize("fps", [0, -1, math.nan, math.inf])
def test_invalid_fps(fps: float) -> None:
    with pytest.raises(ValueError):
        timestamp_to_nearest_frame_index(0.1, fps)


def test_context_and_pilot_positions() -> None:
    indices = context_indices()
    assert len(indices) == len(set(indices)) == 32
    assert indices == sorted(indices) and (indices[0], indices[-1]) == (0, 99)
    assert evenly_spaced_positions(10) == [0, 2, 5, 7, 9]
