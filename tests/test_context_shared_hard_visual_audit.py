"""STEP 6-E5 후보 선정, 보호 및 synthetic contact-sheet smoke 테스트."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drowsiness_detection.evaluation_v2.context_shared_hard_visual_audit import (
    ContextSharedHardVisualAuditError,
    classify_manual_review,
    select_visual_audit_candidates,
)
from drowsiness_detection.sequences_v2.context_sequence_visualizer import (
    VisualizationBlocked,
    render_contact_sheet,
    resolve_sequence_path,
)


def _e4_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    per_sample = []
    hard = []
    for index in range(40):
        label = "drowsy" if index % 2 == 0 else "not_drowsy"
        for backbone_index, backbone in enumerate(("resnet18", "vgg16")):
            loss = (40 - index) / 10 + backbone_index / 100
            per_sample.append({
                "backbone": backbone,
                "video_id": f"v{index:02d}",
                "label": label,
                "predicted_label": label,
                "correct": "True",
                "predicted_class_probability": "0.9",
                "per_sample_ce_loss": str(loss),
                "imputed_count": str(index % 3),
            })
    # 교집합 v00~v12 = 13, backbone별 나머지 7개는 서로 다르게 구성한다.
    sets = {
        "resnet18": [*range(13), *range(13, 20)],
        "vgg16": [*range(13), *range(20, 27)],
    }
    for backbone, indices in sets.items():
        for rank, index in enumerate(indices, start=1):
            hard.append({
                "candidate_type": "loss_top20", "rank": str(rank),
                "backbone": backbone, "video_id": f"v{index:02d}",
            })
    return per_sample, hard


def test_candidate_selection_keeps_exact_hard_and_balanced_control() -> None:
    per_sample, hard = _e4_rows()
    selected = select_visual_audit_candidates(per_sample, hard)
    hard_rows = [row for row in selected if row["group"] == "hard"]
    control = [row for row in selected if row["group"] == "control"]
    assert len(hard_rows) == 13
    assert {row["video_id"] for row in hard_rows} == {f"v{i:02d}" for i in range(13)}
    assert len(control) == 12
    assert sum(row["label"] == "drowsy" for row in control) == 6
    assert sum(row["label"] == "not_drowsy" for row in control) == 6
    assert not ({row["video_id"] for row in hard_rows} & {row["video_id"] for row in control})


def test_wrong_shared_hard_count_is_blocked() -> None:
    per_sample, hard = _e4_rows()
    hard = [row for row in hard if not (
        row["backbone"] == "vgg16" and row["video_id"] == "v12")]
    hard.append({
        "candidate_type": "loss_top20", "rank": "20",
        "backbone": "vgg16", "video_id": "v27",
    })
    with pytest.raises(ContextSharedHardVisualAuditError, match="13개"):
        select_visual_audit_candidates(per_sample, hard)


def test_test_split_is_blocked_before_path_lookup() -> None:
    with pytest.raises(VisualizationBlocked, match="TEST_ACCESS"):
        resolve_sequence_path("unused", "v00", "test")


def test_synthetic_4_by_8_contact_sheet_smoke() -> None:
    rows = []
    images = []
    for index in range(32):
        rows.append({
            "video_id": "synthetic", "split": "val", "label": "drowsy",
            "target_context_index": index,
            "target_canonical_index": index * 3,
            "target_timestamp_sec": index * 0.3,
            "imputed": index >= 28,
            "source_context_index": min(index, 27),
            "source_canonical_index": min(index, 27) * 3,
            "source_image_relpath": f"synthetic/{min(index, 27)}.jpg",
        })
        images.append(np.full((224, 224, 3), index, dtype=np.uint8))
    sheet = render_contact_sheet(rows, images, tile_size=96)
    assert sheet.shape == (4 * (96 + 62), 8 * 96, 3)
    ok, encoded = cv2.imencode(".jpg", sheet)
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == sheet.shape


def test_manual_classification_is_descriptive() -> None:
    def metrics(rate: float) -> dict[str, dict[str, float]]:
        return {name: {"rate": rate} for name in (
            "face_visibility_problem", "major_occlusion", "lighting_issue",
            "extreme_head_pose", "framing_issue", "brief_cue",
            "visually_ambiguous", "label_concern")}
    classification, evidence = classify_manual_review(metrics(0.7), metrics(0.1))
    assert classification == "MIXED"
    assert "descriptive only" in evidence["interpretation"]
