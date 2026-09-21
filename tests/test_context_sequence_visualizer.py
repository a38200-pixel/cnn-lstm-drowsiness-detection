"""STEP 5-A Context sequence read-only visualizer 테스트."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import drowsiness_detection.sequences_v2.context_sequence_visualizer as module


COLUMNS = (
    "video_id", "split", "label", "target_context_index", "target_canonical_index",
    "target_timestamp_sec", "original_available", "imputed", "source_context_index",
    "source_canonical_index", "source_timestamp_sec", "source_image_relpath",
    "canonical_policy_hash", "sequence_missing_policy_hash",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_project(tmp_path: Path, split: str = "train", video_id: str = "n_265",
                     row_count: int = 32) -> tuple[Path, Path, list[Path]]:
    project = tmp_path / "project"
    image_root = project / f"data/interim/preprocessing_v2/canonical/per_video/{split}/{video_id}/context_crops"
    image_root.mkdir(parents=True)
    images = []
    for index in range(24):
        path = image_root / f"ctx_{index:02d}_k{index * 3:03d}.jpg"
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[:, :] = (index, 40, 200)
        assert cv2.imwrite(str(path), image)
        images.append(path)
    rows = []
    for target in range(row_count):
        source = target if target < 24 else 23
        source_relpath = images[source].relative_to(project).as_posix()
        rows.append({
            "video_id": video_id, "split": split, "label": "not_drowsy",
            "target_context_index": target, "target_canonical_index": target * 3,
            "target_timestamp_sec": target * 0.3,
            "original_available": target < 24, "imputed": target >= 24,
            "source_context_index": source, "source_canonical_index": source * 3,
            "source_timestamp_sec": source * 0.3, "source_image_relpath": source_relpath,
            "canonical_policy_hash": module.CANONICAL_POLICY_HASH,
            "sequence_missing_policy_hash": module.SEQUENCE_MISSING_POLICY_HASH,
        })
    sequence = project / f"data/interim/sequences_v2/context/per_video/{split}/{video_id}/sequence.csv"
    sequence.parent.mkdir(parents=True)
    with sequence.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return project, sequence, images


@pytest.mark.parametrize("video_id,split", [("d_10", "train"), ("d_103", "val")])
def test_valid_actual_train_and_val_video_load(video_id, split):
    rows, images, path = module.load_context_sequence(Path("."), video_id, split)
    assert path.as_posix().endswith(f"/{split}/{video_id}/sequence.csv")
    assert len(rows) == len(images) == 32
    assert rows[0]["split"] == split and rows[0]["video_id"] == video_id


def test_valid_sequence_counts_reuse_and_color(tmp_path):
    project, _, _ = _fixture_project(tmp_path)
    rows, images, _ = module.load_context_sequence(project, "n_265")
    summary = module.sequence_summary(rows)
    assert summary["original_count"] == 24
    assert summary["imputed_count"] == 8
    assert summary["unique_source_images"] == 24
    assert summary["imputed_targets"] == list(range(24, 32))
    assert summary["source_reuse"][0]["target_indices"] == list(range(23, 32))
    assert images[0].shape == (224, 224, 3) and images[0].dtype == np.uint8
    pixel = images[0][100, 100]
    assert pixel[0] < 10 and 30 < pixel[1] < 50 and pixel[2] > 190


def test_sequence_requires_exactly_32_rows(tmp_path):
    project, _, _ = _fixture_project(tmp_path, row_count=31)
    with pytest.raises(module.VisualizationBlocked, match="SEQUENCE_LENGTH"):
        module.load_context_sequence(project, "n_265")


def test_target_index_order_validation(tmp_path):
    project, sequence, _ = _fixture_project(tmp_path)
    rows = module._read_csv(sequence)
    rows[10]["target_context_index"] = "11"
    with sequence.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(module.VisualizationBlocked, match="TARGET_ORDER"):
        module.load_context_sequence(project, "n_265")


def test_test_split_is_rejected_without_lookup(tmp_path):
    with pytest.raises(module.VisualizationBlocked, match="VISUALIZATION_BLOCKED_TEST_ACCESS"):
        module.resolve_sequence_path(tmp_path, "n_1", "test")


def test_missing_source_jpeg_rejected(tmp_path):
    project, _, images = _fixture_project(tmp_path)
    images[0].unlink()
    with pytest.raises(FileNotFoundError, match="MISSING_SOURCE"):
        module.load_context_sequence(project, "n_265")


def test_undecodable_jpeg_rejected(tmp_path):
    project, _, images = _fixture_project(tmp_path)
    images[0].write_bytes(b"not a jpeg")
    with pytest.raises(module.VisualizationBlocked, match="UNDECODABLE_JPEG"):
        module.load_context_sequence(project, "n_265")


def test_wrong_image_size_rejected(tmp_path):
    project, _, images = _fixture_project(tmp_path)
    assert cv2.imwrite(str(images[0]), np.zeros((100, 100, 3), dtype=np.uint8))
    with pytest.raises(module.VisualizationBlocked, match="IMAGE_SHAPE"):
        module.load_context_sequence(project, "n_265")


def test_policy_hash_mismatch_rejected(tmp_path):
    project, sequence, _ = _fixture_project(tmp_path)
    rows = module._read_csv(sequence)
    rows[0]["canonical_policy_hash"] = "changed"
    with sequence.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(module.VisualizationBlocked, match="POLICY_MISMATCH"):
        module.load_context_sequence(project, "n_265")


def test_contact_sheet_shape_and_annotation_logic(tmp_path):
    project, _, _ = _fixture_project(tmp_path)
    rows, images, _ = module.load_context_sequence(project, "n_265")
    sheet = module.render_contact_sheet(rows, images, tile_size=180)
    assert sheet.shape == (4 * (180 + 62), 8 * 180, 3)
    original_lines = module.annotation_lines(rows[0], [0])
    imputed_lines = module.annotation_lines(rows[24], list(range(23, 32)))
    assert "ORIGINAL" in original_lines
    assert "IMPUTED" in imputed_lines
    assert any("T24 <- S23" in line for line in imputed_lines)


def test_imputed_pair_sheet_and_no_imputation_case(tmp_path):
    project, _, _ = _fixture_project(tmp_path)
    rows, images, _ = module.load_context_sequence(project, "n_265")
    sheet = module.render_imputed_sheet(rows, images, tile_size=180)
    assert sheet.shape == (8 * (180 + 62), 2 * 180, 3)
    original_rows = [{**row, "imputed": False, "original_available": True} for row in rows]
    empty = module.render_imputed_sheet(original_rows, images, tile_size=180)
    assert empty.shape == (140, 360, 3)


def test_output_filename_and_summary_json_are_deterministic(tmp_path):
    project, _, _ = _fixture_project(tmp_path)
    rows, images, _ = module.load_context_sequence(project, "n_265")
    summary = module.sequence_summary(rows)
    sheet = module.render_contact_sheet(rows, images)
    assert module.output_filename("train", "n_265", "contact") == (
        "train_n_265_context32_contact.jpg")
    image_path, summary_path = module.save_visualization(tmp_path / "output", sheet, summary,
                                                         "contact")
    assert image_path.name == "train_n_265_context32_contact.jpg"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["frames"] == 32 and saved["imputed_count"] == 8
    imputed_sheet = module.render_imputed_sheet(rows, images)
    imputed_path, reused_summary = module.save_visualization(
        tmp_path / "output", imputed_sheet, summary, "imputed")
    assert imputed_path.name == "train_n_265_context32_imputed.jpg"
    assert reused_summary == summary_path


def test_visualization_does_not_modify_sequence_or_jpeg(tmp_path):
    project, sequence, images = _fixture_project(tmp_path)
    before_sequence = _sha256(sequence)
    before_images = {path: _sha256(path) for path in images}
    rows, decoded, _ = module.load_context_sequence(project, "n_265")
    sheet = module.render_contact_sheet(rows, decoded)
    module.save_visualization(tmp_path / "visualization", sheet,
                              module.sequence_summary(rows), "contact")
    assert _sha256(sequence) == before_sequence
    assert {path: _sha256(path) for path in images} == before_images


def test_visualizer_has_no_cnn_feature_or_behavior_dependency():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "features.npy" not in source
    assert "context_full" not in source
    assert "EAR" not in source and "MAR" not in source and "head pose" not in source
