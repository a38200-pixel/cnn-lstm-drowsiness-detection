"""STEP 5-B Behavior sequence read-only visualizer 테스트."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

import drowsiness_detection.sequences_v2.behavior_sequence_visualizer as module


COLUMNS = (
    "video_id", "split", "label", "canonical_index", "target_timestamp_sec",
    "yunet_success", "landmark_success", "pose_success", "behavior_valid",
    *module.FEATURE_NAMES, "canonical_policy_hash", "sequence_missing_policy_hash",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_marker(bundle: Path) -> None:
    marker_path = bundle / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["file_sha256"] = {name: _sha256(bundle / name)
                             for name in ("sequence.csv", "behavior.npz", "summary.json")}
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _fixture_project(tmp_path: Path, split: str = "train", video_id: str = "n_265",
                     feature_shape: tuple[int, int] = (100, 6),
                     feature_names: list[str] | None = None) -> tuple[Path, Path]:
    project = tmp_path / "project"
    bundle = project / f"data/interim/sequences_v2/behavior/per_video/{split}/{video_id}"
    bundle.mkdir(parents=True)
    features = np.linspace(0.1, 6.0, num=feature_shape[0] * feature_shape[1],
                           dtype=np.float32).reshape(feature_shape)
    detector = np.ones(100, dtype=np.bool_)
    landmark = np.ones(100, dtype=np.bool_)
    pose_success = np.ones(100, dtype=np.bool_)
    if feature_shape == (100, 6):
        features[10:13, :2] = np.nan
        detector[10:13] = False
        landmark[10:13] = False
        pose_success[10:13] = False
        feature_valid = np.isfinite(features)
        behavior = detector & landmark & feature_valid[:, :2].all(axis=1)
        pose = pose_success & feature_valid[:, 2:].all(axis=1)
    else:
        feature_valid = np.isfinite(features)
        behavior = np.ones(100, dtype=np.bool_)
        pose = np.ones(100, dtype=np.bool_)
    np.savez_compressed(
        bundle / "behavior.npz", canonical_index=np.arange(100, dtype=np.int16),
        target_timestamp_sec=np.arange(100, dtype=np.float64) / 10.0,
        behavior_features=features, behavior_valid_mask=behavior,
        detector_valid_mask=detector, landmark_valid_mask=landmark,
        pose_valid_mask=pose, feature_valid_mask=feature_valid,
    )
    rows = []
    for index in range(100):
        values = features[index] if features.shape[1] >= 6 else np.full(6, np.nan)
        rows.append({
            "video_id": video_id, "split": split, "label": "not_drowsy",
            "canonical_index": index, "target_timestamp_sec": index / 10.0,
            "yunet_success": bool(detector[index]),
            "landmark_success": bool(landmark[index]),
            "pose_success": bool(pose_success[index]),
            "behavior_valid": bool(behavior[index]),
            **{name: float(values[column]) for column, name in enumerate(module.FEATURE_NAMES)},
            "canonical_policy_hash": module.POLICY_HASH,
            "sequence_missing_policy_hash": module.SEQUENCE_POLICY_HASH,
        })
    with (bundle / "sequence.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "video_id": video_id, "split": split, "label": "not_drowsy",
        "sequence_length": 100, "behavior_valid_count": int(behavior.sum()),
        "behavior_missing_count": int((~behavior).sum()),
        "longest_yunet_missing_run": 3,
        "feature_names": feature_names or list(module.FEATURE_NAMES),
        "feature_shape": list(feature_shape),
        "behavior_feature_schema_hash": module.FEATURE_SCHEMA_HASH,
        "source_metadata_fingerprint": "fixture-fingerprint",
        "canonical_policy_hash": module.POLICY_HASH,
        "sequence_missing_policy_hash": module.SEQUENCE_POLICY_HASH,
    }
    (bundle / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    marker = {
        "video_id": video_id, "source_metadata_fingerprint": "fixture-fingerprint",
        "behavior_feature_schema_hash": module.FEATURE_SCHEMA_HASH,
        "canonical_policy_hash": module.POLICY_HASH,
        "sequence_missing_policy_hash": module.SEQUENCE_POLICY_HASH, "file_sha256": {},
    }
    (bundle / "COMPLETE.json").write_text(json.dumps(marker, indent=2) + "\n",
                                          encoding="utf-8")
    _refresh_marker(bundle)
    return project, bundle


@pytest.mark.parametrize("video_id,split", [("d_10", "train"), ("d_103", "val")])
def test_valid_actual_train_and_val_load(video_id: str, split: str) -> None:
    sequence = module.load_behavior_sequence(Path("."), video_id, split)
    assert sequence.features.shape == (100, 6)
    assert sequence.split == split


def test_resolve_without_split_uses_train_or_val(tmp_path: Path) -> None:
    project, bundle = _fixture_project(tmp_path)
    assert module.resolve_behavior_bundle(project, "n_265") == bundle


def test_test_split_rejected_before_lookup(tmp_path: Path) -> None:
    with pytest.raises(module.BehaviorVisualizationBlocked, match="BLOCKED_TEST_ACCESS"):
        module.resolve_behavior_bundle(tmp_path, "n_1", "test")


def test_ambiguous_video_rejected(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    (project / "data/interim/sequences_v2/behavior/per_video/val/n_265").mkdir(parents=True)
    with pytest.raises(module.BehaviorVisualizationBlocked, match="AMBIGUOUS_VIDEO"):
        module.resolve_behavior_bundle(project, "n_265")


def test_missing_video_rejected(tmp_path: Path) -> None:
    with pytest.raises(module.BehaviorVisualizationBlocked, match="VIDEO_NOT_FOUND"):
        module.resolve_behavior_bundle(tmp_path, "missing")


def test_shape_validation(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path, feature_shape=(100, 5))
    with pytest.raises(module.BehaviorVisualizationBlocked, match="SHAPE_MISMATCH"):
        module.load_behavior_sequence(project, "n_265")


def test_feature_order_validation(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path, feature_names=list(reversed(module.FEATURE_NAMES)))
    with pytest.raises(module.BehaviorVisualizationBlocked, match="FEATURE_ORDER_MISMATCH"):
        module.load_behavior_sequence(project, "n_265")


def test_timeline_is_exactly_zero_to_nine_point_nine() -> None:
    timeline = module.timeline_seconds()
    assert timeline.shape == (100,)
    assert timeline[0] == 0.0 and timeline[-1] == 9.9
    assert np.allclose(np.diff(timeline), 0.1)


def test_nan_is_preserved_without_fill(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    sequence = module.load_behavior_sequence(project, "n_265")
    assert np.isnan(sequence.features[10:13, :2]).all()
    assert np.isfinite(sequence.features[9, :2]).all()


def test_infinity_is_numeric_anomaly(tmp_path: Path) -> None:
    project, bundle = _fixture_project(tmp_path)
    with np.load(bundle / "behavior.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["behavior_features"] = arrays["behavior_features"].copy()
    arrays["behavior_features"][0, 0] = np.inf
    np.savez_compressed(bundle / "behavior.npz", **arrays)
    _refresh_marker(bundle)
    with pytest.raises(module.BehaviorVisualizationBlocked, match="NUMERIC_ANOMALY"):
        module.load_behavior_sequence(project, "n_265")


def test_valid_missing_counts(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    summary = module.sequence_summary(module.load_behavior_sequence(project, "n_265"))
    assert summary["valid_count"] == 97 and summary["missing_count"] == 3


def test_longest_missing_run_calculation(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    summary = module.sequence_summary(module.load_behavior_sequence(project, "n_265"))
    assert summary["longest_missing_run"] == 3
    assert module.false_runs(np.r_[np.ones(97, bool), np.zeros(3, bool)]) == [(97, 99)]


@pytest.mark.parametrize("name", ["detector_valid_mask", "landmark_valid_mask",
                                   "pose_valid_mask", "behavior_valid_mask"])
def test_each_mask_shape_and_dtype(tmp_path: Path, name: str) -> None:
    project, _ = _fixture_project(tmp_path)
    mask = getattr(module.load_behavior_sequence(project, "n_265"), name)
    assert mask.shape == (100,) and mask.dtype == np.bool_


def test_mask_summary(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    summary = module.sequence_summary(module.load_behavior_sequence(project, "n_265"))
    assert summary["mask_counts"]["Behavior valid"] == {"valid": 97, "invalid": 3}


def test_feature_valid_mask_mismatch_rejected(tmp_path: Path) -> None:
    project, bundle = _fixture_project(tmp_path)
    with np.load(bundle / "behavior.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["feature_valid_mask"] = arrays["feature_valid_mask"].copy()
    arrays["feature_valid_mask"][0, 0] = False
    np.savez_compressed(bundle / "behavior.npz", **arrays)
    _refresh_marker(bundle)
    with pytest.raises(module.BehaviorVisualizationBlocked, match="MASK_MISMATCH"):
        module.load_behavior_sequence(project, "n_265")


def test_overview_figure_has_five_panels(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    figure = module.render_overview(module.load_behavior_sequence(project, "n_265"))
    assert len(figure.axes) == 5


def test_masks_figure_creation(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    figure = module.render_masks(module.load_behavior_sequence(project, "n_265"))
    assert len(figure.axes) == 1
    assert "white=valid" in figure.axes[0].get_title()


def test_rendering_does_not_interpolate_nan(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    sequence = module.load_behavior_sequence(project, "n_265")
    before = sequence.features.copy()
    module.render_overview(sequence)
    assert np.array_equal(sequence.features, before, equal_nan=True)


def test_deterministic_output_filename() -> None:
    assert module.output_filename("train", "n_265", "overview") == (
        "train_n_265_behavior100_overview.png")
    assert module.output_filename("val", "d_103", "masks") == (
        "val_d_103_behavior100_masks.png")


def test_summary_json_and_image_save(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    sequence = module.load_behavior_sequence(project, "n_265")
    summary = module.sequence_summary(sequence)
    figure = module.render_overview(sequence)
    image, output = module.save_visualization(
        project, Path("outputs/visualizations/behavior_sequences"), figure, summary, "overview")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert image.name == "train_n_265_behavior100_overview.png"
    assert saved["sequence_shape"] == [100, 6] and saved["missing_slots"] == [10, 11, 12]


def test_source_artifact_immutability(tmp_path: Path) -> None:
    project, bundle = _fixture_project(tmp_path)
    before = {path.name: _sha256(path) for path in bundle.iterdir() if path.is_file()}
    sequence = module.load_behavior_sequence(project, "n_265")
    figure = module.render_masks(sequence)
    module.save_visualization(project, Path("outputs/visualizations/behavior_sequences"),
                              figure, module.sequence_summary(sequence), "masks")
    after = {path.name: _sha256(path) for path in bundle.iterdir() if path.is_file()}
    assert after == before


def test_output_must_be_under_visualization_root(tmp_path: Path) -> None:
    project, _ = _fixture_project(tmp_path)
    with pytest.raises(module.BehaviorVisualizationBlocked, match="OUTSIDE_LOCAL_ROOT"):
        module.ensure_visualization_output(project, tmp_path / "elsewhere")


def test_hash_mismatch_detection(tmp_path: Path) -> None:
    project, bundle = _fixture_project(tmp_path)
    (bundle / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(module.BehaviorVisualizationBlocked, match="PROVENANCE_MISMATCH"):
        module.load_behavior_sequence(project, "n_265")


def test_numeric_summary_does_not_invent_values_for_all_nan() -> None:
    summary = module.feature_numeric_summary(np.full(100, np.nan, dtype=np.float32))
    assert summary == {"finite_count": 0, "nan_count": 100, "min": None,
                       "max": None, "mean": None, "median": None}


def test_suggest_samples_reads_train_val_only(tmp_path: Path) -> None:
    project = tmp_path / "project"
    path = project / "data/interim/sequences_v2/behavior/metadata/behavior_sequence_videos.csv"
    path.parent.mkdir(parents=True)
    columns = ("video_id", "split", "label", "behavior_valid_count",
               "longest_yunet_missing_run")
    rows = [
        {"video_id": "a", "split": "train", "label": "drowsy",
         "behavior_valid_count": 100, "longest_yunet_missing_run": 0},
        {"video_id": "b", "split": "val", "label": "not_drowsy",
         "behavior_valid_count": 99, "longest_yunet_missing_run": 1},
        {"video_id": "c", "split": "train", "label": "not_drowsy",
         "behavior_valid_count": 95, "longest_yunet_missing_run": 4},
        {"video_id": "d", "split": "val", "label": "drowsy",
         "behavior_valid_count": 96, "longest_yunet_missing_run": 5},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    assert [item["category"] for item in module.suggest_samples(project)] == [
        "FULL_VALID", "VALID_99", "LOWEST_VALID", "LONGEST_MISSING_RUN"]


def test_visualizer_has_no_context_or_cnn_artifact_access() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "context_full" not in source
    assert "features.npy" not in source
    assert "data/interim/sequences_v2/context" not in source
