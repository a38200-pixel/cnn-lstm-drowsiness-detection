"""STEP 5-D1 transform·512D extractor contract·pilot·cache·blocker 테스트."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import drowsiness_detection.sequences_v2.cnn_feature_extractor as module
from drowsiness_detection.sequences_v2.cnn_feature_extractor import (
    DependencyBlocked, PretrainedWeightsBlocked, _artifact_complete,
    adapt_resnet_feature_map, adapt_vgg_feature_map, assemble_target_features,
    cnn_feature_policy_hash, decode_rgb_jpeg, environment_preflight, feature_numeric_summary,
    feature_policy, load_config, normalize_rgb_uint8, pilot_summary, require_dependencies,
    select_pilot, unique_source_index, write_blocked_report, write_pilot_manifest,
)


def test_config_semantics():
    config = load_config(module.Path("configs/context_cnn_features.yaml"))
    assert config["geometry_transform"] == "none"
    assert config["augmentation"] == "none"
    assert config["inference"] == {"dtype": "float32", "amp": False,
                                    "gradient": False, "model_mode": "eval"}


def test_normalization_shape_dtype_and_no_geometry_change():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    output = normalize_rgb_uint8(image)
    assert output.shape == (3, 224, 224)
    assert output.dtype == np.float32
    assert np.allclose(output[:, 0, 0], -module.IMAGENET_MEAN[:, 0, 0] /
                       module.IMAGENET_STD[:, 0, 0])


@pytest.mark.parametrize("shape,dtype", [((223, 224, 3), np.uint8),
                                          ((224, 224, 3), np.float32),
                                          ((224, 224, 1), np.uint8)])
def test_transform_rejects_wrong_geometry_or_dtype(shape, dtype):
    with pytest.raises(ValueError, match="expected RGB"):
        normalize_rgb_uint8(np.zeros(shape, dtype=dtype))


def test_opencv_loader_explicit_rgb(tmp_path):
    bgr = np.zeros((224, 224, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    path = tmp_path / "red.jpg"
    assert cv2.imwrite(str(path), bgr)
    rgb = decode_rgb_jpeg(path)
    assert rgb[100, 100, 0] > 240 and rgb[100, 100, 2] < 10


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_resnet18_gap_output_is_512(batch):
    output = adapt_resnet_feature_map(np.zeros((batch, 512, 1, 1), dtype=np.float32))
    assert output.shape == (batch, 512)


@pytest.mark.parametrize("wrong", [(2, 1000), (2, 25088), (2, 4096)])
def test_resnet_rejects_classifier_or_wrong_feature(wrong):
    with pytest.raises(ValueError, match="512D GAP"):
        adapt_resnet_feature_map(np.zeros(wrong, dtype=np.float32))


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_vgg16_features_adaptive_gap_output_is_512(batch):
    output = adapt_vgg_feature_map(np.ones((batch, 512, 7, 7), dtype=np.float32))
    assert output.shape == (batch, 512)
    assert np.all(output == 1)


@pytest.mark.parametrize("wrong", [(2, 25088), (2, 4096), (2, 1000), (2, 256, 7, 7)])
def test_vgg_rejects_flatten_classifier_logits_and_wrong_channels(wrong):
    with pytest.raises(ValueError, match="512-channel"):
        adapt_vgg_feature_map(np.zeros(wrong, dtype=np.float32))


def test_backbone_policy_hashes_are_separate_and_deterministic():
    config = load_config(module.Path("configs/context_cnn_features.yaml"))
    resnet = cnn_feature_policy_hash(config, "resnet18", "ResNet18_Weights.IMAGENET1K_V1")
    assert resnet == cnn_feature_policy_hash(config, "resnet18", "ResNet18_Weights.IMAGENET1K_V1")
    assert resnet != cnn_feature_policy_hash(config, "vgg16", "VGG16_Weights.IMAGENET1K_V1")


def test_feature_policy_excludes_machine_run_settings():
    config = load_config(module.Path("configs/context_cnn_features.yaml"))
    policy = feature_policy(config, "resnet18", "ResNet18_Weights.IMAGENET1K_V1")
    assert not {"device", "batch_size", "gpu", "created_at"} & set(policy)
    assert policy["output_dim"] == 512 and policy["frozen_pretrained"]


def synthetic_universe():
    strata = [("train", "drowsy"), ("train", "not_drowsy"),
              ("val", "drowsy"), ("val", "not_drowsy")]
    rows = []
    for index in range(1669):
        split, label = strata[index % 4]
        rows.append({"video_id": f"n{index:04d}", "split": split, "label": label,
                     "imputed_slots": "0"})
    for index in range(8):
        split, label = strata[index % 4]
        rows.append({"video_id": f"i{index:04d}", "split": split, "label": label,
                     "imputed_slots": str(8 - index)})
    return rows


def test_pilot_selection_deterministic_stratified(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "_read_csv", lambda _: synthetic_universe())
    first, second = select_pilot(tmp_path), select_pilot(tmp_path)
    assert first == second and len(first) == 16
    summary = pilot_summary(first)
    assert summary["no_imputation"] == summary["imputation_required"] == 8
    assert summary["train"] > 0 and summary["val"] > 0
    assert summary["drowsy"] > 0 and summary["not_drowsy"] > 0
    assert summary["test_rows"] == 0 and summary["target_rows_per_backbone"] == 512


def test_pilot_rejects_test(monkeypatch, tmp_path):
    rows = synthetic_universe()
    rows[0]["split"] = "test"
    monkeypatch.setattr(module, "_read_csv", lambda _: rows)
    with pytest.raises(ValueError, match="TEST_LEAKAGE"):
        select_pilot(tmp_path)


def test_source_deduplication_and_target_mapping():
    mappings = [{"source_image_relpath": "b.jpg"}, {"source_image_relpath": "a.jpg"},
                {"source_image_relpath": "b.jpg"}]
    sources, ids = unique_source_index(mappings)
    assert sources == ["a.jpg", "b.jpg"] and ids == [1, 0, 1]


def test_imputed_source_reuse_assembles_exact_features():
    source = np.arange(2 * 512, dtype=np.float32).reshape(2, 512)
    ids = [0] * 16 + [1] * 16
    result = assemble_target_features(source, ids, 1)
    assert result.shape == (1, 32, 512) and result.dtype == np.float32
    assert np.array_equal(result[0, 0], result[0, 15])
    assert np.array_equal(result[0, 16], result[0, 31])


@pytest.mark.parametrize("kind", ["nan", "inf", "zero"])
def test_feature_numeric_anomaly_detection(kind):
    array = np.ones((1, 32, 512), dtype=np.float32)
    if kind == "nan":
        array[0, 0, 0] = np.nan
    elif kind == "inf":
        array[0, 0, 0] = np.inf
    else:
        array[0, 0] = 0
    summary = feature_numeric_summary(array)
    assert summary[{"nan": "nan_count", "inf": "inf_count", "zero": "all_zero_count"}[kind]] > 0


def test_dependency_blocker_lists_missing_packages():
    with pytest.raises(DependencyBlocked, match="torch, torchvision"):
        require_dependencies({"torch_import": False, "torchvision_import": False})


def test_current_preflight_does_not_install_dependencies():
    environment = environment_preflight()
    assert environment["status"] in {"PASS", "STEP_5D1_BLOCKED_DEPENDENCY"}


def test_random_weight_fallback_rejected(monkeypatch):
    monkeypatch.setattr(module, "environment_preflight",
                        lambda: {"torch_import": True, "torchvision_import": True})
    fake_torch = SimpleNamespace()
    fake_weights = SimpleNamespace(DEFAULT=object())
    fake_models = SimpleNamespace(ResNet18_Weights=fake_weights,
                                  resnet18=lambda **_: (_ for _ in ()).throw(RuntimeError("no weights")))
    fake_nn = SimpleNamespace()
    def fake_import(name):
        return {"torch": fake_torch, "torchvision.models": fake_models,
                "torch.nn": fake_nn}[name]
    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    with pytest.raises(PretrainedWeightsBlocked, match="BLOCKED_PRETRAINED_WEIGHTS"):
        module.build_torch_extractor("resnet18", "cpu")


def test_resume_marker_requires_same_backbone_policy_and_pilot(tmp_path):
    root = tmp_path / "features"
    marker = root / "resnet18/COMPLETE.json"
    marker.parent.mkdir(parents=True)
    pilot = [{"video_id": "v1"}]
    marker.write_text(json.dumps({"backbone": "resnet18", "cnn_feature_policy_hash": "hash",
                                  "canonical_policy_hash": module.POLICY_HASH,
                                  "sequence_missing_policy_hash": module.SEQUENCE_POLICY_HASH,
                                  "pilot_video_ids": ["v1"]}), encoding="utf-8")
    assert _artifact_complete(root, "resnet18", "hash", pilot)
    assert not _artifact_complete(root, "vgg16", "hash", pilot)
    assert not _artifact_complete(root, "resnet18", "wrong", pilot)


def test_manifest_deterministic_no_overwrite(tmp_path):
    rows = [{key: value for key, value in zip(module.PILOT_COLUMNS,
            (0, "v1", "train", "drowsy", 0, "path", "reason"))}]
    path = tmp_path / "manifest.csv"
    write_pilot_manifest(path, rows)
    before = path.read_bytes()
    write_pilot_manifest(path, rows)
    assert path.read_bytes() == before


def test_blocked_report_records_no_feature_artifact(tmp_path):
    dry = {"environment": {"python_version": "3.12", "torch_error": "missing",
                            "torchvision_error": "missing"}}
    path = write_blocked_report(tmp_path, dry)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert not data["feature_artifacts_created"]
    assert not data["random_weight_fallback"]
    assert (path.parent / "step5d_cnn_feature_pilot_report.txt").is_file()


def test_full_universe_matches_frozen_step5a_metadata():
    root = module.Path("data/interim/sequences_v2/context")
    rows = module.select_full_context(root)
    summary = module.full_summary(rows)
    assert summary == {
        "videos": 1677, "train": 1382, "val": 295,
        "drowsy": 800, "not_drowsy": 877, "sequence_length": 32,
        "target_rows": 53664, "imputed_target_refs": 699,
        "context_only_videos": 157, "test_rows": 0,
    }
    assert len({row["video_id"] for row in rows}) == 1677
    assert [row["feature_video_index"] for row in rows] == list(range(1677))
    assert "n_246" not in {row["video_id"] for row in rows}


def test_full_universe_rejects_test_split(monkeypatch):
    path = module.Path("data/interim/sequences_v2/context/metadata/context_sequence_videos.csv")
    rows = module._read_csv(path)
    rows[0] = {**rows[0], "split": "test"}
    monkeypatch.setattr(module, "_read_csv", lambda _: rows)
    with pytest.raises(ValueError, match="STEP_5D2_BLOCKED_TEST_LEAKAGE"):
        module.select_full_context(path.parents[1])


def test_full_mapping_and_dedup_match_frozen_step5a_artifacts():
    root = module.Path("data/interim/sequences_v2/context")
    videos = module.select_full_context(root)
    mappings = module.load_full_mapping(module.Path("."), videos)
    sources, source_ids = module.unique_source_index(mappings)
    assert len(mappings) == 53664
    assert len(sources) == 52965
    assert len(source_ids) - len(sources) == 699
    assert len({(row["video_id"], row["target_context_index"]) for row in mappings}) == 53664
    assert all(row["split"] in {"train", "val"} for row in mappings)


@pytest.mark.parametrize("resnet,vgg", [(16, 8), (8, 16), (64, 16)])
def test_full_batch_equality_requirement(resnet, vgg):
    with pytest.raises(ValueError, match="STEP_5D2_BLOCKED_BATCH_MISMATCH"):
        module.validate_full_batch_sizes(resnet, vgg)
    module.validate_full_batch_sizes(16, 16)


def test_full_feature_policy_is_identical_to_d1():
    config = load_config(module.Path("configs/context_cnn_features.yaml"))
    assert module.cnn_feature_policy_hash(
        config, "resnet18", module.RESOLVED_WEIGHTS["resnet18"]
    ) == "0b8d72abdc9f3fbdb44934100c507ebc249ea8225d01fe1e4ca0de2236c4cd3e"
    assert module.cnn_feature_policy_hash(
        config, "vgg16", module.RESOLVED_WEIGHTS["vgg16"]
    ) == "1edc39db86531ea3240f767722236ab45ee89d607811770308c30b526f0b2bc0"


def test_full_mapping_fingerprint_is_deterministic_and_order_sensitive():
    rows = [{"video_id": "v1", "target_context_index": str(index),
             "target_canonical_index": str(index * 3), "target_timestamp_sec": str(index / 10),
             "original_available": "True", "imputed": "False",
             "source_context_index": str(index), "source_canonical_index": str(index * 3),
             "source_image_relpath": f"relative/{index}.jpg"} for index in range(2)]
    first = module.mapping_fingerprint(rows)
    assert first == module.mapping_fingerprint(rows)
    assert first != module.mapping_fingerprint(list(reversed(rows)))


def test_full_resume_requires_all_frozen_provenance(tmp_path):
    root = tmp_path / "context_full"
    marker = root / "resnet18/COMPLETE.json"
    marker.parent.mkdir(parents=True)
    videos = [{"video_id": "v1"}, {"video_id": "v2"}]
    summary = marker.parent / "full_feature_summary.json"
    source_features = marker.parent / "source_features.npy"
    summary.write_text("{}", encoding="utf-8")
    np.save(source_features, np.ones((1, 512), dtype=np.float32), allow_pickle=False)
    marker.write_text(json.dumps({
        "backbone": "resnet18", "cnn_feature_policy_hash": "policy",
        "canonical_policy_hash": module.POLICY_HASH,
        "sequence_missing_policy_hash": module.SEQUENCE_POLICY_HASH,
        "full_video_ids": ["v1", "v2"], "source_mapping_fingerprint": "mapping",
        "full_feature_summary_sha256": module.file_sha256(summary),
        "source_features_sha256": module.file_sha256(source_features),
    }), encoding="utf-8")
    assert module._full_artifact_complete(root, "resnet18", "policy", videos, "mapping")
    assert not module._full_artifact_complete(root, "resnet18", "policy", videos, "changed")
    assert not module._full_artifact_complete(root, "vgg16", "policy", videos, "mapping")


def test_full_preflight_report_generation(tmp_path):
    result = {
        "status": "STEP_5D2_FULL_DRY_RUN_PASS",
        "full_context": {"videos": 1677, "train": 1382, "val": 295,
                         "drowsy": 800, "not_drowsy": 877, "target_rows": 53664,
                         "unique_source_images": 52965, "imputed_target_refs": 699,
                         "context_only_videos": 157},
        "source_mapping_fingerprint": "abc",
    }
    path = module.write_full_preflight_report(tmp_path, result)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == result["status"]
    assert "CNN execution: false" in (path.parent / "step5d2_full_feature_report.txt").read_text(
        encoding="utf-8")


def test_pilot_and_full_output_roots_are_isolated():
    pilot = module.Path("data/interim/features_v2/context_pilot")
    full = module.Path("data/interim/features_v2/context_full")
    assert pilot != full and full not in pilot.parents and pilot not in full.parents


def test_pilot_full_regression_compares_features_and_source_mapping(tmp_path):
    pilot_root = tmp_path / "data/interim/features_v2/context_pilot/resnet18"
    full_root = tmp_path / "data/interim/features_v2/context_full/resnet18"
    relative = module.Path("per_video/train/v1")
    (pilot_root / relative).mkdir(parents=True)
    (full_root / relative).mkdir(parents=True)
    (pilot_root / "COMPLETE.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "data/metadata/experiment2_step5d_cnn_feature_pilot_manifest.csv"
    manifest.parent.mkdir(parents=True)
    module._write_csv(manifest, module.PILOT_COLUMNS, [{
        "pilot_order": 0, "video_id": "v1", "split": "train", "label": "drowsy",
        "imputed_count": 0, "context_sequence_path": "sequence.csv",
        "selection_reason": "test",
    }])
    features = np.ones((32, 512), dtype=np.float32)
    np.save(pilot_root / relative / "features.npy", features, allow_pickle=False)
    np.save(full_root / relative / "features.npy", features, allow_pickle=False)
    mapping_rows = [{key: ("same.jpg" if key == "source_image_relpath" else index)
                     for key in module.MAPPING_COLUMNS} for index in range(32)]
    module._write_csv(pilot_root / relative / "feature_mapping.csv",
                      module.MAPPING_COLUMNS, mapping_rows)
    module._write_csv(full_root / relative / "feature_mapping.csv",
                      module.MAPPING_COLUMNS, mapping_rows)
    result = module.compare_pilot_full_features(tmp_path, full_root, "resnet18")
    assert result["status"] == "PASS"
    assert result["allclose_mismatch"] == 0
    assert result["source_mapping_mismatch"] == 0
    assert result["exact_equal_videos"] == 1
