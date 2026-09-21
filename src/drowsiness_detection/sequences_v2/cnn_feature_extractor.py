"""STEP 5-D1 frozen Context JPEG용 pretrained CNN feature pilot 도구."""

from __future__ import annotations

import csv
import importlib
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from drowsiness_detection.preprocessing_v2.preprocessing_provenance import file_sha256, stable_hash
from drowsiness_detection.sequences_v2.context_sequence_builder import POLICY_HASH, SEQUENCE_POLICY_HASH


BACKBONES = ("resnet18", "vgg16")
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
PILOT_COLUMNS = ("pilot_order", "video_id", "split", "label", "imputed_count",
                 "context_sequence_path", "selection_reason")
FULL_COLUMNS = ("feature_video_index", "video_id", "split", "label", "imputed_count",
                "context_sequence_path")
EXPECTED_FULL_COUNTS = {
    "videos": 1677, "train": 1382, "val": 295, "drowsy": 800,
    "not_drowsy": 877, "target_rows": 53664, "unique_source_images": 52965,
    "imputed_target_refs": 699, "context_only_videos": 157,
}
RESOLVED_WEIGHTS = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "vgg16": "VGG16_Weights.IMAGENET1K_V1",
}
MAPPING_COLUMNS = (
    "target_context_index", "target_canonical_index", "target_timestamp_sec",
    "original_available", "imputed", "source_context_index", "source_canonical_index",
    "source_image_relpath", "source_feature_id", "feature_row_index",
)
EXPECTED_CONFIG = {
    "version": 1,
    "input": {"image_size": 224, "sequence_length": 32, "channel_order": "RGB",
              "input_scale": "uint8_to_float32_div_255"},
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "geometry_transform": "none", "augmentation": "none",
    "backbones": {
        "resnet18": {"torchvision_model": "resnet18", "weights": "DEFAULT",
                     "classifier_removed": "fc", "pooling": "native_avgpool_1x1", "output_dim": 512},
        "vgg16": {"torchvision_model": "vgg16", "weights": "DEFAULT",
                  "classifier_removed": "classifier", "pooling": "adaptive_avgpool_1x1_after_features",
                  "output_dim": 512}},
    "inference": {"dtype": "float32", "amp": False, "gradient": False,
                  "model_mode": "eval"},
    "pilot": {"videos": 16, "no_imputation": 8, "imputation_required": 8},
}


class DependencyBlocked(RuntimeError):
    """필수 torch 실행 환경이 없어 random fallback 없이 중단됨을 나타낸다."""


class PretrainedWeightsBlocked(RuntimeError):
    """ImageNet weight를 불러오지 못해 random weight로 진행하지 않음을 나타낸다."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise ValueError("STEP_5D1_BLOCKED_POLICY_MISMATCH: CNN feature semantic config")
    return config


def environment_preflight() -> dict[str, Any]:
    """설치를 시도하지 않고 현재 interpreter의 torch/CUDA 상태만 기록한다."""
    result: dict[str, Any] = {"python_version": platform.python_version(),
                              "python_implementation": platform.python_implementation()}
    try:
        torch = importlib.import_module("torch")
        result.update(torch_import=True, torch_version=str(torch.__version__),
                      cuda_available=bool(torch.cuda.is_available()),
                      torch_cuda_runtime=torch.version.cuda,
                      gpu_device_count=int(torch.cuda.device_count()),
                      cudnn_version=torch.backends.cudnn.version())
        result["gpu_device_name"] = (torch.cuda.get_device_name(0)
                                     if result["cuda_available"] else None)
        result["float32_inference_supported"] = True
    except Exception as error:
        result.update(torch_import=False, torch_error=f"{type(error).__name__}: {error}",
                      cuda_available=False, torch_cuda_runtime=None, gpu_device_count=0,
                      gpu_device_name=None, cudnn_version=None,
                      float32_inference_supported=False)
    try:
        torchvision = importlib.import_module("torchvision")
        result.update(torchvision_import=True, torchvision_version=str(torchvision.__version__))
    except Exception as error:
        result.update(torchvision_import=False,
                      torchvision_error=f"{type(error).__name__}: {error}")
    result["status"] = ("PASS" if result.get("torch_import") and result.get("torchvision_import")
                        else "STEP_5D1_BLOCKED_DEPENDENCY")
    return result


def require_dependencies(environment: Mapping[str, Any]) -> None:
    missing = [name for name in ("torch", "torchvision")
               if not environment.get(f"{name}_import")]
    if missing:
        raise DependencyBlocked("STEP_5D1_BLOCKED_DEPENDENCY: " + ", ".join(missing))


def decode_rgb_jpeg(path: Path) -> np.ndarray:
    """OpenCV BGR 결과를 명시적으로 RGB uint8로 바꾸며 geometry는 변경하지 않는다."""
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (224, 224, 3) or bgr.dtype != np.uint8:
        raise ValueError(f"STEP_5D1_REVIEW_REQUIRED: Context JPEG RGB 224x224 오류 {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def normalize_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Resize/crop 없이 RGB HWC uint8를 ImageNet-normalized CHW float32로 바꾼다."""
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("STEP_5D1_REVIEW_REQUIRED: expected RGB uint8 224x224x3")
    chw = np.transpose(image, (2, 0, 1)).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray((chw - IMAGENET_MEAN) / IMAGENET_STD, dtype=np.float32)


def adapt_resnet_feature_map(output: np.ndarray) -> np.ndarray:
    """Mock/회귀 검사용 ResNet GAP 결과를 512D로 제한한다."""
    array = np.asarray(output)
    if array.ndim == 4 and array.shape[1:] == (512, 1, 1):
        array = array[:, :, 0, 0]
    if array.ndim != 2 or array.shape[1] != 512:
        raise ValueError("ResNet18 classifier logits가 아닌 512D GAP feature가 필요합니다")
    return array.astype(np.float32, copy=False)


def adapt_vgg_feature_map(output: np.ndarray) -> np.ndarray:
    """VGG features의 공간축만 GAP하고 25088/4096/1000D 출력을 거부한다."""
    array = np.asarray(output)
    if array.ndim != 4 or array.shape[1] != 512:
        raise ValueError("VGG16 model.features의 512-channel feature map이 필요합니다")
    return array.astype(np.float32, copy=False).mean(axis=(2, 3), dtype=np.float32)


def feature_policy(config: Mapping[str, Any], backbone: str, resolved_weights_enum: str) -> dict[str, Any]:
    if backbone not in BACKBONES or not resolved_weights_enum:
        raise ValueError("backbone/weights enum이 필요합니다")
    return {"backbone_name": backbone, **config["backbones"][backbone],
            "weights_enum": resolved_weights_enum, "input": config["input"],
            "normalization": config["normalization"],
            "geometry_transform": config["geometry_transform"],
            "augmentation": config["augmentation"], "dtype": config["inference"]["dtype"],
            "amp": config["inference"]["amp"], "frozen_pretrained": True}


def cnn_feature_policy_hash(config: Mapping[str, Any], backbone: str,
                            resolved_weights_enum: str) -> str:
    return stable_hash(feature_policy(config, backbone, resolved_weights_enum))


def _select_group(rows: Sequence[Mapping[str, Any]], count: int, imputed: bool) -> list[Mapping[str, Any]]:
    candidates = [row for row in rows if (int(row["imputed_slots"]) > 0) is imputed]
    strata = (("train", "drowsy"), ("train", "not_drowsy"),
              ("val", "drowsy"), ("val", "not_drowsy"))
    selected: list[Mapping[str, Any]] = []
    for split, label in strata:
        members = [row for row in candidates if row["split"] == split and row["label"] == label]
        members.sort(key=lambda row: ((-int(row["imputed_slots"]) if imputed else 0), row["video_id"]))
        if not members:
            raise ValueError(f"STEP_5D1_REVIEW_REQUIRED: pilot stratum 없음 {split}/{label}")
        selected.append(members[0])
    remaining = [row for row in candidates if row not in selected]
    remaining.sort(key=lambda row: ((-int(row["imputed_slots"]) if imputed else 0),
                                    row["split"], row["label"], row["video_id"]))
    return selected + remaining[:count - len(selected)]


def select_pilot(context_root: Path) -> list[dict[str, Any]]:
    """각 imputation group에서 네 split×label strata를 포함하는 결정적 8+8 표본을 고른다."""
    videos = _read_csv(context_root / "metadata/context_sequence_videos.csv")
    if len(videos) != 1677 or any(row["split"] not in {"train", "val"} for row in videos):
        raise ValueError("STEP_5D1_BLOCKED_TEST_LEAKAGE: Context eligible universe")
    selected = _select_group(videos, 8, False) + _select_group(videos, 8, True)
    if len(selected) != 16 or len({row["video_id"] for row in selected}) != 16:
        raise ValueError("STEP_5D1_REVIEW_REQUIRED: pilot selection count/duplicate")
    output = []
    for order, row in enumerate(selected):
        imputed = int(row["imputed_slots"])
        output.append({"pilot_order": order, "video_id": row["video_id"],
                       "split": row["split"], "label": row["label"],
                       "imputed_count": imputed,
                       "context_sequence_path":
                           f"data/interim/sequences_v2/context/per_video/{row['split']}/{row['video_id']}/sequence.csv",
                       "selection_reason": ("deterministic_no_imputation_stratified"
                                            if imputed == 0 else "deterministic_high_imputation_stratified")})
    return output


def pilot_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"videos": len(rows), "train": sum(row["split"] == "train" for row in rows),
            "val": sum(row["split"] == "val" for row in rows),
            "drowsy": sum(row["label"] == "drowsy" for row in rows),
            "not_drowsy": sum(row["label"] == "not_drowsy" for row in rows),
            "no_imputation": sum(int(row["imputed_count"]) == 0 for row in rows),
            "imputation_required": sum(int(row["imputed_count"]) > 0 for row in rows),
            "target_rows_per_backbone": 32 * len(rows), "test_rows": 0}


def select_full_context(context_root: Path) -> list[dict[str, Any]]:
    """STEP 5-A의 결정적 순서를 유지해 전체 Context eligible 목록을 만든다."""
    videos = _read_csv(context_root / "metadata/context_sequence_videos.csv")
    if len(videos) != EXPECTED_FULL_COUNTS["videos"]:
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: video count")
    if len({row["video_id"] for row in videos}) != len(videos):
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: duplicate video")
    if any(row["split"] not in {"train", "val"} for row in videos):
        raise ValueError("STEP_5D2_BLOCKED_TEST_LEAKAGE")
    if any(int(row["sequence_length"]) != 32 for row in videos):
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: sequence length")
    rows = [{"feature_video_index": index, "video_id": row["video_id"],
             "split": row["split"], "label": row["label"],
             "imputed_count": int(row["imputed_slots"]),
             "context_sequence_path":
                 f"data/interim/sequences_v2/context/per_video/{row['split']}/{row['video_id']}/sequence.csv",
             "behavior_eligible": str(row["behavior_eligible"]).casefold() == "true"}
            for index, row in enumerate(videos)]
    summary = full_summary(rows)
    for key in ("videos", "train", "val", "drowsy", "not_drowsy",
                "target_rows", "imputed_target_refs", "context_only_videos"):
        if summary[key] != EXPECTED_FULL_COUNTS[key]:
            raise ValueError(f"STEP_5D2_BLOCKED_FULL_UNIVERSE: {key}")
    if any(row["video_id"] == "n_246" for row in rows):
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: n_246 must be excluded")
    return rows


def full_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "videos": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "val": sum(row["split"] == "val" for row in rows),
        "drowsy": sum(row["label"] == "drowsy" for row in rows),
        "not_drowsy": sum(row["label"] == "not_drowsy" for row in rows),
        "sequence_length": 32,
        "target_rows": len(rows) * 32,
        "imputed_target_refs": sum(int(row["imputed_count"]) for row in rows),
        "context_only_videos": sum(not bool(row.get("behavior_eligible", True)) for row in rows),
        "test_rows": 0,
    }


def validate_full_batch_sizes(resnet_batch_size: int, vgg_batch_size: int) -> None:
    """Full controlled comparison은 두 backbone 모두 batch 16만 허용한다."""
    if resnet_batch_size != 16 or vgg_batch_size != 16:
        raise ValueError("STEP_5D2_BLOCKED_BATCH_MISMATCH: full mode requires 16 / 16")


def load_full_mapping(project_root: Path, full_rows: Sequence[Mapping[str, Any]],
                      check_source_files: bool = True) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, int]] = set()
    for video in full_rows:
        rows = _read_csv(project_root / str(video["context_sequence_path"]))
        indices = [int(row["target_context_index"]) for row in rows]
        if len(rows) != 32 or indices != list(range(32)):
            raise ValueError(f"STEP_5D2_BLOCKED_FULL_UNIVERSE: mapping {video['video_id']}")
        for row in rows:
            if row["split"] not in {"train", "val"}:
                raise ValueError("STEP_5D2_BLOCKED_TEST_LEAKAGE")
            if (row["video_id"] != video["video_id"]
                    or row["canonical_policy_hash"] != POLICY_HASH
                    or row["sequence_missing_policy_hash"] != SEQUENCE_POLICY_HASH):
                raise ValueError("STEP_5D2_BLOCKED_FEATURE_POLICY_MISMATCH")
            target = (row["video_id"], int(row["target_context_index"]))
            if target in seen_targets:
                raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: duplicate target index")
            seen_targets.add(target)
            if check_source_files and not (project_root / row["source_image_relpath"]).is_file():
                raise FileNotFoundError(f"STEP_5D2_BLOCKED_MISSING_SOURCE: {row['source_image_relpath']}")
            mappings.append({**row, "feature_video_index": int(video["feature_video_index"])})
    if len(mappings) != EXPECTED_FULL_COUNTS["target_rows"]:
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: target rows")
    return mappings


def load_pilot_mapping(project_root: Path, pilot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    mappings = []
    for pilot in pilot_rows:
        rows = _read_csv(project_root / str(pilot["context_sequence_path"]))
        if len(rows) != 32 or [int(row["target_context_index"]) for row in rows] != list(range(32)):
            raise ValueError(f"STEP_5D1_REVIEW_REQUIRED: Context mapping {pilot['video_id']}")
        if any(row["split"] not in {"train", "val"} or row["video_id"] != pilot["video_id"]
               or row["canonical_policy_hash"] != POLICY_HASH
               or row["sequence_missing_policy_hash"] != SEQUENCE_POLICY_HASH for row in rows):
            raise ValueError("STEP_5D1_BLOCKED_POLICY_MISMATCH: Context sequence")
        mappings.extend({**row, "pilot_order": int(pilot["pilot_order"])} for row in rows)
    return mappings


def unique_source_index(mappings: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[int]]:
    """정렬된 source path마다 하나의 ID를 부여해 중복 inference를 방지한다."""
    sources = sorted({str(row["source_image_relpath"]) for row in mappings})
    lookup = {path: index for index, path in enumerate(sources)}
    return sources, [lookup[str(row["source_image_relpath"])] for row in mappings]


def assemble_target_features(source_features: np.ndarray, source_ids: Sequence[int],
                             video_count: int) -> np.ndarray:
    if source_features.ndim != 2 or source_features.shape[1] != 512:
        raise ValueError("source feature는 [N,512]여야 합니다")
    if len(source_ids) != video_count * 32:
        raise ValueError("target/source mapping row 수 불일치")
    output = source_features[np.asarray(source_ids, dtype=np.int64)].reshape(video_count, 32, 512)
    if output.dtype != np.float32:
        output = output.astype(np.float32)
    return output


def feature_numeric_summary(features: np.ndarray) -> dict[str, Any]:
    if features.ndim < 2 or features.shape[-1] != 512:
        raise ValueError("feature dimension은 512여야 합니다")
    flat = features.reshape(-1, 512)
    norms = np.linalg.norm(flat, axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
        return {"finite_count": int(np.isfinite(flat).sum()), "nan_count": int(np.isnan(flat).sum()),
                "inf_count": int(np.isinf(flat).sum()), "all_zero_count": int(np.count_nonzero(norms == 0)),
                "mean": float(flat.mean()), "std": float(flat.std()), "min": float(flat.min()),
                "max": float(flat.max()), "l2_mean": float(norms.mean()),
                "l2_median": float(np.median(norms)), "l2_min": float(norms.min()),
                "l2_max": float(norms.max())}


def build_torch_extractor(backbone: str, device: str):
    """DEFAULT ImageNet weight만 허용하며 실패 시 random initialization으로 fallback하지 않는다."""
    environment = environment_preflight()
    require_dependencies(environment)
    torch = importlib.import_module("torch")
    models = importlib.import_module("torchvision.models")
    nn = importlib.import_module("torch.nn")
    try:
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            pretrained = models.resnet18(weights=weights)
            extractor = nn.Sequential(*list(pretrained.children())[:-1], nn.Flatten(1))
        elif backbone == "vgg16":
            weights = models.VGG16_Weights.DEFAULT
            pretrained = models.vgg16(weights=weights)
            extractor = nn.Sequential(pretrained.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone}")
    except Exception as error:
        raise PretrainedWeightsBlocked(
            f"STEP_5D1_BLOCKED_PRETRAINED_WEIGHTS: {backbone}: {type(error).__name__}: {error}") from error
    extractor.eval().requires_grad_(False).to(device)
    enum_name = f"{type(weights).__name__}.{weights.name}"
    return torch, extractor, enum_name


def dry_run(project_root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    context_root = project_root / "data/interim/sequences_v2/context"
    pilot = select_pilot(context_root)
    mappings = load_pilot_mapping(project_root, pilot)
    sources, _ = unique_source_index(mappings)
    environment = environment_preflight()
    return {"status": ("STEP_5D1_DRY_RUN_PASS" if environment["status"] == "PASS"
                       else "STEP_5D1_IMPLEMENTED_BLOCKED_DEPENDENCY"),
            "environment": environment, "context_eligible_universe": 1677,
            "pilot": pilot_summary(pilot), "pilot_rows": pilot,
            "backbones": list(BACKBONES), "weights_requested": "DEFAULT ImageNet pretrained",
            "resolved_weights": "UNRESOLVED_DEPENDENCY" if environment["status"] != "PASS" else "AT_RUNTIME",
            "unique_source_images": len(sources), "output_dimension": 512,
            "normalization": config["normalization"], "geometry_transform": "none",
            "cnn_execution": False, "full_extraction": False, "test_access": 0,
            "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH}


def full_dry_run(project_root: Path, config_path: Path, resnet_batch_size: int = 16,
                 vgg_batch_size: int = 16, write_report: bool = True) -> dict[str, Any]:
    """CNN을 실행하지 않고 전체 입력 universe와 게시 계획을 검증한다."""
    config = load_config(config_path)
    validate_full_batch_sizes(resnet_batch_size, vgg_batch_size)
    context_root = project_root / "data/interim/sequences_v2/context"
    videos = select_full_context(context_root)
    mappings = load_full_mapping(project_root, videos)
    sources, source_ids = unique_source_index(mappings)
    summary = full_summary(videos)
    summary["unique_source_images"] = len(sources)
    if len(sources) != EXPECTED_FULL_COUNTS["unique_source_images"]:
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: unique source images")
    if len(source_ids) - len(sources) != EXPECTED_FULL_COUNTS["imputed_target_refs"]:
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: source reuse count")
    policy_hashes = {backbone: cnn_feature_policy_hash(config, backbone, enum_name)
                     for backbone, enum_name in RESOLVED_WEIGHTS.items()}
    environment = environment_preflight()
    result = {
        "status": "STEP_5D2_FULL_DRY_RUN_PASS",
        "environment": environment,
        "full_context": summary,
        "backbones": list(BACKBONES),
        "resolved_weights": dict(RESOLVED_WEIGHTS),
        "cnn_feature_policy_hashes": policy_hashes,
        "batch_size": {"resnet18": resnet_batch_size, "vgg16": vgg_batch_size},
        "feature_dimension": 512,
        "normalization": config["normalization"],
        "geometry_transform": "none",
        "source_mapping_fingerprint": mapping_fingerprint(mappings),
        "context_only_included": summary["context_only_videos"],
        "n_246_included": False,
        "cnn_execution": False,
        "actual_full_inference": False,
        "test_access": 0,
        "canonical_policy_hash": POLICY_HASH,
        "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
    }
    if write_report:
        write_full_preflight_report(project_root, result)
    return result


def write_full_preflight_report(project_root: Path, result: Mapping[str, Any]) -> Path:
    root = project_root / "outputs/features_v2/context_cnn/full"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "step5d2_full_feature_preflight.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    values = result["full_context"]
    report = root / "step5d2_full_feature_report.txt"
    report.write_text("\n".join([
        "STEP 5-D2 Full Context CNN Feature Extraction Preflight", "",
        f"Status: {result['status']}",
        f"Videos: {values['videos']} (train {values['train']}, val {values['val']})",
        f"Labels: drowsy {values['drowsy']}, not_drowsy {values['not_drowsy']}",
        f"Target rows: {values['target_rows']}",
        f"Unique source images: {values['unique_source_images']}",
        f"Imputed target references: {values['imputed_target_refs']}",
        f"Context-only included: {values['context_only_videos']}",
        "n_246 excluded: true", "Batch size: resnet18 16 / vgg16 16",
        "CNN execution: false", "Test access: 0 (SEALED)",
        f"Source mapping fingerprint: {result['source_mapping_fingerprint']}", "",
    ]), encoding="utf-8")
    return json_path


def write_pilot_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        existing = _read_csv(path)
        if existing != [{key: str(value) for key, value in row.items()} for row in rows]:
            raise FileExistsError(f"기존 pilot manifest 충돌: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, PILOT_COLUMNS, rows)


def write_blocked_report(project_root: Path, dry: Mapping[str, Any]) -> Path:
    """Feature artifact 없이 환경 blocker와 계획만 별도 report에 기록한다."""
    root = project_root / "outputs/features_v2/context_cnn/pilot"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "step5d_cnn_feature_pilot_preflight.json"
    value = {**dry, "decision": "STEP_5D1_IMPLEMENTED_BLOCKED_BY_ENVIRONMENT",
             "feature_artifacts_created": False, "random_weight_fallback": False,
             "full_extraction_started": False}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = root / "step5d_cnn_feature_pilot_report.txt"
    report.write_text("\n".join([
        "STEP 5-D1 CNN Feature Extraction Preflight & Controlled Pilot", "", "[Scope]",
        "16-video deterministic pilot only; full extraction is forbidden.", "", "[Environment]",
        f"Python: {dry['environment']['python_version']}",
        f"torch: {dry['environment'].get('torch_error', dry['environment'].get('torch_version'))}",
        f"torchvision: {dry['environment'].get('torchvision_error', dry['environment'].get('torchvision_version'))}",
        "", "[Pilot Selection]", "16 videos: 8 no-imputation + 8 imputation-required; train/val and labels included.",
        "", "[ImageNet Preprocessing]", "Implemented: RGB uint8 224x224, /255, ImageNet mean/std; no resize/crop/augmentation.",
        "", "[ResNet18 Extractor]", "Implemented; pretrained DEFAULT required; fc removed; native GAP; 512D.",
        "", "[VGG16 Extractor]", "Implemented; pretrained DEFAULT required; features + adaptive GAP 1x1; 512D; classifier unused.",
        "", "[Feature Shape]", "Not executed because torch/torchvision are unavailable.",
        "", "[Provenance]", "Random-weight fallback is forbidden. Backbones remain separate.",
        "", "[Test Protection]", "Test access 0; SEALED.", "", "[Decision]",
        "STEP 5-D1: IMPLEMENTED / BLOCKED BY ENVIRONMENT",
        "STEP_5D1_BLOCKED_DEPENDENCY: torch, torchvision", "STEP 5-D2: NOT STARTED", "",
    ]), encoding="utf-8")
    return path


def mapping_fingerprint(mappings: Sequence[Mapping[str, Any]]) -> str:
    """Machine path를 제외한 target/source mapping 순서와 출처를 고정한다."""
    keys = ("video_id", "target_context_index", "target_canonical_index", "target_timestamp_sec",
            "original_available", "imputed", "source_context_index", "source_canonical_index",
            "source_image_relpath")
    return stable_hash({"rows": [{key: row[key] for key in keys} for row in mappings]})


def _artifact_complete(root: Path, backbone: str, policy_hash: str,
                       pilot: Sequence[Mapping[str, Any]],
                       source_mapping_fingerprint: str | None = None) -> bool:
    marker = root / backbone / "COMPLETE.json"
    if not marker.is_file():
        return False
    data = json.loads(marker.read_text(encoding="utf-8"))
    matches = (data.get("backbone") == backbone
            and data.get("cnn_feature_policy_hash") == policy_hash
            and data.get("canonical_policy_hash") == POLICY_HASH
            and data.get("sequence_missing_policy_hash") == SEQUENCE_POLICY_HASH
            and data.get("pilot_video_ids") == [row["video_id"] for row in pilot])
    return matches and (source_mapping_fingerprint is None
                        or data.get("source_mapping_fingerprint") == source_mapping_fingerprint)


def _extract_unique_sources(torch, extractor, sources: Sequence[str], project_root: Path,
                            device: str, batch_size: int) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(sources), batch_size):
            batch_paths = sources[start:start + batch_size]
            arrays = np.stack([normalize_rgb_uint8(decode_rgb_jpeg(project_root / path))
                               for path in batch_paths])
            tensor = torch.from_numpy(arrays).to(device=device, dtype=torch.float32)
            output = extractor(tensor)
            if tuple(output.shape) != (len(batch_paths), 512):
                raise ValueError(f"STEP_5D1_REVIEW_REQUIRED: extractor output {tuple(output.shape)}")
            chunks.append(output.detach().cpu().to(dtype=torch.float32).numpy())
    features = np.concatenate(chunks).astype(np.float32, copy=False)
    numeric = feature_numeric_summary(features)
    if numeric["nan_count"] or numeric["inf_count"] or numeric["all_zero_count"]:
        raise ValueError("STEP_5D1_REVIEW_REQUIRED: feature numeric anomaly")
    return features


def _publish_backbone(stage: Path, backbone: str, pilot: Sequence[Mapping[str, Any]],
                      mappings: Sequence[Mapping[str, Any]], sources: Sequence[str],
                      source_ids: Sequence[int], source_features: np.ndarray,
                      target_features: np.ndarray, policy: Mapping[str, Any], policy_hash: str,
                      runtime: Mapping[str, Any]) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=True)
    np.save(stage / "source_features.npy", source_features, allow_pickle=False)
    first_source: dict[str, Mapping[str, Any]] = {}
    for row in mappings:
        first_source.setdefault(str(row["source_image_relpath"]), row)
    source_rows = [{"source_feature_id": index, "source_image_relpath": path,
                    "video_id": first_source[path]["video_id"],
                    "source_context_index": first_source[path]["source_context_index"],
                    "source_canonical_index": first_source[path]["source_canonical_index"]}
                   for index, path in enumerate(sources)]
    _write_csv(stage / "source_feature_index.csv",
               ("source_feature_id", "source_image_relpath", "video_id",
                "source_context_index", "source_canonical_index"), source_rows)
    imputed_mismatch = 0
    global_mapping_fingerprint = mapping_fingerprint(mappings)
    for order, pilot_row in enumerate(pilot):
        video_rows = mappings[order * 32:(order + 1) * 32]
        video_ids = source_ids[order * 32:(order + 1) * 32]
        bundle = stage / "per_video" / pilot_row["split"] / pilot_row["video_id"]
        bundle.mkdir(parents=True)
        array = target_features[order]
        np.save(bundle / "features.npy", array, allow_pickle=False)
        feature_rows = []
        for index, (row, source_id) in enumerate(zip(video_rows, video_ids)):
            feature_rows.append({key: row[key] for key in MAPPING_COLUMNS[:-2]}
                                | {"source_feature_id": source_id, "feature_row_index": index})
            if str(row["imputed"]).casefold() == "true":
                source_target = int(row["source_context_index"])
                if not np.array_equal(array[index], array[source_target]):
                    imputed_mismatch += 1
        _write_csv(bundle / "feature_mapping.csv", MAPPING_COLUMNS, feature_rows)
        np.savez_compressed(bundle / "masks.npz",
                            original_context_valid_mask=np.asarray(
                                [str(row["original_available"]).casefold() == "true" for row in video_rows],
                                dtype=np.bool_),
                            context_imputed_mask=np.asarray(
                                [str(row["imputed"]).casefold() == "true" for row in video_rows],
                                dtype=np.bool_))
        summary = {"video_id": pilot_row["video_id"], "backbone": backbone,
                   "feature_shape": [32, 512], "dtype": "float32",
                   "imputed_count": int(pilot_row["imputed_count"]),
                   "cnn_feature_policy_hash": policy_hash,
                   "source_mapping_fingerprint": mapping_fingerprint(video_rows),
                   "canonical_policy_hash": POLICY_HASH,
                   "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH}
        (bundle / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        complete = {**summary, "file_sha256": {
            name: file_sha256(bundle / name) for name in
            ("features.npy", "feature_mapping.csv", "masks.npz", "summary.json")}}
        (bundle / "COMPLETE.json").write_text(
            json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if imputed_mismatch:
        raise ValueError("STEP_5D1_REVIEW_REQUIRED: imputed feature mismatch")
    numeric = feature_numeric_summary(target_features)
    summary = {"backbone": backbone, "feature_policy": policy,
               "cnn_feature_policy_hash": policy_hash, "pilot_videos": len(pilot),
               "target_rows": len(pilot) * 32, "unique_source_images": len(sources),
               "feature_dimension": 512, "dtype": "float32", **numeric,
               "imputed_feature_mismatch": imputed_mismatch, "wrong_source_mapping": 0,
               "runtime": dict(runtime), "test_used": False,
               "canonical_policy_hash": POLICY_HASH,
               "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH}
    (stage / "pilot_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage / "COMPLETE.json").write_text(json.dumps({
        "backbone": backbone, "cnn_feature_policy_hash": policy_hash,
        "canonical_policy_hash": POLICY_HASH,
        "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        "pilot_video_ids": [row["video_id"] for row in pilot],
        "source_mapping_fingerprint": global_mapping_fingerprint,
        "pilot_feature_summary_sha256": file_sha256(stage / "pilot_feature_summary.json"),
        "source_features_sha256": file_sha256(stage / "source_features.npy")},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_pilot(project_root: Path, config_path: Path, output_root: Path,
              report_root: Path, manifest_path: Path, device: str = "auto",
              resnet_batch_size: int = 64, vgg_batch_size: int = 16,
              resume: bool = False) -> dict[str, Any]:
    """16개 pilot만 두 backbone으로 별도 추출하며 full universe에는 절대 쓰지 않는다."""
    config = load_config(config_path)
    context_root = project_root / "data/interim/sequences_v2/context"
    pilot = select_pilot(context_root)
    write_pilot_manifest(manifest_path, pilot)
    mappings = load_pilot_mapping(project_root, pilot)
    sources, source_ids = unique_source_index(mappings)
    source_mapping_fingerprint = mapping_fingerprint(mappings)
    environment = environment_preflight()
    require_dependencies(environment)
    torch = importlib.import_module("torch")
    resolved_device = ("cuda" if environment["cuda_available"] else "cpu") if device == "auto" else device
    if resolved_device == "cuda" and not environment["cuda_available"]:
        raise DependencyBlocked("STEP_5D1_BLOCKED_DEPENDENCY: CUDA requested but unavailable")
    target_ids = [row["video_id"] for row in pilot]
    summaries = []
    total_start = time.perf_counter()
    for backbone, batch_size in (("resnet18", resnet_batch_size), ("vgg16", vgg_batch_size)):
        load_start = time.perf_counter()
        torch_module, extractor, enum_name = build_torch_extractor(backbone, resolved_device)
        load_sec = time.perf_counter() - load_start
        policy = feature_policy(config, backbone, enum_name)
        policy_hash = stable_hash(policy)
        if _artifact_complete(output_root, backbone, policy_hash, pilot,
                              source_mapping_fingerprint):
            if not resume:
                raise FileExistsError(f"기존 {backbone} pilot artifact가 있습니다")
            summaries.append(json.loads(
                (output_root / backbone / "pilot_feature_summary.json").read_text(encoding="utf-8")))
            continue
        final = output_root / backbone
        if final.exists():
            raise FileExistsError(f"STEP_5D1_REVIEW_REQUIRED: conflicting {backbone} output")
        if resolved_device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        extraction_start = time.perf_counter()
        source_features = _extract_unique_sources(torch_module, extractor, sources, project_root,
                                                  resolved_device, batch_size)
        extraction_sec = time.perf_counter() - extraction_start
        target_features = assemble_target_features(source_features, source_ids, len(pilot))
        runtime = {"model_load_sec": load_sec, "feature_extraction_sec": extraction_sec,
                   "total_sec": load_sec + extraction_sec,
                   "source_images_per_sec": len(sources) / extraction_sec,
                   "device": resolved_device, "batch_size": batch_size,
                   "amp": False, "peak_gpu_memory_bytes":
                       int(torch.cuda.max_memory_allocated()) if resolved_device == "cuda" else 0}
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".step5d1_{backbone}_", dir=output_root) as temporary:
            summary = _publish_backbone(Path(temporary), backbone, pilot, mappings, sources,
                                        source_ids, source_features, target_features, policy,
                                        policy_hash, runtime)
            Path(temporary).rename(final)
        summaries.append(summary)
        del extractor
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
    result = {"status": "STEP_5D1_CNN_FEATURE_EXTRACTION_PREFLIGHT_AND_PILOT_COMPLETE",
              "environment": environment, "pilot": pilot_summary(pilot),
              "pilot_video_ids": target_ids, "unique_source_images": len(sources),
              "backbones": summaries, "backbone_fusion": False,
              "total_wall_sec": time.perf_counter() - total_start,
              "test_access": 0, "full_extraction_started": False}
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "step5d_cnn_feature_pilot_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_root / "step5d_cnn_feature_pilot_report.txt").write_text("\n".join([
        "STEP 5-D1 CNN Feature Extraction Preflight & Controlled Pilot", "",
        "[Scope] 16-video pilot only; full extraction not started.",
        f"[Environment] torch {environment['torch_version']}; torchvision {environment['torchvision_version']}; device {resolved_device}.",
        "[Pilot Selection] Deterministic 8 no-imputation + 8 imputation-required.",
        "[Input Sequence Integrity] 16 x 32 target mappings; test 0.",
        "[ImageNet Preprocessing] RGB 224x224 /255 + mean/std; no geometry transform.",
        "[ResNet18 Extractor] DEFAULT pretrained; fc removed; native GAP; 512D.",
        "[VGG16 Extractor] DEFAULT pretrained; features + adaptive GAP; classifier unused; 512D.",
        "[Feature Shape] [32,512]/video for each independent backbone.",
        "[Feature Numeric Sanity] NaN/Inf/all-zero 0.",
        "[Imputed Feature Consistency] Exact source feature reuse; mismatch 0.",
        "[Provenance] Separate policy hash and artifact per backbone; no fusion.",
        "[Test Protection] SEALED.", "[Decision] STEP 5-D1 COMPLETE; STEP 5-D2 NOT STARTED.", "",
    ]), encoding="utf-8")
    return result


def _full_artifact_complete(root: Path, backbone: str, policy_hash: str,
                            videos: Sequence[Mapping[str, Any]],
                            source_mapping_fingerprint: str) -> bool:
    backbone_root = root / backbone
    marker = backbone_root / "COMPLETE.json"
    if not marker.is_file():
        return False
    data = json.loads(marker.read_text(encoding="utf-8"))
    summary = backbone_root / "full_feature_summary.json"
    source_features = backbone_root / "source_features.npy"
    return (summary.is_file() and source_features.is_file()
            and data.get("full_feature_summary_sha256") == file_sha256(summary)
            and data.get("source_features_sha256") == file_sha256(source_features)
            and data.get("backbone") == backbone
            and data.get("cnn_feature_policy_hash") == policy_hash
            and data.get("canonical_policy_hash") == POLICY_HASH
            and data.get("sequence_missing_policy_hash") == SEQUENCE_POLICY_HASH
            and data.get("full_video_ids") == [row["video_id"] for row in videos]
            and data.get("source_mapping_fingerprint") == source_mapping_fingerprint)


def _checkpoint_provenance(torch, backbone: str) -> dict[str, Any]:
    """Torch Hub의 실제 cache 파일이 있을 때만 checkpoint SHA-256을 기록한다."""
    filename = {"resnet18": "resnet18-f37072fd.pth",
                "vgg16": "vgg16-397923af.pth"}[backbone]
    candidate = Path(torch.hub.get_dir()) / "checkpoints" / filename
    return {"cached_checkpoint": filename,
            "checkpoint_sha256": file_sha256(candidate) if candidate.is_file() else None}


def compare_pilot_full_features(project_root: Path, full_backbone_root: Path,
                                backbone: str) -> dict[str, Any]:
    """실제 full 게시 후 D1 pilot과 동일 target feature를 회귀 비교한다."""
    pilot_root = project_root / "data/interim/features_v2/context_pilot" / backbone
    manifest_path = project_root / "data/metadata/experiment2_step5d_cnn_feature_pilot_manifest.csv"
    if not (pilot_root / "COMPLETE.json").is_file() or not manifest_path.is_file():
        return {"status": "PILOT_ARTIFACT_UNAVAILABLE", "videos_compared": 0,
                "allclose_mismatch": None, "exact_equal_videos": None,
                "source_mapping_mismatch": None, "rtol": 1e-5, "atol": 1e-6}
    mismatch = 0
    exact = 0
    source_mapping_mismatch = 0
    rows = _read_csv(manifest_path)
    for row in rows:
        relative = Path("per_video") / row["split"] / row["video_id"] / "features.npy"
        pilot = np.load(pilot_root / relative, allow_pickle=False)
        full = np.load(full_backbone_root / relative, allow_pickle=False)
        pilot_mapping = _read_csv(pilot_root / relative.parent / "feature_mapping.csv")
        full_mapping = _read_csv(full_backbone_root / relative.parent / "feature_mapping.csv")
        source_mapping_mismatch += sum(
            left["source_image_relpath"] != right["source_image_relpath"]
            for left, right in zip(pilot_mapping, full_mapping))
        if not np.allclose(pilot, full, rtol=1e-5, atol=1e-6):
            mismatch += 1
        if np.array_equal(pilot, full):
            exact += 1
    return {"status": "PASS" if mismatch == 0 and source_mapping_mismatch == 0 else "FAIL",
            "videos_compared": len(rows), "allclose_mismatch": mismatch,
            "exact_equal_videos": exact,
            "source_mapping_mismatch": source_mapping_mismatch,
            "rtol": 1e-5, "atol": 1e-6}


def _publish_full_backbone(stage: Path, backbone: str,
                           videos: Sequence[Mapping[str, Any]],
                           mappings: Sequence[Mapping[str, Any]], sources: Sequence[str],
                           source_ids: Sequence[int], source_features: np.ndarray,
                           target_features: np.ndarray, policy: Mapping[str, Any],
                           policy_hash: str, runtime: Mapping[str, Any],
                           checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=True)
    np.save(stage / "source_features.npy", source_features, allow_pickle=False)
    first_source: dict[str, Mapping[str, Any]] = {}
    for row in mappings:
        first_source.setdefault(str(row["source_image_relpath"]), row)
    source_rows = [{"source_feature_id": index, "source_image_relpath": path,
                    "video_id": first_source[path]["video_id"],
                    "source_context_index": first_source[path]["source_context_index"],
                    "source_canonical_index": first_source[path]["source_canonical_index"]}
                   for index, path in enumerate(sources)]
    _write_csv(stage / "source_feature_index.csv",
               ("source_feature_id", "source_image_relpath", "video_id",
                "source_context_index", "source_canonical_index"), source_rows)
    _write_csv(stage / "full_feature_videos.csv", FULL_COLUMNS,
               [{key: row[key] for key in FULL_COLUMNS} for row in videos])
    imputed_mismatch = 0
    wrong_source_mapping = 0
    for order, video in enumerate(videos):
        video_rows = mappings[order * 32:(order + 1) * 32]
        video_ids = source_ids[order * 32:(order + 1) * 32]
        bundle = stage / "per_video" / video["split"] / video["video_id"]
        bundle.mkdir(parents=True)
        array = target_features[order]
        np.save(bundle / "features.npy", array, allow_pickle=False)
        feature_rows = []
        for index, (row, source_id) in enumerate(zip(video_rows, video_ids)):
            if sources[source_id] != row["source_image_relpath"]:
                wrong_source_mapping += 1
            feature_rows.append({key: row[key] for key in MAPPING_COLUMNS[:-2]}
                                | {"source_feature_id": source_id, "feature_row_index": index})
            if str(row["imputed"]).casefold() == "true":
                source_target = int(row["source_context_index"])
                if not np.array_equal(array[index], array[source_target]):
                    imputed_mismatch += 1
        _write_csv(bundle / "feature_mapping.csv", MAPPING_COLUMNS, feature_rows)
        np.savez_compressed(bundle / "masks.npz",
                            original_context_valid_mask=np.asarray(
                                [str(row["original_available"]).casefold() == "true"
                                 for row in video_rows], dtype=np.bool_),
                            context_imputed_mask=np.asarray(
                                [str(row["imputed"]).casefold() == "true"
                                 for row in video_rows], dtype=np.bool_))
        summary = {"video_id": video["video_id"], "backbone": backbone,
                   "feature_shape": [32, 512], "dtype": "float32",
                   "imputed_count": int(video["imputed_count"]),
                   "cnn_feature_policy_hash": policy_hash,
                   "source_mapping_fingerprint": mapping_fingerprint(video_rows),
                   "canonical_policy_hash": POLICY_HASH,
                   "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH}
        (bundle / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        complete = {**summary, "file_sha256": {
            name: file_sha256(bundle / name) for name in
            ("features.npy", "feature_mapping.csv", "masks.npz", "summary.json")}}
        (bundle / "COMPLETE.json").write_text(
            json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if imputed_mismatch:
        raise ValueError("STEP_5D2_REVIEW_REQUIRED: imputed feature mismatch")
    if wrong_source_mapping:
        raise ValueError("STEP_5D2_REVIEW_REQUIRED: wrong source mapping")
    numeric = feature_numeric_summary(target_features)
    if numeric["nan_count"] or numeric["inf_count"] or numeric["all_zero_count"]:
        raise ValueError("STEP_5D2_REVIEW_REQUIRED: feature numeric anomaly")
    global_fingerprint = mapping_fingerprint(mappings)
    summary = {"backbone": backbone, "feature_policy": policy,
               "cnn_feature_policy_hash": policy_hash, "full_videos": len(videos),
               "sequence_length": 32, "target_rows": len(mappings),
               "unique_source_images": len(sources), "feature_dimension": 512,
               "dtype": "float32", **numeric,
               "imputed_feature_mismatch": 0, "wrong_source_mapping": 0,
               "runtime": dict(runtime), "environment_provenance": environment_preflight(),
               "checkpoint_provenance": dict(checkpoint), "test_used": False,
               "canonical_policy_hash": POLICY_HASH,
               "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
               "source_mapping_fingerprint": global_fingerprint}
    summary_path = stage / "full_feature_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    (stage / "COMPLETE.json").write_text(json.dumps({
        "backbone": backbone, "cnn_feature_policy_hash": policy_hash,
        "canonical_policy_hash": POLICY_HASH,
        "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        "full_video_ids": [row["video_id"] for row in videos],
        "source_mapping_fingerprint": global_fingerprint,
        "full_feature_summary_sha256": file_sha256(summary_path),
        "source_features_sha256": file_sha256(stage / "source_features.npy")},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_full(project_root: Path, config_path: Path, output_root: Path,
             report_root: Path, device: str = "auto", resnet_batch_size: int = 16,
             vgg_batch_size: int = 16, resume: bool = False) -> dict[str, Any]:
    """전체 Context eligible train/val을 backbone별 원자적 artifact로 추출한다."""
    config = load_config(config_path)
    validate_full_batch_sizes(resnet_batch_size, vgg_batch_size)
    videos = select_full_context(project_root / "data/interim/sequences_v2/context")
    mappings = load_full_mapping(project_root, videos)
    sources, source_ids = unique_source_index(mappings)
    if len(sources) != EXPECTED_FULL_COUNTS["unique_source_images"]:
        raise ValueError("STEP_5D2_BLOCKED_FULL_UNIVERSE: unique source images")
    source_mapping_fingerprint = mapping_fingerprint(mappings)
    environment = environment_preflight()
    require_dependencies(environment)
    torch = importlib.import_module("torch")
    resolved_device = ("cuda" if environment["cuda_available"] else "cpu") if device == "auto" else device
    if resolved_device == "cuda" and not environment["cuda_available"]:
        raise DependencyBlocked("STEP_5D2_BLOCKED_DEPENDENCY: CUDA requested but unavailable")
    summaries = []
    total_start = time.perf_counter()
    for backbone, batch_size in (("resnet18", resnet_batch_size), ("vgg16", vgg_batch_size)):
        expected_hash = cnn_feature_policy_hash(config, backbone, RESOLVED_WEIGHTS[backbone])
        final = output_root / backbone
        if _full_artifact_complete(output_root, backbone, expected_hash, videos,
                                   source_mapping_fingerprint):
            if not resume:
                raise FileExistsError(f"기존 {backbone} full artifact가 있습니다")
            summary = json.loads((final / "full_feature_summary.json").read_text(encoding="utf-8"))
            summary["pilot_full_regression"] = compare_pilot_full_features(
                project_root, final, backbone)
            summaries.append(summary)
            continue
        if final.exists():
            raise FileExistsError(f"STEP_5D2_RESUME_CONFLICT: {backbone}")
        load_start = time.perf_counter()
        torch_module, extractor, enum_name = build_torch_extractor(backbone, resolved_device)
        load_sec = time.perf_counter() - load_start
        if enum_name != RESOLVED_WEIGHTS[backbone]:
            raise ValueError("STEP_5D2_BLOCKED_FEATURE_POLICY_MISMATCH: resolved weights")
        policy = feature_policy(config, backbone, enum_name)
        policy_hash = stable_hash(policy)
        if policy_hash != expected_hash:
            raise ValueError("STEP_5D2_BLOCKED_FEATURE_POLICY_MISMATCH")
        if resolved_device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        extraction_start = time.perf_counter()
        source_features = _extract_unique_sources(torch_module, extractor, sources, project_root,
                                                  resolved_device, batch_size)
        extraction_sec = time.perf_counter() - extraction_start
        target_features = assemble_target_features(source_features, source_ids, len(videos))
        runtime = {"model_load_sec": load_sec, "feature_extraction_sec": extraction_sec,
                   "total_sec": load_sec + extraction_sec,
                   "source_images_per_sec": len(sources) / extraction_sec,
                   "device": resolved_device, "batch_size": batch_size, "amp": False,
                   "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                       if resolved_device == "cuda" else 0}
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".step5d2_{backbone}_", dir=output_root) as temporary:
            summary = _publish_full_backbone(
                Path(temporary), backbone, videos, mappings, sources, source_ids,
                source_features, target_features, policy, policy_hash, runtime,
                _checkpoint_provenance(torch, backbone))
            Path(temporary).rename(final)
        summary["pilot_full_regression"] = compare_pilot_full_features(
            project_root, final, backbone)
        summaries.append(summary)
        del extractor, source_features, target_features
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
    status_fields = full_run_status(summaries)
    result = {**status_fields,
              "environment": environment, "full_context": full_summary(videos),
              "unique_source_images": len(sources), "backbones": summaries,
              "source_mapping_fingerprint": source_mapping_fingerprint,
              "backbone_fusion": False, "total_wall_sec": time.perf_counter() - total_start,
              "test_access": 0}
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "step5d2_full_feature_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def full_run_status(summaries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Extraction 완료와 pilot/full integrity 상태를 서로 분리한다."""
    regression_failed = any(
        summary.get("pilot_full_regression", {}).get("status") == "FAIL"
        for summary in summaries)
    return {"status": ("STEP_5D2_FULL_EXTRACTION_COMPLETED_REVIEW_REQUIRED"
                       if regression_failed
                       else "STEP_5D2_FULL_CNN_FEATURE_EXTRACTION_COMPLETE"),
            "extraction_status": "COMPLETE",
            "integrity_status": "REVIEW_REQUIRED" if regression_failed else "PASS"}
