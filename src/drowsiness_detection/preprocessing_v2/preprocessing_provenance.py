"""Canonical 전처리의 정책과 원본 파일 provenance를 계산한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    """모델 파일을 streaming 방식으로 한 번 해시한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Mapping[str, Any]) -> str:
    """키 순서와 공백에 독립적인 SHA256을 반환한다."""

    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_fingerprint(path: Path) -> dict[str, Any]:
    """전체 영상 해시 대신 경로·크기·수정시각을 기록한다."""

    stat = path.stat()
    return {"source_path": str(path.resolve()), "source_file_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def policy_payload(config: Mapping[str, Any], yunet_sha256: str, dlib_sha256: str) -> dict[str, Any]:
    """출력 경로·로그·모드를 제외한 전처리 의미만 고정한다."""

    return {
        "sampling": config["sampling"], "context": config["context"],
        "detector": {**config["detector"], "model_path": None, "model_sha256": yunet_sha256},
        "landmark": {"roi_policy": config["landmark"]["roi_policy"], "model_sha256": dlib_sha256},
        "face_selection": "largest_valid_bbox", "fallback": "none",
        "feature_versions": {"ear": "step2_dlib68_v1", "mar": "step2_dlib68_v1", "pose": "step2_solvepnp_v1"},
        "context_index_policy": "half_up_linspace_0_99_32_v1",
        "resize_policy": "area_if_both_shrink_else_linear",
    }
