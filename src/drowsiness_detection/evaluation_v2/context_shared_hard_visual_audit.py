"""STEP 6-E5 shared validation hard-sample 시각 감사 준비 및 요약."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from drowsiness_detection.sequences_v2.context_sequence_visualizer import (
    load_context_sequence,
    render_contact_sheet,
)


BACKBONES = ("resnet18", "vgg16")
EXPECTED_HARD_COUNT = 13
CONTROL_PER_LABEL = 6
LABELS = ("drowsy", "not_drowsy")
MANUAL_FIELDS = (
    "video_id",
    "group",
    "label",
    "face_visible",
    "major_occlusion",
    "lighting_issue",
    "extreme_head_pose",
    "framing_issue",
    "eye_closure_visible",
    "yawning_visible",
    "head_nod_or_pose_change",
    "drowsiness_cue_duration",
    "label_impression",
    "notes",
)
AUDIT_FIELDS = (
    "video_id",
    "split",
    "label",
    "group",
    "selection_rank",
    "resnet_predicted_label",
    "resnet_ce",
    "resnet_confidence",
    "vgg_predicted_label",
    "vgg_ce",
    "vgg_confidence",
    "imputed_count",
    "contact_sheet_relpath",
)
ALLOWED_MANUAL_VALUES = {
    "face_visible": {"yes", "partial", "poor"},
    "major_occlusion": {"yes", "no"},
    "lighting_issue": {"yes", "no"},
    "extreme_head_pose": {"yes", "no"},
    "framing_issue": {"yes", "no"},
    "eye_closure_visible": {"clear", "weak", "none", "unclear"},
    "yawning_visible": {"clear", "weak", "none", "unclear"},
    "head_nod_or_pose_change": {"clear", "weak", "none", "unclear"},
    "drowsiness_cue_duration": {"sustained", "brief", "none_visible", "unclear"},
    "label_impression": {
        "visually_consistent", "visually_ambiguous",
        "apparently_conflicting", "cannot_judge",
    },
}


class ContextSharedHardVisualAuditError(RuntimeError):
    """후보 계약, split 보호 또는 수동 검토 schema 위반."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContextSharedHardVisualAuditError(f"필수 CSV가 없습니다: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ContextSharedHardVisualAuditError(f"CSV가 비어 있습니다: {path}")
    return rows


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ContextSharedHardVisualAuditError(f"{field}에 NaN/Inf가 있습니다")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprints(paths: Iterable[Path]) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ContextSharedHardVisualAuditError(f"읽기 전용 artifact가 없습니다: {resolved}")
        stat = resolved.stat()
        result[str(resolved)] = (stat.st_size, stat.st_mtime_ns, _sha256(resolved))
    return result


def _verify_unchanged(before: Mapping[str, tuple[int, int, str]], description: str) -> None:
    after = _fingerprints(Path(path) for path in before)
    if dict(before) != after:
        raise ContextSharedHardVisualAuditError(f"{description}가 감사 중 변경되었습니다")


def _per_video_backbone_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Mapping[str, str]]]:
    result: dict[str, dict[str, Mapping[str, str]]] = {}
    for row in rows:
        backbone = str(row.get("backbone", ""))
        if backbone not in BACKBONES:
            raise ContextSharedHardVisualAuditError(f"지원하지 않는 backbone입니다: {backbone}")
        video_id = str(row.get("video_id", ""))
        if not video_id or backbone in result.setdefault(video_id, {}):
            raise ContextSharedHardVisualAuditError("video/backbone row가 없거나 중복됩니다")
        result[video_id][backbone] = row
    if any(set(by_backbone) != set(BACKBONES) for by_backbone in result.values()):
        raise ContextSharedHardVisualAuditError("모든 video에 두 backbone 결과가 필요합니다")
    return result


def select_visual_audit_candidates(
    per_sample_rows: Sequence[Mapping[str, str]],
    hard_sample_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """E4 공통 Top-20 13개와 class-balanced shared-easy control을 고른다."""

    paired = _per_video_backbone_rows(per_sample_rows)
    top20 = {
        backbone: {
            str(row["video_id"])
            for row in hard_sample_rows
            if row.get("candidate_type") == "loss_top20" and row.get("backbone") == backbone
        }
        for backbone in BACKBONES
    }
    if any(len(top20[backbone]) != 20 for backbone in BACKBONES):
        raise ContextSharedHardVisualAuditError("각 backbone의 E4 loss Top-20이 정확히 필요합니다")
    hard_ids = top20["resnet18"] & top20["vgg16"]
    if len(hard_ids) != EXPECTED_HARD_COUNT:
        raise ContextSharedHardVisualAuditError(
            f"공통 hard set은 {EXPECTED_HARD_COUNT}개여야 합니다: {len(hard_ids)}")

    def combined(video_id: str, group: str, rank: int) -> dict[str, Any]:
        resnet, vgg = paired[video_id]["resnet18"], paired[video_id]["vgg16"]
        if resnet["label"] != vgg["label"]:
            raise ContextSharedHardVisualAuditError(f"backbone label 불일치: {video_id}")
        if int(resnet["imputed_count"]) != int(vgg["imputed_count"]):
            raise ContextSharedHardVisualAuditError(f"backbone imputed_count 불일치: {video_id}")
        return {
            "video_id": video_id,
            "split": "val",
            "label": resnet["label"],
            "group": group,
            "selection_rank": rank,
            "resnet_predicted_label": resnet["predicted_label"],
            "resnet_ce": _finite_float(resnet["per_sample_ce_loss"], "resnet_ce"),
            "resnet_confidence": _finite_float(
                resnet["predicted_class_probability"], "resnet_confidence"),
            "vgg_predicted_label": vgg["predicted_label"],
            "vgg_ce": _finite_float(vgg["per_sample_ce_loss"], "vgg_ce"),
            "vgg_confidence": _finite_float(
                vgg["predicted_class_probability"], "vgg_confidence"),
            "imputed_count": int(resnet["imputed_count"]),
            "contact_sheet_relpath": f"contact_sheets/{group}/{rank:02d}_{video_id}.jpg",
        }

    hard_order = sorted(
        hard_ids,
        key=lambda video_id: (
            -max(float(paired[video_id][name]["per_sample_ce_loss"]) for name in BACKBONES),
            -sum(float(paired[video_id][name]["per_sample_ce_loss"]) for name in BACKBONES),
            video_id,
        ),
    )
    selected = [combined(video_id, "hard", rank)
                for rank, video_id in enumerate(hard_order, start=1)]

    # 두 모델 모두 맞힌 후보 중 worst-backbone CE가 가장 낮은 순서로 뽑는다.
    # 따라서 한 모델만 매우 쉬운 표본이 control을 독점하지 않는다.
    for label in LABELS:
        eligible = []
        for video_id, by_backbone in paired.items():
            if video_id in hard_ids or by_backbone["resnet18"]["label"] != label:
                continue
            if not all(str(by_backbone[name]["correct"]).casefold() == "true"
                       for name in BACKBONES):
                continue
            ce_values = [float(by_backbone[name]["per_sample_ce_loss"]) for name in BACKBONES]
            eligible.append((max(ce_values), sum(ce_values) / 2.0, video_id))
        eligible.sort()
        if len(eligible) < CONTROL_PER_LABEL:
            raise ContextSharedHardVisualAuditError(f"{label} control 후보가 부족합니다")
        for class_rank, (_, _, video_id) in enumerate(
            eligible[:CONTROL_PER_LABEL], start=1
        ):
            row = combined(video_id, "control", class_rank)
            row["contact_sheet_relpath"] = (
                f"contact_sheets/control/{label}_{class_rank:02d}_{video_id}.jpg")
            selected.append(row)
    if len(selected) != EXPECTED_HARD_COUNT + CONTROL_PER_LABEL * len(LABELS):
        raise ContextSharedHardVisualAuditError("최종 후보 수가 잘못되었습니다")
    return selected


def _manual_template(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row[field] if field in ("video_id", "group", "label") else ""
             for field in MANUAL_FIELDS}
            for row in candidates]


def save_contact_sheet(path: Path, sheet: np.ndarray) -> None:
    """이미 구성된 4×8 sheet만 JPEG로 저장한다. frame sampling은 수행하지 않는다."""

    if sheet.ndim != 3 or sheet.shape[2] != 3 or sheet.dtype != np.uint8:
        raise ContextSharedHardVisualAuditError("contact sheet image 계약이 잘못되었습니다")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ContextSharedHardVisualAuditError("contact sheet JPEG encoding에 실패했습니다")
    encoded.tofile(path)


def prepare_context_shared_hard_visual_audit(
    project_root: Path | str,
    output_dir: Path | str,
    *,
    tile_size: int = 180,
) -> dict[str, Any]:
    """E4 CSV와 기존 validation Context 32-frame만 이용해 review pack을 만든다."""

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ContextSharedHardVisualAuditError(
            f"기존 수동 검토 결과 보호를 위해 output overwrite를 차단합니다: {output}")
    if tile_size < 96:
        raise ContextSharedHardVisualAuditError("tile_size는 96 이상이어야 합니다")
    e4_dir = root / "outputs/audits_v2/context_val_hard_samples"
    e4_paths = [e4_dir / "per_sample_loss.csv", e4_dir / "hard_sample_top20.csv",
                e4_dir / "summary.json"]
    e4_before = _fingerprints(e4_paths)
    e4_summary = json.loads(e4_paths[2].read_text(encoding="utf-8"))
    if e4_summary.get("classification") != "SHARED_BACKBONE_HARD_SAMPLES":
        raise ContextSharedHardVisualAuditError("E4 classification 계약이 다릅니다")
    candidates = select_visual_audit_candidates(
        _read_csv(e4_paths[0]), _read_csv(e4_paths[1]))

    output.parent.mkdir(parents=True, exist_ok=True)
    source_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=f".{output.name}_", dir=output.parent) as temp_name:
        staging = Path(temp_name) / output.name
        staging.mkdir()
        for candidate in candidates:
            # split을 명시적으로 val로 고정한다. 자동 train/test 탐색은 하지 않는다.
            rows, images, sequence_path = load_context_sequence(
                root, str(candidate["video_id"]), "val")
            if rows[0]["label"] != candidate["label"]:
                raise ContextSharedHardVisualAuditError(
                    f"E4/sequence label 불일치: {candidate['video_id']}")
            if sum(bool(row["imputed"]) for row in rows) != int(candidate["imputed_count"]):
                raise ContextSharedHardVisualAuditError(
                    f"E4/sequence imputed_count 불일치: {candidate['video_id']}")
            paths = [sequence_path, *(Path(row["source_path"]) for row in rows)]
            before = _fingerprints(paths)
            sheet = render_contact_sheet(rows, images, tile_size=tile_size)
            expected_shape = (4 * (tile_size + 62), 8 * tile_size, 3)
            if sheet.shape != expected_shape:
                raise ContextSharedHardVisualAuditError(
                    f"4x8 contact sheet shape가 아닙니다: {sheet.shape}")
            save_contact_sheet(staging / str(candidate["contact_sheet_relpath"]), sheet)
            _verify_unchanged(before, f"{candidate['video_id']} canonical sequence/image")
            source_paths.extend(paths)

        _write_csv(staging / "audit_candidates.csv", AUDIT_FIELDS, candidates)
        _write_csv(staging / "manual_review.csv", MANUAL_FIELDS, _manual_template(candidates))
        _verify_unchanged(e4_before, "E4 output")
        staging.replace(output)

    return {
        "status": "WAITING_FOR_MANUAL_REVIEW",
        "hard_count": EXPECTED_HARD_COUNT,
        "control_count": CONTROL_PER_LABEL * len(LABELS),
        "control_label_counts": {label: CONTROL_PER_LABEL for label in LABELS},
        "contact_sheet_layout": "4x8",
        "new_sampling": False,
        "model_inference": False,
        "training": False,
        "test_access_count": 0,
        "source_artifact_count_read_only": len(set(source_paths)),
        "output_dir": str(output),
    }


def _rate(rows: Sequence[Mapping[str, str]], predicate: Any) -> dict[str, Any]:
    count = sum(bool(predicate(row)) for row in rows)
    return {"count": count, "total": len(rows), "rate": count / len(rows)}


def _group_review_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "face_visibility_problem": _rate(rows, lambda row: row["face_visible"] in {"partial", "poor"}),
        "major_occlusion": _rate(rows, lambda row: row["major_occlusion"] == "yes"),
        "lighting_issue": _rate(rows, lambda row: row["lighting_issue"] == "yes"),
        "extreme_head_pose": _rate(rows, lambda row: row["extreme_head_pose"] == "yes"),
        "framing_issue": _rate(rows, lambda row: row["framing_issue"] == "yes"),
        "brief_cue": _rate(rows, lambda row: row["drowsiness_cue_duration"] == "brief"),
        "visually_ambiguous": _rate(rows, lambda row: row["label_impression"] == "visually_ambiguous"),
        "label_concern": _rate(
            rows,
            lambda row: row["label_impression"] in {
                "visually_ambiguous", "apparently_conflicting"},
        ),
    }


def classify_manual_review(
    hard: Mapping[str, Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """단순 count/rate 차이에 기반한 descriptive 분류만 수행한다."""

    quality_fields = (
        "face_visibility_problem", "major_occlusion", "lighting_issue",
        "extreme_head_pose", "framing_issue",
    )
    repeated_quality = [field for field in quality_fields
                        if hard[field]["rate"] >= 0.50
                        and hard[field]["rate"] - control[field]["rate"] >= 0.20]
    brief = (hard["brief_cue"]["rate"] >= 0.50
             and hard["brief_cue"]["rate"] - control["brief_cue"]["rate"] >= 0.20)
    ambiguity = (hard["label_concern"]["rate"] >= 0.50
                 and hard["label_concern"]["rate"] - control["label_concern"]["rate"] >= 0.20)
    patterns = []
    if repeated_quality:
        patterns.append("VISUAL_DATA_QUALITY_PATTERN")
    if brief:
        patterns.append("BRIEF_OR_AMBIGUOUS_TEMPORAL_CUES")
    if ambiguity:
        patterns.append("POSSIBLE_LABEL_AMBIGUITY")
    classification = (
        "NO_OBVIOUS_VISUAL_EXPLANATION" if not patterns
        else patterns[0] if len(patterns) == 1
        else "MIXED"
    )
    return classification, {
        "matched_patterns": patterns,
        "repeated_quality_fields": repeated_quality,
        "rule": "hard rate >= 0.50 and hard-control difference >= 0.20",
        "interpretation": "descriptive only; not a label-error or model-cause determination",
    }


def summarize_context_shared_hard_visual_audit(output_dir: Path | str) -> dict[str, Any]:
    """사람이 완료한 manual_review.csv를 count/rate 수준으로 요약한다."""

    output = Path(output_dir).resolve()
    candidate_rows = _read_csv(output / "audit_candidates.csv")
    review_rows = _read_csv(output / "manual_review.csv")
    candidates = {row["video_id"]: row for row in candidate_rows}
    reviews = {row["video_id"]: row for row in review_rows}
    if len(candidates) != 25 or len(reviews) != 25 or set(candidates) != set(reviews):
        raise ContextSharedHardVisualAuditError("candidate/manual review universe는 같은 25개여야 합니다")
    for video_id, row in reviews.items():
        source = candidates[video_id]
        if row["group"] != source["group"] or row["label"] != source["label"]:
            raise ContextSharedHardVisualAuditError(f"manual identity field 변경 금지: {video_id}")
        for field, allowed in ALLOWED_MANUAL_VALUES.items():
            if row[field] not in allowed:
                raise ContextSharedHardVisualAuditError(
                    f"수동 검토 미완료/허용값 위반: {video_id}/{field}={row[field]!r}")
    grouped = {
        group: [row for row in review_rows if row["group"] == group]
        for group in ("hard", "control")
    }
    if len(grouped["hard"]) != 13 or len(grouped["control"]) != 12:
        raise ContextSharedHardVisualAuditError("hard/control count가 13/12가 아닙니다")
    group_summary = {group: _group_review_summary(rows) for group, rows in grouped.items()}
    classification, evidence = classify_manual_review(
        group_summary["hard"], group_summary["control"])
    summary = {
        "step": "6-E5",
        "status": "MANUAL_REVIEW_SUMMARIZED",
        "classification": classification,
        "classification_evidence": evidence,
        "groups": group_summary,
        "candidate_counts": dict(Counter(row["group"] for row in candidate_rows)),
        "prior_audits": {
            "E1": "NO_OBVIOUS_SEQUENCE_QUALITY_SHIFT",
            "E2": "NO_OBVIOUS_FEATURE_DISTRIBUTION_SHIFT",
            "E3": "NO_OBVIOUS_TEMPORAL_SHIFT",
            "E4": "SHARED_BACKBONE_HARD_SAMPLES",
        },
        "protections": {
            "automatic_relabel": False,
            "automatic_exclusion": False,
            "training": False,
            "model_inference": False,
            "test_access_count": 0,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
