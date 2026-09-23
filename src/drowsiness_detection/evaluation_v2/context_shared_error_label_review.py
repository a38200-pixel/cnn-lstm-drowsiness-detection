"""STEP 6-E6 shared-error validation label consistency 수동 감사."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_CANDIDATE_COUNT = 42
KNOWN_MISMATCH_CANDIDATES = ("d_593", "d_595", "d_283", "d_627")
REVIEW_CANDIDATE_FIELDS = (
    "video_id",
    "split",
    "dataset_label",
    "resnet_prediction",
    "vgg_prediction",
    "resnet_ce",
    "vgg_ce",
    "original_video_path",
)
MANUAL_REVIEW_FIELDS = (
    "video_id",
    "dataset_label",
    "resnet_prediction",
    "vgg_prediction",
    "resnet_ce",
    "vgg_ce",
    "frame32_review",
    "original_video_review",
    "label_consistency",
    "audit_status",
    "notes",
)
FRAME32_VALUES = {
    "visually_consistent",
    "visually_ambiguous",
    "apparently_conflicting",
    "cannot_judge",
}
ORIGINAL_VIDEO_VALUES = {
    "drowsy_like", "not_drowsy_like", "ambiguous", "cannot_judge"}
LABEL_CONSISTENCY_VALUES = {"consistent", "ambiguous", "conflicting", "cannot_judge"}
AUDIT_STATUS_VALUES = {
    "reviewed_consistent",
    "reviewed_ambiguous",
    "manual_label_mismatch_candidate",
    "cannot_judge",
}
STATUS_TO_CONSISTENCY = {
    "reviewed_consistent": "consistent",
    "reviewed_ambiguous": "ambiguous",
    "manual_label_mismatch_candidate": "conflicting",
    "cannot_judge": "cannot_judge",
}
LIMITATION_EN = (
    "This audit is restricted to 42 validation samples misclassified by both "
    "ResNet18-LSTM and VGG16-LSTM. The sample was selected based on model errors "
    "and is therefore not representative of the full validation split. "
    "Observed label inconsistencies must not be interpreted as the dataset-wide "
    "label error rate. frame32_review was not required for all 42 samples; "
    "the primary evidence for E6 is manual review of the original videos."
)
LIMITATION_KO = (
    "이 감사는 ResNet18-LSTM과 VGG16-LSTM이 모두 오분류한 validation 표본 42개로 "
    "제한된다. 모델 오류를 기준으로 선택된 표본이므로 전체 validation split을 대표하지 "
    "않으며, 관찰된 label 불일치 비율을 dataset 전체 label 오류율로 해석해서는 안 된다. "
    "frame32_review는 42개 전체의 필수 항목이 아니며, E6의 주된 근거는 원본 영상에 대한 "
    "수동 검토이다."
)


class ContextSharedErrorLabelReviewError(RuntimeError):
    """후보 universe, split 보호 또는 수동 검토 schema 위반."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContextSharedErrorLabelReviewError(f"필수 CSV가 없습니다: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ContextSharedErrorLabelReviewError(f"CSV가 비어 있습니다: {path}")
    return rows


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ContextSharedErrorLabelReviewError(f"{field}에 NaN/Inf가 있습니다")
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
            raise ContextSharedErrorLabelReviewError(f"읽기 전용 입력이 없습니다: {resolved}")
        stat = resolved.stat()
        result[str(resolved)] = (stat.st_size, stat.st_mtime_ns, _sha256(resolved))
    return result


def _verify_unchanged(before: Mapping[str, tuple[int, int, str]], description: str) -> None:
    after = _fingerprints(Path(path) for path in before)
    if dict(before) != after:
        raise ContextSharedErrorLabelReviewError(f"{description}가 실행 중 변경되었습니다")


def select_both_incorrect_candidates(
    overlap_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """E4의 명시적 both_incorrect 행만 선택하고 정확히 42개인지 확인한다."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in overlap_rows:
        if source.get("category") != "both_incorrect":
            continue
        video_id = str(source.get("video_id", ""))
        if not video_id or video_id in seen:
            raise ContextSharedErrorLabelReviewError("both_incorrect video_id가 없거나 중복됩니다")
        if (str(source.get("resnet18_correct", "")).casefold() != "false"
                or str(source.get("vgg16_correct", "")).casefold() != "false"):
            raise ContextSharedErrorLabelReviewError("both_incorrect correct flag가 잘못되었습니다")
        label = str(source.get("label", ""))
        if label not in {"drowsy", "not_drowsy"}:
            raise ContextSharedErrorLabelReviewError(f"지원하지 않는 label입니다: {label}")
        seen.add(video_id)
        selected.append({
            "video_id": video_id,
            "dataset_label": label,
            "resnet_prediction": str(source["resnet18_predicted_label"]),
            "vgg_prediction": str(source["vgg16_predicted_label"]),
            "resnet_ce": _finite_float(source["resnet18_ce_loss"], "resnet_ce"),
            "vgg_ce": _finite_float(source["vgg16_ce_loss"], "vgg_ce"),
        })
    if len(selected) != EXPECTED_CANDIDATE_COUNT:
        raise ContextSharedErrorLabelReviewError(
            f"both_incorrect는 {EXPECTED_CANDIDATE_COUNT}개여야 합니다: {len(selected)}")
    missing_known = set(KNOWN_MISMATCH_CANDIDATES) - seen
    if missing_known:
        raise ContextSharedErrorLabelReviewError(
            f"기존 수동 확인 4개 중 E4 집합에 없는 video가 있습니다: {sorted(missing_known)}")
    return sorted(selected, key=lambda row: str(row["video_id"]))


def join_validation_metadata(
    project_root: Path | str,
    candidates: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, str]],
    *,
    require_video_exists: bool = True,
) -> list[dict[str, Any]]:
    """validation metadata의 기존 원본 경로만 연결하며 영상은 열지 않는다."""

    root = Path(project_root).resolve()
    raw_root = (root / "data/raw").resolve()
    metadata: dict[str, Mapping[str, str]] = {}
    for row in metadata_rows:
        split = str(row.get("split", ""))
        if split == "test":
            raise ContextSharedErrorLabelReviewError("test metadata 접근은 금지됩니다")
        if split != "val":
            raise ContextSharedErrorLabelReviewError(f"val.csv에 val 외 split이 있습니다: {split}")
        video_id = str(row.get("video_id", ""))
        if not video_id or video_id in metadata:
            raise ContextSharedErrorLabelReviewError("validation metadata video_id가 없거나 중복됩니다")
        metadata[video_id] = row

    result: list[dict[str, Any]] = []
    for candidate in candidates:
        video_id = str(candidate["video_id"])
        if video_id not in metadata:
            raise ContextSharedErrorLabelReviewError(f"validation metadata 누락: {video_id}")
        source = metadata[video_id]
        if source.get("label") != candidate["dataset_label"]:
            raise ContextSharedErrorLabelReviewError(f"E4/metadata label 불일치: {video_id}")
        relative_path = Path(str(source.get("video_path", "")))
        if relative_path.is_absolute() or "test" in relative_path.parts:
            raise ContextSharedErrorLabelReviewError(f"금지된 원본 영상 경로입니다: {relative_path}")
        resolved = (root / relative_path).resolve()
        if not resolved.is_relative_to(raw_root):
            raise ContextSharedErrorLabelReviewError(f"data/raw 밖의 영상 경로입니다: {resolved}")
        if require_video_exists and not resolved.is_file():
            raise ContextSharedErrorLabelReviewError(f"원본 validation 영상이 없습니다: {resolved}")
        result.append({**dict(candidate), "split": "val",
                       "original_video_path": relative_path.as_posix()})
    return result


def build_manual_review_template(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """기존 원본 영상 확인 4개만 고정 값으로 prefill한다."""

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = {field: candidate.get(field, "") for field in MANUAL_REVIEW_FIELDS}
        if candidate["video_id"] in KNOWN_MISMATCH_CANDIDATES:
            if candidate["dataset_label"] != "drowsy":
                raise ContextSharedErrorLabelReviewError(
                    f"기존 mismatch candidate의 dataset label이 drowsy가 아닙니다: {candidate['video_id']}")
            row.update({
                "frame32_review": "apparently_conflicting",
                "original_video_review": "not_drowsy_like",
                "label_consistency": "conflicting",
                "audit_status": "manual_label_mismatch_candidate",
                "notes": "Existing manual review: no observable drowsiness cue.",
            })
        rows.append(row)
    validate_known_prefill(rows)
    return rows


def validate_known_prefill(rows: Sequence[Mapping[str, Any]]) -> None:
    keyed = {str(row["video_id"]): row for row in rows}
    expected = {
        "dataset_label": "drowsy",
        "frame32_review": "apparently_conflicting",
        "original_video_review": "not_drowsy_like",
        "label_consistency": "conflicting",
        "audit_status": "manual_label_mismatch_candidate",
    }
    for video_id in KNOWN_MISMATCH_CANDIDATES:
        if video_id not in keyed:
            raise ContextSharedErrorLabelReviewError(f"prefill 대상 누락: {video_id}")
        mismatches = {field: keyed[video_id].get(field)
                      for field, value in expected.items() if keyed[video_id].get(field) != value}
        if mismatches:
            raise ContextSharedErrorLabelReviewError(
                f"기존 수동 확인 prefill 불일치: {video_id} {mismatches}")


def prepare_context_shared_error_label_review(
    project_root: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """원본 영상은 열지 않고 review candidate와 manual template만 만든다."""

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ContextSharedErrorLabelReviewError(
            f"기존 결과 overwrite를 차단합니다: {output}")
    overlap_path = root / "outputs/audits_v2/context_val_hard_samples/backbone_error_overlap.csv"
    metadata_path = root / "data/metadata/val.csv"
    inputs_before = _fingerprints((overlap_path, metadata_path))
    candidates = select_both_incorrect_candidates(_read_csv(overlap_path))
    candidates = join_validation_metadata(root, candidates, _read_csv(metadata_path))
    manual_rows = build_manual_review_template(candidates)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}_", dir=output.parent) as temp_name:
        staging = Path(temp_name) / output.name
        staging.mkdir()
        _write_csv(staging / "review_candidates.csv", REVIEW_CANDIDATE_FIELDS, candidates)
        _write_csv(staging / "manual_label_review.csv", MANUAL_REVIEW_FIELDS, manual_rows)
        _verify_unchanged(inputs_before, "E4/validation metadata")
        staging.replace(output)
    return {
        "status": "WAITING_FOR_MANUAL_ORIGINAL_VIDEO_REVIEW",
        "candidate_count": len(candidates),
        "known_manual_mismatch_candidates": list(KNOWN_MISMATCH_CANDIDATES),
        "original_videos_opened": 0,
        "model_inference": False,
        "training": False,
        "test_access_count": 0,
        "output_dir": str(output),
        "limitation_en": LIMITATION_EN,
        "limitation_ko": LIMITATION_KO,
    }


def validate_and_summarize_manual_review(
    candidate_rows: Sequence[Mapping[str, str]],
    review_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """완료된 42개 review를 검증하고 전체/label별 count와 rate를 계산한다."""

    candidates = {str(row["video_id"]): row for row in candidate_rows}
    reviews = {str(row["video_id"]): row for row in review_rows}
    if (len(candidates) != EXPECTED_CANDIDATE_COUNT
            or len(reviews) != EXPECTED_CANDIDATE_COUNT
            or set(candidates) != set(reviews)):
        raise ContextSharedErrorLabelReviewError("candidate/review universe는 같은 42개여야 합니다")
    identity_fields = (
        "dataset_label", "resnet_prediction", "vgg_prediction", "resnet_ce", "vgg_ce")
    for video_id, review in reviews.items():
        candidate = candidates[video_id]
        for field in identity_fields:
            if str(review.get(field, "")) != str(candidate.get(field, "")):
                raise ContextSharedErrorLabelReviewError(
                    f"manual review identity field 변경 금지: {video_id}/{field}")
        frame32_review = str(review.get("frame32_review", ""))
        if frame32_review and frame32_review not in FRAME32_VALUES:
            raise ContextSharedErrorLabelReviewError(
                f"frame32_review 허용값 위반: {video_id}")
        if review.get("original_video_review") not in ORIGINAL_VIDEO_VALUES:
            raise ContextSharedErrorLabelReviewError(
                f"original_video_review 미완료/허용값 위반: {video_id}")
        if review.get("label_consistency") not in LABEL_CONSISTENCY_VALUES:
            raise ContextSharedErrorLabelReviewError(
                f"label_consistency 미완료/허용값 위반: {video_id}")
        status = str(review.get("audit_status", ""))
        if status not in AUDIT_STATUS_VALUES:
            raise ContextSharedErrorLabelReviewError(
                f"audit_status 미완료/허용값 위반: {video_id}")
        if review["label_consistency"] != STATUS_TO_CONSISTENCY[status]:
            raise ContextSharedErrorLabelReviewError(
                f"label_consistency/audit_status 불일치: {video_id}")
    validate_known_prefill(review_rows)

    def counts_and_rates(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        counts = Counter(str(row["label_consistency"]) for row in rows)
        return {
            value: {"count": counts[value], "rate": counts[value] / len(rows)}
            for value in ("consistent", "ambiguous", "conflicting", "cannot_judge")
        }

    by_label = {
        label: counts_and_rates([
            row for row in review_rows if row["dataset_label"] == label])
        for label in ("drowsy", "not_drowsy")
    }
    mismatches = sorted(
        row["video_id"] for row in review_rows
        if row["audit_status"] == "manual_label_mismatch_candidate")
    frame32_reviewed_count = sum(
        bool(str(row.get("frame32_review", "")).strip()) for row in review_rows)
    return {
        "step": "6-E6",
        "status": "MANUAL_REVIEW_SUMMARIZED",
        "scope": {
            "selection": "both_backbones_incorrect",
            "validation_sample_count": EXPECTED_CANDIDATE_COUNT,
            "representative_of_full_validation": False,
        },
        "overall": counts_and_rates(review_rows),
        "by_dataset_label": by_label,
        "frame32_review_coverage": {
            "reviewed_count": frame32_reviewed_count,
            "total_count": EXPECTED_CANDIDATE_COUNT,
            "required": False,
        },
        "manual_label_mismatch_candidate_video_ids": mismatches,
        "limitations": {"en": LIMITATION_EN, "ko": LIMITATION_KO},
        "protections": {
            "original_metadata_label_changed": False,
            "split_changed": False,
            "samples_excluded": False,
            "training": False,
            "model_inference": False,
            "feature_extraction": False,
            "test_access_count": 0,
        },
    }


def summarize_context_shared_error_label_review(output_dir: Path | str) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ContextSharedErrorLabelReviewError(
            f"기존 summary overwrite를 차단합니다: {summary_path}")
    summary = validate_and_summarize_manual_review(
        _read_csv(output / "review_candidates.csv"),
        _read_csv(output / "manual_label_review.csv"),
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
