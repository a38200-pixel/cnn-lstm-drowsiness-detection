"""영상을 수정하거나 전처리하지 않고 2차 실험 원본 데이터를 검증한다."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mov", ".mkv"})
METADATA_FILENAMES = (
    "sust_ddd_video_metadata.csv",
    "train.csv",
    "val.csv",
    "test.csv",
    "split_summary.csv",
)
SPLIT_NAMES = ("train", "val", "test")
REPORT_FILENAMES = (
    "source_validation_report.txt",
    "source_validation_summary.json",
    "metadata_schema.json",
    "split_integrity.json",
    "raw_video_inventory.csv",
)


@dataclass(frozen=True)
class ValidationConfig:
    """한 번의 검증 실행에 사용할 입력 및 출력 경로 설정."""

    project_root: Path
    raw_dir: Path
    metadata_dir: Path
    output_dir: Path
    dlib_predictor: Path
    yunet_model: Path
    samples_per_label: int = 1
    skip_decode: bool = False
    skip_asset_load: bool = False


@dataclass
class CsvData:
    """파싱한 CSV 내용과 추론한 스키마 정보."""

    name: str
    path: Path
    columns: list[str]
    rows: list[dict[str, str]]
    schema: dict[str, Any]


def resolve_path(value: str | Path, project_root: Path) -> Path:
    """CLI 경로를 프로젝트 루트 기준의 절대 경로로 변환한다."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def build_config(
    project_root: str | Path,
    raw_dir: str | Path,
    metadata_dir: str | Path,
    output_dir: str | Path,
    dlib_predictor: str | Path,
    yunet_model: str | Path,
    samples_per_label: int = 1,
    skip_decode: bool = False,
    skip_asset_load: bool = False,
) -> ValidationConfig:
    """CLI에서 받은 경로 값으로 검증 설정을 구성한다."""

    root = Path(project_root).expanduser().resolve()
    if samples_per_label < 1:
        raise ValueError("samples_per_label must be at least 1")
    return ValidationConfig(
        project_root=root,
        raw_dir=resolve_path(raw_dir, root),
        metadata_dir=resolve_path(metadata_dir, root),
        output_dir=resolve_path(output_dir, root),
        dlib_predictor=resolve_path(dlib_predictor, root),
        yunet_model=resolve_path(yunet_model, root),
        samples_per_label=samples_per_label,
        skip_decode=skip_decode,
        skip_asset_load=skip_asset_load,
    )


def _display_path(path: Path, project_root: Path) -> str:
    """가능하면 재현 가능한 프로젝트 상대 경로를 반환한다."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def check_required_paths(config: ValidationConfig) -> list[dict[str, Any]]:
    """원본을 변경하지 않고 모든 필수 경로의 존재 여부를 확인한다."""

    checks = [
        ("raw SUST-DDD dataset directory", config.raw_dir, "directory"),
        *[
            (filename, config.metadata_dir / filename, "file")
            for filename in METADATA_FILENAMES
        ],
        ("Dlib68 predictor", config.dlib_predictor, "file"),
        ("YuNet ONNX", config.yunet_model, "file"),
    ]
    results: list[dict[str, Any]] = []
    for label, path, expected_type in checks:
        found = path.is_dir() if expected_type == "directory" else path.is_file()
        results.append(
            {
                "item": label,
                "path": _display_path(path, config.project_root),
                "expected_type": expected_type,
                "status": "FOUND" if found else "MISSING",
            }
        )
    return results


def _normalise_column_name(name: str) -> str:
    """보수적인 후보 탐색을 위해 컬럼 이름을 정규화한다."""

    return "_".join(name.strip().lower().replace("-", "_").split())


def infer_column_candidates(columns: Sequence[str]) -> dict[str, list[str]]:
    """실제 컬럼에서 식별자, 라벨, 원본 경로 후보를 추론한다."""

    normalised = {column: _normalise_column_name(column) for column in columns}

    def ordered_candidates(
        preferred: Sequence[str], predicate: Any
    ) -> list[str]:
        exact = [
            column
            for expected in preferred
            for column, value in normalised.items()
            if value == expected
        ]
        inferred = [
            column
            for column, value in normalised.items()
            if column not in exact and predicate(value)
        ]
        return exact + inferred

    video_ids = ordered_candidates(
        ("video_id", "source_video", "filename", "file_name"),
        lambda value: (
            ("video" in value and "id" in value)
            or "filename" in value
            or value.endswith("file_name")
        ),
    )
    labels = ordered_candidates(
        ("label", "class", "target", "status"),
        lambda value: any(token in value for token in ("label", "class", "target")),
    )
    source_paths = ordered_candidates(
        ("video_path", "source_path", "path", "filepath", "file_path"),
        lambda value: "path" in value or "source" in value or "filename" in value,
    )
    return {
        "video_id_candidates": video_ids,
        "label_candidates": labels,
        "source_path_candidates": source_paths,
    }


def read_csv_data(path: Path) -> CsvData:
    """CSV 하나를 파싱하고 구조 진단 정보를 계산한다."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            columns = [column.strip() for column in reader.fieldnames]
            if any(not column for column in columns):
                raise ValueError("CSV contains an empty column name")
            if len(columns) != len(set(columns)):
                raise ValueError("CSV contains duplicate column names")

            rows: list[dict[str, str]] = []
            for row_number, raw_row in enumerate(reader, start=2):
                if None in raw_row:
                    raise ValueError(
                        f"row {row_number} has more values than the header"
                    )
                rows.append(
                    {
                        column: (raw_row.get(original) or "").strip()
                        for original, column in zip(reader.fieldnames, columns)
                    }
                )
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV is not valid UTF-8: {path}") from exc
    except csv.Error as exc:
        raise ValueError(f"CSV parsing failed for {path}: {exc}") from exc

    row_signatures = [tuple(row[column] for column in columns) for row in rows]
    duplicate_row_count = len(row_signatures) - len(set(row_signatures))
    null_counts = {
        column: sum(1 for row in rows if not row[column]) for column in columns
    }
    candidates = infer_column_candidates(columns)
    schema = {
        "path": str(path.resolve()),
        "row_count": len(rows),
        "columns": columns,
        "null_count_total": sum(null_counts.values()),
        "null_counts_by_column": null_counts,
        "duplicate_row_count": duplicate_row_count,
        **candidates,
        "selected_video_id_column": _first_or_none(candidates["video_id_candidates"]),
        "selected_label_column": _first_or_none(candidates["label_candidates"]),
        "selected_source_path_column": _first_or_none(
            candidates["source_path_candidates"]
        ),
    }
    return CsvData(path.name, path, columns, rows, schema)


def _first_or_none(values: Sequence[str]) -> str | None:
    """첫 번째 후보를 반환하고 후보가 없으면 None을 반환한다."""

    return values[0] if values else None


def load_metadata(metadata_dir: Path) -> tuple[dict[str, CsvData], list[str]]:
    """사용 가능한 metadata CSV를 읽고 파싱 오류를 보존한다."""

    datasets: dict[str, CsvData] = {}
    errors: list[str] = []
    for filename in METADATA_FILENAMES:
        path = metadata_dir / filename
        if not path.is_file():
            continue
        try:
            datasets[filename] = read_csv_data(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{filename}: {exc}")
    return datasets, errors


def _selected_column(data: CsvData, kind: str) -> str | None:
    """CSV 프로필에서 선택된 추론 컬럼을 가져온다."""

    return data.schema.get(f"selected_{kind}_column")


def analyse_split_integrity(
    datasets: Mapping[str, CsvData],
) -> tuple[dict[str, Any], list[str]]:
    """split 내부 중복과 frozen split 사이의 교집합을 측정한다."""

    errors: list[str] = []
    split_ids: dict[str, set[str]] = {}
    split_details: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        filename = f"{split}.csv"
        data = datasets.get(filename)
        if data is None:
            errors.append(f"Cannot inspect {split}: {filename} was not loaded")
            continue
        id_column = _selected_column(data, "video_id")
        if id_column is None:
            errors.append(f"Cannot inspect {split}: no video identifier column found")
            continue
        ids = [row[id_column] for row in data.rows if row[id_column]]
        counts = Counter(ids)
        duplicate_ids = sorted(value for value, count in counts.items() if count > 1)
        split_ids[split] = set(ids)
        split_details[split] = {
            "video_id_column": id_column,
            "row_count": len(data.rows),
            "non_empty_id_count": len(ids),
            "unique_video_id_count": len(split_ids[split]),
            "duplicate_video_id_count": len(duplicate_ids),
            "duplicate_video_id_examples": duplicate_ids[:20],
        }
        if len(ids) != len(data.rows):
            errors.append(f"{split} contains {len(data.rows) - len(ids)} empty video IDs")
        if duplicate_ids:
            errors.append(
                f"{split} contains {len(duplicate_ids)} duplicated video IDs"
            )

    overlap: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        key = f"{left}_vs_{right}"
        values = sorted(split_ids.get(left, set()) & split_ids.get(right, set()))
        overlap[key] = {"count": len(values), "examples": values[:20]}
        if values:
            errors.append(f"{left} vs {right} overlap contains {len(values)} video IDs")

    return {"splits": split_details, "pairwise_overlap": overlap}, errors


def analyse_label_distribution(
    datasets: Mapping[str, CsvData],
) -> tuple[dict[str, Any], list[str]]:
    """각 frozen split의 라벨별 개수와 비율을 계산한다."""

    distributions: dict[str, Any] = {}
    warnings: list[str] = []
    for split in SPLIT_NAMES:
        data = datasets.get(f"{split}.csv")
        if data is None:
            continue
        label_column = _selected_column(data, "label")
        id_column = _selected_column(data, "video_id")
        if label_column is None:
            warnings.append(f"{split}: no label column was found")
            continue

        # Video-level metadata는 ID마다 한 행이어야 한다. ID가 중복되면 영상을
        # 한 번만 집계하고 진단 보고서에는 첫 번째 라벨을 유지한다.
        video_labels: dict[str, str] = {}
        conflicting_ids: list[str] = []
        for row_number, row in enumerate(data.rows, start=2):
            video_id = row.get(id_column, "") if id_column else f"row:{row_number}"
            label = row[label_column] or "<EMPTY>"
            if video_id in video_labels and video_labels[video_id] != label:
                conflicting_ids.append(video_id)
            video_labels.setdefault(video_id, label)
        counts = Counter(video_labels.values())
        total = len(video_labels)
        distributions[split] = {
            "video_id_column": id_column,
            "label_column": label_column,
            "total_videos": total,
            "labels": {
                label: {
                    "count": count,
                    "ratio": (count / total) if total else 0.0,
                }
                for label, count in sorted(counts.items())
            },
            "conflicting_label_id_count": len(set(conflicting_ids)),
            "conflicting_label_id_examples": sorted(set(conflicting_ids))[:20],
        }
        if conflicting_ids:
            warnings.append(
                f"{split}: {len(set(conflicting_ids))} video IDs have conflicting labels"
            )
    return distributions, warnings


def _normalise_reference(value: str) -> str:
    """원본 파일명을 바꾸지 않고 구분자와 대소문자 표현을 정규화한다."""

    return value.strip().replace("\\", "/").lstrip("./").casefold()


def inventory_raw_videos(
    raw_dir: Path, project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """지원하는 영상을 조사하고 경로, 파일명, stem 검색 인덱스를 구성한다."""

    inventory: list[dict[str, Any]] = []
    indexes: dict[str, list[int]] = defaultdict(list)
    if not raw_dir.is_dir():
        return inventory, indexes

    for path in sorted(raw_dir.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.casefold() not in VIDEO_EXTENSIONS:
            continue
        relative_project = _display_path(path, project_root)
        relative_raw = path.relative_to(raw_dir).as_posix()
        index = len(inventory)
        inventory.append(
            {
                "relative_path": relative_project,
                "raw_relative_path": relative_raw,
                "filename": path.name,
                "stem": path.stem,
                "extension": path.suffix.casefold(),
                "size_bytes": path.stat().st_size,
                "matched_metadata": False,
            }
        )
        for key in {
            f"project:{_normalise_reference(relative_project)}",
            f"raw:{_normalise_reference(relative_raw)}",
            f"name:{path.name.casefold()}",
            f"stem:{path.stem.casefold()}",
        }:
            indexes[key].append(index)
    return inventory, indexes


def _candidate_inventory_indexes(
    row: Mapping[str, str],
    id_column: str | None,
    path_column: str | None,
    indexes: Mapping[str, list[int]],
) -> list[int]:
    """경로, 파일명, stem/ID 순으로 고유한 원본 영상을 찾는다."""

    keys: list[str] = []
    if path_column and row.get(path_column):
        reference = _normalise_reference(row[path_column])
        keys.extend(
            (
                f"project:{reference}",
                f"raw:{reference}",
                f"name:{Path(reference).name.casefold()}",
                f"stem:{Path(reference).stem.casefold()}",
            )
        )
    if id_column and row.get(id_column):
        keys.append(f"stem:{row[id_column].strip().casefold()}")

    for key in keys:
        matches = indexes.get(key, [])
        if len(matches) == 1:
            return matches
    return []


def analyse_raw_video_matching(
    metadata: CsvData | None,
    inventory: list[dict[str, Any]],
    indexes: Mapping[str, list[int]],
) -> tuple[dict[str, Any], list[str]]:
    """마스터 metadata 참조와 수정하지 않은 원본 영상 목록을 비교한다."""

    if metadata is None:
        return {
            "raw_video_count": len(inventory),
            "metadata_video_count": 0,
            "error": "master metadata was not loaded",
        }, ["Raw matching could not run because master metadata was not loaded"]

    id_column = _selected_column(metadata, "video_id")
    path_column = _selected_column(metadata, "source_path")
    missing: list[str] = []
    matched_indexes: set[int] = set()
    metadata_keys: list[str] = []
    for row_number, row in enumerate(metadata.rows, start=2):
        identifier = (
            row.get(id_column, "")
            or row.get(path_column, "")
            or f"row:{row_number}"
        )
        metadata_keys.append(identifier)
        matches = _candidate_inventory_indexes(row, id_column, path_column, indexes)
        if matches:
            matched_indexes.add(matches[0])
        else:
            missing.append(identifier)

    for index in matched_indexes:
        inventory[index]["matched_metadata"] = True

    filename_counts = Counter(item["filename"].casefold() for item in inventory)
    duplicate_filenames = sorted(
        filename for filename, count in filename_counts.items() if count > 1
    )
    unmatched_raw = [
        item["relative_path"] for item in inventory if not item["matched_metadata"]
    ]
    duplicate_metadata_references = len(metadata_keys) - len(set(metadata_keys))
    result = {
        "master_metadata_file": metadata.name,
        "video_id_column": id_column,
        "source_path_column": path_column,
        "supported_extensions": sorted(VIDEO_EXTENSIONS),
        "discovered_extensions": sorted({item["extension"] for item in inventory}),
        "raw_video_count": len(inventory),
        "metadata_video_count": len(metadata.rows),
        "matched_metadata_video_count": len(metadata.rows) - len(missing),
        "metadata_missing_from_raw_count": len(missing),
        "metadata_missing_from_raw_examples": missing[:50],
        "raw_missing_from_metadata_count": len(unmatched_raw),
        "raw_missing_from_metadata_examples": unmatched_raw[:50],
        "duplicate_raw_filename_count": len(duplicate_filenames),
        "duplicate_raw_filename_examples": duplicate_filenames[:50],
        "duplicate_metadata_reference_count": duplicate_metadata_references,
    }
    warnings: list[str] = []
    if missing:
        warnings.append(f"{len(missing)} metadata videos were not found under raw_dir")
    if unmatched_raw:
        warnings.append(f"{len(unmatched_raw)} raw videos were not found in master metadata")
    if duplicate_filenames:
        warnings.append(f"{len(duplicate_filenames)} duplicate raw filenames were found")
    if duplicate_metadata_references:
        warnings.append(
            f"Master metadata contains {duplicate_metadata_references} duplicate references"
        )
    return result, warnings


def _resolve_row_video_path(
    row: Mapping[str, str],
    data: CsvData,
    config: ValidationConfig,
    inventory: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, list[int]],
) -> Path | None:
    """split의 한 행을 조사된 원본 영상 경로 하나에 대응시킨다."""

    matches = _candidate_inventory_indexes(
        row,
        _selected_column(data, "video_id"),
        _selected_column(data, "source_path"),
        indexes,
    )
    if not matches:
        return None
    stored_path = Path(str(inventory[matches[0]]["relative_path"]))
    return stored_path if stored_path.is_absolute() else config.project_root / stored_path


def select_decode_samples(
    datasets: Mapping[str, CsvData], samples_per_label: int
) -> dict[str, list[dict[str, str]]]:
    """각 split과 라벨에서 소수의 행을 결정론적으로 선택한다."""

    selected: dict[str, list[dict[str, str]]] = {}
    for split in SPLIT_NAMES:
        data = datasets.get(f"{split}.csv")
        if data is None:
            continue
        label_column = _selected_column(data, "label")
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in data.rows:
            label = row.get(label_column, "<UNKNOWN>") if label_column else "<UNKNOWN>"
            groups[label or "<EMPTY>"].append(row)
        selected[split] = [
            row
            for label in sorted(groups)
            for row in groups[label][:samples_per_label]
        ]
    return selected


def run_decode_smoke_test(
    datasets: Mapping[str, CsvData],
    config: ValidationConfig,
    inventory: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, list[int]],
) -> tuple[dict[str, Any], list[str]]:
    """결정론적으로 고른 소수 영상에서 한 프레임을 열어 decode한다."""

    if config.skip_decode:
        return {"status": "SKIPPED", "reason": "--skip-decode was used", "samples": []}, []
    try:
        import cv2  # type: ignore
    except (ImportError, OSError) as exc:
        message = f"OpenCV import failed: {exc}"
        return {"status": "ERROR", "error": message, "samples": []}, [message]

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    selections = select_decode_samples(datasets, config.samples_per_label)
    for split in SPLIT_NAMES:
        data = datasets.get(f"{split}.csv")
        if data is None:
            continue
        id_column = _selected_column(data, "video_id")
        label_column = _selected_column(data, "label")
        for row in selections.get(split, []):
            identifier = row.get(id_column, "") if id_column else ""
            label = row.get(label_column, "") if label_column else ""
            path = _resolve_row_video_path(row, data, config, inventory, indexes)
            sample: dict[str, Any] = {
                "split": split,
                "video_id": identifier,
                "label": label,
                "path": _display_path(path, config.project_root) if path else None,
            }
            if path is None:
                sample.update({"status": "ERROR", "error": "raw video was not resolved"})
                errors.append(f"{split}/{identifier}: raw video was not resolved")
                results.append(sample)
                continue

            capture = None
            try:
                capture = cv2.VideoCapture(str(path))
                opened = bool(capture.isOpened())
                fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
                frame_count = (
                    int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0
                )
                width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) if opened else 0
                height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) if opened else 0
                decoded, _frame = capture.read() if opened else (False, None)
                duration = (frame_count / fps) if fps > 0 else None
                status = "OK" if opened and decoded and fps > 0 else "ERROR"
                sample.update(
                    {
                        "status": status,
                        "opened": opened,
                        "first_frame_decoded": bool(decoded),
                        "fps": fps,
                        "frame_count": frame_count,
                        "width": width,
                        "height": height,
                        "duration_seconds": duration,
                    }
                )
                if status == "ERROR":
                    causes = []
                    if not opened:
                        causes.append("VideoCapture did not open")
                    if opened and not decoded:
                        causes.append("first frame could not be decoded")
                    if fps <= 0:
                        causes.append("FPS is zero or invalid")
                    sample["error"] = "; ".join(causes)
                    errors.append(f"{split}/{identifier}: {sample['error']}")
            except Exception as exc:
                # OpenCV 예외 클래스는 빌드마다 다를 수 있다. 실행을 중단하지 않고
                # 정확한 예외 클래스와 메시지를 보고서에 보존한다.
                message = f"{type(exc).__name__}: {exc}"
                sample.update({"status": "ERROR", "error": message})
                errors.append(f"{split}/{identifier}: {message}")
            finally:
                if capture is not None:
                    capture.release()
            results.append(sample)

    status = "OK" if results and not errors else "ERROR"
    if not results:
        errors.append("No videos were selected for decode smoke testing")
        status = "ERROR"
    return {"status": status, "sample_count": len(results), "samples": results}, errors


def collect_environment() -> dict[str, str | None]:
    """PyTorch를 import하거나 모델 작업을 시작하지 않고 버전을 수집한다."""

    def package_version(distributions: Iterable[str]) -> str | None:
        for distribution in distributions:
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    return {
        "python": platform.python_version(),
        "opencv": package_version(("opencv-python", "opencv-contrib-python", "opencv-python-headless")),
        "pytorch": package_version(("torch",)),
        "dlib": package_version(("dlib",)),
        "os": platform.platform(),
    }


def run_asset_load_test(config: ValidationConfig) -> tuple[dict[str, Any], list[str]]:
    """얼굴 검출을 실행하지 않고 Dlib68과 YuNet 모델을 불러온다."""

    if config.skip_asset_load:
        return {"status": "SKIPPED", "reason": "--skip-asset-load was used"}, []

    results: dict[str, Any] = {}
    errors: list[str] = []
    if not config.dlib_predictor.is_file():
        results["dlib68"] = {"status": "ERROR", "error": "predictor file is missing"}
        errors.append("Dlib68 predictor file is missing")
    else:
        try:
            import dlib  # type: ignore

            dlib.shape_predictor(str(config.dlib_predictor))
            results["dlib68"] = {"status": "OK"}
        except Exception as exc:
            # Dlib 로딩 실패의 예외 형식은 빌드마다 다를 수 있다.
            message = f"{type(exc).__name__}: {exc}"
            results["dlib68"] = {"status": "ERROR", "error": message}
            errors.append(f"Dlib68 load failed: {message}")

    if not config.yunet_model.is_file():
        results["yunet"] = {"status": "ERROR", "error": "ONNX file is missing"}
        errors.append("YuNet ONNX file is missing")
    else:
        try:
            import cv2  # type: ignore

            if not hasattr(cv2, "FaceDetectorYN"):
                raise RuntimeError("installed OpenCV does not provide FaceDetectorYN")
            cv2.FaceDetectorYN.create(
                str(config.yunet_model),
                "",
                (320, 320),
                score_threshold=0.9,
                nms_threshold=0.3,
                top_k=5000,
            )
            results["yunet"] = {"status": "OK"}
        except Exception as exc:
            # cv2.error가 항상 RuntimeError를 상속한다고 보장할 수 없다.
            message = f"{type(exc).__name__}: {exc}"
            results["yunet"] = {"status": "ERROR", "error": message}
            errors.append(f"YuNet load failed: {message}")

    results["status"] = "OK" if not errors else "ERROR"
    return results, errors


def _write_json(path: Path, payload: Any) -> None:
    """결정론적이며 읽기 쉬운 UTF-8 JSON을 기록한다."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_raw_inventory(path: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    """원본 영상 목록과 metadata 대응 상태를 기록한다."""

    fieldnames = [
        "relative_path",
        "raw_relative_path",
        "filename",
        "stem",
        "extension",
        "size_bytes",
        "matched_metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)


def _json_text(value: Any) -> str:
    """중첩된 보고서 데이터를 간결하고 결정론적으로 변환한다."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def render_report(summary: Mapping[str, Any]) -> str:
    """사람이 읽을 수 있는 전체 검증 보고서를 생성한다."""

    sections = [
        ("Environment", summary["environment"]),
        ("Required Paths", summary["required_paths"]),
        ("Metadata Summary", summary["metadata_summary"]),
        ("Split Integrity", summary["split_integrity"]),
        ("Raw Video Matching", summary["raw_video_matching"]),
        ("Label Distribution", summary["label_distribution"]),
        ("Video Decode Smoke Test", summary["video_decode_smoke_test"]),
        ("Asset Load Test", summary["asset_load_test"]),
        ("Warnings", summary["warnings"] or ["None"]),
        (
            "Final Decision",
            {
                "decision": summary["decision"],
                "reasons": summary["decision_reasons"],
            },
        ),
    ]
    header = [
        "Experiment 2 — Source Data Validation",
        f"Project root: {summary['inputs']['project_root']}",
        f"Raw directory: {summary['inputs']['raw_dir']}",
        f"Metadata directory: {summary['inputs']['metadata_dir']}",
        f"Output directory: {summary['inputs']['output_dir']}",
        "",
    ]
    body: list[str] = header
    for title, content in sections:
        body.extend((f"[{title}]", _json_text(content), ""))
    return "\n".join(body).rstrip() + "\n"


def _metadata_summary(datasets: Mapping[str, CsvData]) -> dict[str, Any]:
    """최상위 요약에 사용할 간결한 metadata 진단 결과를 반환한다."""

    return {
        filename: {
            "row_count": data.schema["row_count"],
            "columns": data.schema["columns"],
            "null_count_total": data.schema["null_count_total"],
            "duplicate_row_count": data.schema["duplicate_row_count"],
            "selected_video_id_column": data.schema["selected_video_id_column"],
            "selected_label_column": data.schema["selected_label_column"],
            "selected_source_path_column": data.schema[
                "selected_source_path_column"
            ],
        }
        for filename, data in datasets.items()
    }


def determine_decision(
    required_paths: Sequence[Mapping[str, Any]],
    hard_errors: Sequence[str],
    warnings: Sequence[str],
) -> tuple[str, list[str]]:
    """보수적인 원본 데이터 검증 판정과 그 이유를 결정한다."""

    missing = [item["item"] for item in required_paths if item["status"] == "MISSING"]
    reasons = [f"Missing required path: {item}" for item in missing] + list(hard_errors)
    if reasons:
        return "SOURCE_VALIDATION_FAILED", reasons
    if warnings:
        return "READY_WITH_WARNINGS", list(warnings)
    return "READY_FOR_PREPROCESSING_AUDIT", ["All STEP 1 checks passed"]


def run_source_validation(config: ValidationConfig) -> dict[str, Any]:
    """STEP 1 검증을 실행하고 필수 결과 파일을 모두 기록한다."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    required_paths = check_required_paths(config)
    datasets, csv_errors = load_metadata(config.metadata_dir)
    schema_payload = {
        filename: data.schema for filename, data in sorted(datasets.items())
    }

    split_integrity, split_errors = analyse_split_integrity(datasets)
    label_distribution, label_warnings = analyse_label_distribution(datasets)
    inventory, indexes = inventory_raw_videos(config.raw_dir, config.project_root)
    raw_matching, raw_warnings = analyse_raw_video_matching(
        datasets.get("sust_ddd_video_metadata.csv"), inventory, indexes
    )
    decode_results, decode_errors = run_decode_smoke_test(
        datasets, config, inventory, indexes
    )
    asset_results, asset_errors = run_asset_load_test(config)

    raw_errors: list[str] = []
    if raw_matching.get("metadata_missing_from_raw_count", 0):
        raw_errors.append(
            f"{raw_matching['metadata_missing_from_raw_count']} metadata videos "
            "were not found under raw_dir"
        )
        raw_warnings = [
            warning
            for warning in raw_warnings
            if "metadata videos were not found under raw_dir" not in warning
        ]
    hard_errors = csv_errors + split_errors + raw_errors + decode_errors + asset_errors
    warnings = label_warnings + raw_warnings
    if config.skip_decode:
        warnings.append("Video decode smoke test was skipped by request")
    if config.skip_asset_load:
        warnings.append("Asset load smoke test was skipped by request")
    for filename, data in sorted(datasets.items()):
        duplicate_count = data.schema["duplicate_row_count"]
        if duplicate_count:
            warnings.append(f"{filename}: {duplicate_count} duplicate rows")
    decision, decision_reasons = determine_decision(
        required_paths, hard_errors, warnings
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "project_root": str(config.project_root),
            "raw_dir": str(config.raw_dir),
            "metadata_dir": str(config.metadata_dir),
            "output_dir": str(config.output_dir),
            "dlib_predictor": str(config.dlib_predictor),
            "yunet_model": str(config.yunet_model),
            "samples_per_label": config.samples_per_label,
            "skip_decode": config.skip_decode,
            "skip_asset_load": config.skip_asset_load,
        },
        "environment": collect_environment(),
        "required_paths": required_paths,
        "metadata_summary": _metadata_summary(datasets),
        "split_integrity": split_integrity,
        "raw_video_matching": raw_matching,
        "label_distribution": label_distribution,
        "video_decode_smoke_test": decode_results,
        "asset_load_test": asset_results,
        "errors": hard_errors,
        "warnings": warnings,
        "decision": decision,
        "decision_reasons": decision_reasons,
        "generated_files": list(REPORT_FILENAMES),
    }

    _write_json(config.output_dir / "metadata_schema.json", schema_payload)
    _write_json(config.output_dir / "split_integrity.json", split_integrity)
    write_raw_inventory(config.output_dir / "raw_video_inventory.csv", inventory)
    _write_json(config.output_dir / "source_validation_summary.json", summary)
    report = render_report(summary)
    (config.output_dir / "source_validation_report.txt").write_text(
        report, encoding="utf-8", newline="\n"
    )
    return summary
