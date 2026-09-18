"""2차 실험 원본 데이터 검증을 위한 명령행 진입점."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = DEFAULT_PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.validate_source_data import (  # noqa: E402
    REPORT_FILENAMES,
    build_config,
    run_source_validation,
)


def build_parser() -> argparse.ArgumentParser:
    """프로젝트 상대 경로를 기본값으로 사용하는 CLI parser를 생성한다."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate SUST-DDD raw videos, frozen split metadata, and detector "
            "assets without preprocessing the full dataset."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
        help="Project root used to resolve relative paths (default: repository root).",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/SUST Driver Drowsiness Dataset",
        help="Raw SUST-DDD directory, absolute or relative to --project-root.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/metadata",
        help="Metadata directory, absolute or relative to --project-root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/preprocessing_v2/source_validation",
        help="Report output directory, absolute or relative to --project-root.",
    )
    parser.add_argument(
        "--dlib-predictor",
        default="assets/dlib/shape_predictor_68_face_landmarks.dat",
        help="Dlib68 predictor path, absolute or relative to --project-root.",
    )
    parser.add_argument(
        "--yunet-model",
        default="assets/yunet/face_detection_yunet_2023mar.onnx",
        help="YuNet ONNX path, absolute or relative to --project-root.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=1,
        help="Decode-smoke-test sample count per label in each split (default: 1).",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip VideoCapture smoke tests; recorded explicitly in the report.",
    )
    parser.add_argument(
        "--skip-asset-load",
        action="store_true",
        help="Skip Dlib/YuNet load tests; recorded explicitly in the report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """인자를 해석하고 검증을 실행한 뒤 간결한 결과를 출력한다."""

    args = build_parser().parse_args(argv)
    try:
        config = build_config(
            project_root=args.project_root,
            raw_dir=args.raw_dir,
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            dlib_predictor=args.dlib_predictor,
            yunet_model=args.yunet_model,
            samples_per_label=args.samples_per_label,
            skip_decode=args.skip_decode,
            skip_asset_load=args.skip_asset_load,
        )
        summary = run_source_validation(config)
    except (OSError, ValueError) as exc:
        print(f"ERROR: source validation could not complete: {exc}", file=sys.stderr)
        return 2

    print("Required paths:")
    for item in summary["required_paths"]:
        print(f"  {item['status']:7} {item['item']}: {item['path']}")
    overlap = summary["split_integrity"].get("pairwise_overlap", {})
    print("Frozen split overlap:")
    for label, key in (
        ("train vs val", "train_vs_val"),
        ("train vs test", "train_vs_test"),
        ("val vs test", "val_vs_test"),
    ):
        count = overlap.get(key, {}).get("count", "NOT_CHECKED")
        print(f"  {label}: {count}")
    print(f"Decision: {summary['decision']}")
    print(f"Reports: {config.output_dir}")
    for filename in REPORT_FILENAMES:
        print(f"  - {filename}")
    return 1 if summary["decision"] == "SOURCE_VALIDATION_FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
