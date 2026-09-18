"""STEP 2 detector/landmark 호환성 audit의 명령행 진입점."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.detector_landmark_audit import (  # noqa: E402
    describe_dry_run,
    load_audit_config,
    run_audit,
)


def build_parser() -> argparse.ArgumentParser:
    """기본 config와 안전한 dry-run 옵션을 제공한다."""

    parser = argparse.ArgumentParser(
        description="train/validation 소규모 표본에서 HOG→YuNet→Dlib68 호환성을 audit합니다."
    )
    parser.add_argument(
        "--config",
        default="configs/detector_landmark_audit.yaml",
        help="프로젝트 루트 기준 YAML config 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="영상과 detector를 열지 않고 config, metadata 범위, sampling 계획만 확인",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """설정을 검증하고 dry-run 또는 실제 소규모 audit을 실행한다."""

    args = build_parser().parse_args(argv)
    try:
        config = load_audit_config(Path(args.config), PROJECT_ROOT)
        if args.dry_run:
            print(json.dumps(describe_dry_run(config), ensure_ascii=False, indent=2))
            return 0
        summary = run_audit(config)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: STEP 2 audit 실행 실패: {exc}", file=sys.stderr)
        return 2

    print(f"Decision: {summary['decision']}")
    print(f"Processed frames: {summary['dataset_scope']['candidate_frames_processed']}")
    print(f"Output: {config.path('output_dir')}")
    print("manual_review.csv와 contact sheet를 직접 검토해야 합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

