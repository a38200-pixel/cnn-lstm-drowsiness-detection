"""STEP 2-B HOG/YuNet primary detector 비교의 명령행 진입점."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.detector_primary_comparison import (  # noqa: E402
    describe_comparison_dry_run,
    load_comparison_config,
    run_primary_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    """기본 config와 영상 처리를 하지 않는 dry-run을 제공한다."""

    parser = argparse.ArgumentParser(
        description="STEP 2-A의 동일 frame에서 HOG와 YuNet primary 후보를 비교합니다."
    )
    parser.add_argument(
        "--config", default="configs/detector_primary_comparison.yaml",
        help="프로젝트 루트 기준 STEP 2-B YAML config 경로",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="영상 없이 STEP 2-A 표본 재사용, split, 출력 경로만 검증",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """설정을 읽고 dry-run 또는 실제 160-frame 비교를 실행한다."""

    args = build_parser().parse_args(argv)
    try:
        config = load_comparison_config(Path(args.config), PROJECT_ROOT)
        if args.dry_run:
            print(json.dumps(describe_comparison_dry_run(config), ensure_ascii=False, indent=2))
            return 0
        summary = run_primary_comparison(config)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: STEP 2-B 비교 실행 실패: {exc}", file=sys.stderr)
        return 2
    print(f"Decision: {summary['decision']}")
    print(f"Frames: {summary['scope']['frame_count']}")
    print(f"Output: {config.path('output_dir')}")
    print("manual_review.csv와 comparison image를 직접 검토해야 합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

