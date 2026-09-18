"""STEP 2-B 결과에서 수동 시각 검토 팩을 생성한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drowsiness_detection.preprocessing_v2.review_pack import build_review_pack


def parse_args() -> argparse.Namespace:
    """명령행 인수를 읽는다."""

    parser = argparse.ArgumentParser(description="STEP 2-B 수동 시각 검토 팩 생성")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "detector_primary_comparison.yaml",
        help="STEP 2-B 설정 파일",
    )
    parser.add_argument("--top-count", type=int, default=8, help="지표별 상위 후보 수")
    parser.add_argument("--max-samples", type=int, default=45, help="최대 최종 표본 수")
    parser.add_argument("--normal-samples", type=int, default=10, help="균형 정상 참조 표본 수")
    parser.add_argument("--dry-run", action="store_true", help="파일을 만들지 않고 선정 통계만 출력")
    return parser.parse_args()


def main() -> int:
    """검토 팩을 만들고 생성 통계를 JSON으로 출력한다."""

    args = parse_args()
    result = build_review_pack(
        PROJECT_ROOT,
        args.config,
        top_count=args.top_count,
        maximum_total=args.max_samples,
        normal_target=args.normal_samples,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
