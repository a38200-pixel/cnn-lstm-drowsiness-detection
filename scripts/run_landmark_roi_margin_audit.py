"""STEP 2-C YuNet landmark ROI margin audit 명령행 진입점이다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drowsiness_detection.preprocessing_v2.landmark_roi_margin_audit import (
    describe_roi_audit_dry_run,
    load_roi_audit_config,
    run_roi_margin_audit,
)


def parse_args() -> argparse.Namespace:
    """CLI 인수를 읽는다."""

    parser = argparse.ArgumentParser(description="STEP 2-C YuNet Landmark ROI Margin Audit")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "landmark_roi_margin_audit.yaml",
        help="STEP 2-C YAML 설정 경로",
    )
    parser.add_argument("--dry-run", action="store_true", help="model 실행과 파일 생성 없이 입력 범위만 확인")
    return parser.parse_args()


def main() -> int:
    """dry-run 또는 실제 audit를 실행한다."""

    args = parse_args()
    config = load_roi_audit_config(args.config, PROJECT_ROOT)
    result = describe_roi_audit_dry_run(config) if args.dry_run else run_roi_margin_audit(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
