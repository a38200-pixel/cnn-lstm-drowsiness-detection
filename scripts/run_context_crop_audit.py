"""STEP 2-D Context CNN crop 정책 audit 명령행 진입점이다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.context_crop_audit import (
    describe_context_crop_dry_run,
    load_context_crop_config,
    run_context_crop_audit,
)


def main() -> int:
    """설정 경로와 dry-run 여부를 해석해 audit를 시작한다."""

    parser = argparse.ArgumentParser(description="STEP 2-D Context CNN Crop Policy Audit")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "context_crop_audit.yaml")
    parser.add_argument("--dry-run", action="store_true", help="영상 decode와 파일 생성 없이 입력만 확인")
    args = parser.parse_args()
    config = load_context_crop_config(args.config, PROJECT_ROOT)
    result = describe_context_crop_dry_run(config) if args.dry_run else run_context_crop_audit(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
