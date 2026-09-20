"""STEP 3-A canonical 전처리의 명시적 실행 및 metadata-only dry-run CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.canonical_preprocessing import dry_run, execute, load_config


def main() -> int:
    """dry-run에서는 영상과 모델을 열지 않고 실행 계획만 확인한다."""

    parser = argparse.ArgumentParser(description="STEP 3 canonical preprocessing")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "canonical_preprocessing.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    result = dry_run(config, args.mode, args.manifest) if args.dry_run else execute(config, args.mode, args.manifest, args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
