"""STEP 4-A canonical train/val 결측·품질 자동 audit CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.quality_missing_audit import analyze, dry_run, write_audit


def main() -> int:
    """Dry-run은 읽기만 하고 실제 모드만 별도 결과를 발행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path,
                        default=PROJECT_ROOT / "data/interim/preprocessing_v2/canonical")
    parser.add_argument("--step3c-summary", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/full/full_preprocessing_summary.json")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        result = dry_run(args.canonical_root, args.step3c_summary)
    else:
        if args.output_dir.exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {args.output_dir}")
        result = write_audit(analyze(args.canonical_root, args.step3c_summary), args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
