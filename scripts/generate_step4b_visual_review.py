"""STEP 4-B 36개 후보의 수동 시각 검토 팩을 생성한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.step4b_visual_review import dry_run, generate_pack, load_plan


def main() -> int:
    """Dry-run에서는 raw video/JPEG를 열거나 결과를 쓰지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path,
                        default=PROJECT_ROOT / "data/interim/preprocessing_v2/canonical")
    parser.add_argument("--audit-root", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.canonical_root, args.audit_root)
    result = dry_run(plan, args.output_dir) if args.dry_run else generate_pack(plan, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
