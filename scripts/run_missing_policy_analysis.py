"""STEP 4-C1 Context/Behavior 결측 정책 후보의 train/val 영향 시뮬레이션."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.missing_policy_analysis import analyze, dry_run, load_inputs, write_analysis


def main() -> int:
    """Dry-run과 실제 영향 분석 모두 raw video·Test를 열지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path,
                        default=PROJECT_ROOT / "data/interim/preprocessing_v2/canonical")
    parser.add_argument("--audit-root", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4c_policy_analysis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    states = load_inputs(args.canonical_root, args.audit_root)
    result = (dry_run(states, args.canonical_root, args.audit_root, args.output_dir) if args.dry_run
              else write_analysis(analyze(states), args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
