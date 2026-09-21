"""STEP 5-C Context·Behavior sequence의 full cross-branch integrity audit를 실행한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.sequences_v2.cross_branch_audit import (  # noqa: E402
    prepare_dry_run, run_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="bundle 집합과 예상 관계만 검증")
    args = parser.parse_args()
    canonical = PROJECT_ROOT / "data/interim/preprocessing_v2/canonical"
    freeze = PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4c_policy_freeze"
    context = PROJECT_ROOT / "data/interim/sequences_v2/context"
    context_reports = PROJECT_ROOT / "outputs/sequences_v2/context_sequence"
    behavior = PROJECT_ROOT / "data/interim/sequences_v2/behavior"
    behavior_reports = PROJECT_ROOT / "outputs/sequences_v2/behavior_sequence"
    output = PROJECT_ROOT / "outputs/sequences_v2/cross_branch_audit"
    if args.dry_run:
        result = prepare_dry_run(canonical, freeze, context, behavior,
                                 output.relative_to(PROJECT_ROOT))
    else:
        result = run_audit(
            PROJECT_ROOT, canonical, freeze, context, context_reports, behavior,
            behavior_reports, PROJECT_ROOT / "configs/sequence_missing_policy.yaml", output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
