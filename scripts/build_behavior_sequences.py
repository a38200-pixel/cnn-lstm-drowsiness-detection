"""동결 B2 정책으로 train/val Behavior 100-slot sequence를 구성한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.sequences_v2.behavior_sequence_builder import (  # noqa: E402
    EXPECTED, FEATURE_NAMES, FEATURE_SCHEMA_HASH, POLICY_HASH, SEQUENCE_POLICY_HASH,
    load_plan, materialize, plan_statistics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="입력과 계획만 검증하고 출력하지 않음")
    parser.add_argument("--resume", action="store_true", help="일치하는 COMPLETE bundle을 검증 후 건너뜀")
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--dry-run과 --resume을 함께 사용할 수 없습니다")
    canonical = PROJECT_ROOT / "data/interim/preprocessing_v2/canonical"
    freeze = PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4c_policy_freeze"
    output = PROJECT_ROOT / "data/interim/sequences_v2/behavior"
    reports = PROJECT_ROOT / "outputs/sequences_v2/behavior_sequence"
    plans, excluded = load_plan(canonical, freeze, PROJECT_ROOT / "configs/sequence_missing_policy.yaml")
    counts = plan_statistics(plans, excluded, EXPECTED)
    if args.dry_run:
        result = {"status": "STEP_5B_DRY_RUN_PASS", **counts, "sequence_length": 100,
                  "feature_names": list(FEATURE_NAMES),
                  "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
                  "canonical_policy_hash": POLICY_HASH,
                  "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
                  "planned_sequence_root": output.relative_to(PROJECT_ROOT).as_posix(),
                  "planned_report_root": reports.relative_to(PROJECT_ROOT).as_posix(),
                  "test_rows": 0, "raw_video_opened": False, "interpolation_count": 0,
                  "context_sequence_accessed": False, "cnn_executed": False}
    else:
        result = materialize(plans, excluded, output, reports, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
