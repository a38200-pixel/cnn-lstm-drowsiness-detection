"""STEP 4-C2 Context C3·Behavior B2 정책을 별도 semantic hash로 동결한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.missing_policy_freeze import dry_run, load_config, prepare_freeze, write_freeze


def main() -> int:
    """Dry-run은 읽기만 하고 실제 모드에서만 새 freeze 결과를 발행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/sequence_missing_policy.yaml")
    parser.add_argument("--canonical-root", type=Path,
                        default=PROJECT_ROOT / "data/interim/preprocessing_v2/canonical")
    parser.add_argument("--audit-root", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4c_policy_freeze")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepared = prepare_freeze(load_config(args.config), args.canonical_root, args.audit_root)
    result = dry_run(prepared, args.config, args.output_dir) if args.dry_run else write_freeze(prepared, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
