"""STEP 3-B용 20개 영상 manifest를 metadata만으로 고른다."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.canonical_preprocessing import load_config, load_input_rows, pilot_rows, source_path
from drowsiness_detection.preprocessing_v2.canonical_writer import write_csv


def main() -> int:
    """기본 dry-run에서는 영구 파일을 만들지 않는다."""

    parser = argparse.ArgumentParser(description="STEP 3 deterministic pilot manifest")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "canonical_preprocessing.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "metadata" / "experiment2_step3_pilot_manifest.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    selected = pilot_rows(load_input_rows(config))
    prepared = [{"video_id": row["video_id"], "split": row["split"], "label": row["label"],
                 "source_path": str(source_path(row, config)), "pilot_order": row["pilot_order"],
                 "selection_stratum": row["selection_stratum"], "selection_method": row["selection_method"]}
                for row in selected]
    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(f"기존 manifest 덮어쓰기 금지: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output, ("video_id", "split", "label", "source_path", "pilot_order",
                               "selection_stratum", "selection_method"), prepared)
    print(json.dumps({"mode": "DRY_RUN" if args.dry_run else "MANIFEST_CREATED", "video_count": len(prepared),
                      "strata": dict(Counter(row["selection_stratum"] for row in prepared)),
                      "test_split_used": False, "video_opened": False,
                      "output": str(args.output), "video_ids": [row["video_id"] for row in prepared]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
