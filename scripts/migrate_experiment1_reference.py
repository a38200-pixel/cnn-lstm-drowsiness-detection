"""Copy selected Experiment-1 reference artifacts into this scaffold.

Usage:
    python scripts/migrate_experiment1_reference.py --src "C:/.../졸음탐지" --dst "."

This script intentionally does NOT copy raw videos, V1/V2 NPZ caches, or checkpoints by default.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

PREPROCESS_REPORTS = [
    "outputs/preprocessing_audit/final_audit_report.txt",
    "outputs/preprocessing_audit_v2/final_audit_report_v2.txt",
    "outputs/fallback_comparison/comparison_report.txt",
    "outputs/fallback_quality_validation/quality_report.txt",
    "outputs/feature_calibration/feature_calibration_report.txt",
    "outputs/feature_calibration/calibration_decision.json",
    "outputs/sequence_feasibility/sequence_feasibility_report.txt",
    "outputs/dataset_preparation/dataset_preparation_report.txt",
]
TRAINING_REPORTS = [
    "outputs/training/training_experiment_report.txt",
    "outputs/generalization_gap/generalization_gap_report.txt",
    "outputs/label_ambiguity_analysis/label_ambiguity_analysis_report.txt",
    "outputs/controlled_epoch_replay/controlled_replay_report.txt",
    "outputs/bn_buffer_swap/bn_swap_report.txt",
    "outputs/groupnorm_ablation/groupnorm_ablation_report.txt",
    "outputs/optimizer_ablation/optimizer_ablation_report.txt",
]
METADATA = [
    "data/metadata/sust_ddd_video_metadata.csv",
    "data/metadata/train.csv",
    "data/metadata/val.csv",
    "data/metadata/test.csv",
    "data/metadata/split_summary.csv",
]
ASSETS = [
    ("assets/shape_predictor_68_face_landmarks.dat", "assets/dlib/shape_predictor_68_face_landmarks.dat"),
    ("assets/models/yunet/face_detection_yunet_2023mar.onnx", "assets/yunet/face_detection_yunet_2023mar.onnx"),
]


def copy_file(src_root: Path, dst_root: Path, rel_src: str, rel_dst: str | None = None) -> bool:
    s = src_root / rel_src
    d = dst_root / (rel_dst or rel_src)
    if not s.exists():
        print(f"[MISSING] {s}")
        return False
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)
    print(f"[COPIED] {s} -> {d}")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Experiment-1 project root")
    p.add_argument("--dst", default=".", help="Experiment-2 scaffold root")
    args = p.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    for rel in METADATA:
        copy_file(src, dst, rel)
    for s, d in ASSETS:
        copy_file(src, dst, s, d)
    for rel in PREPROCESS_REPORTS:
        copy_file(src, dst, rel, f"experiment_1_reference/reports/preprocessing/{Path(rel).name}")
    for rel in TRAINING_REPORTS:
        copy_file(src, dst, rel, f"experiment_1_reference/reports/training/{Path(rel).name}")

    gn_dir = src / "outputs/final_protocol_ablation/gn_optimizer_3seed_analysis"
    if gn_dir.exists():
        target = dst / "experiment_1_reference/metrics/final_gn_optimizer_3seed/gn_optimizer_3seed_analysis"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(gn_dir, target)
        print(f"[COPIED TREE] {gn_dir} -> {target}")
    else:
        print(f"[MISSING] {gn_dir}")

    print("\nRaw SUST-DDD video folder is intentionally not copied automatically. Copy or link it manually after verifying disk usage.")

if __name__ == "__main__":
    main()
