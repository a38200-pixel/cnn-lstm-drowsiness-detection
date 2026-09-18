from pathlib import Path

REQUIRED = [
    "data/metadata/sust_ddd_video_metadata.csv",
    "data/metadata/train.csv",
    "data/metadata/val.csv",
    "data/metadata/test.csv",
    "data/metadata/split_summary.csv",
    "assets/dlib/shape_predictor_68_face_landmarks.dat",
    "assets/yunet/face_detection_yunet_2023mar.onnx",
    "configs/preprocessing_v2.yaml",
    "configs/sequence_v2.yaml",
    "configs/model_resnet18_lstm.yaml",
    "configs/model_vgg16_lstm.yaml",
    "configs/behavior_rules.yaml",
    "configs/training_v2.yaml",
]

root = Path(__file__).resolve().parents[1]
missing = [p for p in REQUIRED if not (root / p).exists()]
if missing:
    print("Missing required files:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)
print("Required Experiment-2 structure/assets check: OK")
