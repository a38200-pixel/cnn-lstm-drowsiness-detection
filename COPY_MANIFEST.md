# 1차 프로젝트 -> 2차 프로젝트 복사/참조 Manifest

## A. 반드시 가져갈 것

### 원본 데이터
- `data/raw/SUST Driver Drowsiness Dataset/` -> `data/raw/SUST Driver Drowsiness Dataset/`
- `data/raw/README.md` -> `data/raw/README.md`

### 고정 video-level split/metadata
- `data/metadata/sust_ddd_video_metadata.csv`
- `data/metadata/train.csv`
- `data/metadata/val.csv`
- `data/metadata/test.csv`
- `data/metadata/split_summary.csv`

### 2차 전처리에 필요한 모델 자산
- `assets/shape_predictor_68_face_landmarks.dat` -> `assets/dlib/shape_predictor_68_face_landmarks.dat`
- `assets/models/yunet/face_detection_yunet_2023mar.onnx` -> `assets/yunet/face_detection_yunet_2023mar.onnx`

### 환경/패키지 기준 파일
- `requirements.txt`
- `pyproject.toml`

## B. 1차 실험 근거로 보존할 보고서

### 데이터/전처리
- `outputs/preprocessing_audit/final_audit_report.txt`
- `outputs/preprocessing_audit_v2/final_audit_report_v2.txt`
- `outputs/fallback_comparison/comparison_report.txt`
- `outputs/fallback_quality_validation/quality_report.txt`
- `outputs/feature_calibration/feature_calibration_report.txt`
- `outputs/feature_calibration/calibration_decision.json`
- `outputs/sequence_feasibility/sequence_feasibility_report.txt`
- `outputs/dataset_preparation/dataset_preparation_report.txt`

### 모델/학습
- `outputs/training/training_experiment_report.txt`
- `outputs/generalization_gap/generalization_gap_report.txt`
- `outputs/label_ambiguity_analysis/label_ambiguity_analysis_report.txt`
- `outputs/controlled_epoch_replay/controlled_replay_report.txt`
- `outputs/bn_buffer_swap/bn_swap_report.txt`
- `outputs/groupnorm_ablation/groupnorm_ablation_report.txt`
- `outputs/optimizer_ablation/optimizer_ablation_report.txt`
- `outputs/final_protocol_ablation/gn_optimizer_3seed_analysis/` 전체

복사 위치 권장:
- 전처리 보고서 -> `experiment_1_reference/reports/preprocessing/`
- 학습 보고서 -> `experiment_1_reference/reports/training/`

## C. 비교용 metrics/config

### 초기 Model A/B
각 seed(42/43/44)의 아래 파일만 우선 보존:
- `run_summary.json`
- `metrics_history.csv`
- `config.json`
- `environment.json` (존재하면)

### 최종 GN optimizer 3-seed
각 run의 아래 파일:
- `run_summary.json`
- `metrics_history.csv`
- `config.json`
- `environment.json` (존재하면)

복사 위치:
- 초기 A/B -> `experiment_1_reference/metrics/initial_model_ab/`
- 최종 GN optimizer -> `experiment_1_reference/metrics/final_gn_optimizer_3seed/`

## D. 선택적으로만 가져갈 것

### 기존 checkpoint
다음 목적이 있을 때만:
- 1차 모델 재평가
- baseline 재현
- 명시적인 transfer-learning 비교

복사 위치: `experiment_1_reference/checkpoints_optional/`

### V1/V2 NPZ cache
2차 기본 설계에서는 **복사하지 않는 것을 권장**합니다.
이유: 2차는 Full Face RGB, 10초 문맥, Head Pose를 새로 사용하므로 기존 grayscale ROI NPZ가 주 입력이 될 수 없습니다.
필요하면 원래 1차 프로젝트의 경로를 read-only reference로 유지하세요.

## E. 2차 학습에 재사용하면 안 되는 것
- `data/processed/sequences_t20_s10/train_sequences.csv`
- `data/processed/sequences_t20_s10/val_sequences.csv`
- `data/processed/sequences_t20_s10/test_sequences.csv`
- `data/processed/sequences_t20_s10/normalization_stats.json`
- 3DDFA 보정 결과를 production feature처럼 사용하는 파일
- 실패한 중간 checkpoint

이 파일들은 1차 재현용 reference일 뿐 2차 데이터 파이프라인 입력이 아닙니다.
