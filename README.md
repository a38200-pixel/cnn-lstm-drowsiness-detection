# SUST-DDD Experiment 2 Scaffold

이 폴더는 1차 실험 코드를 덮어쓰지 않고, 2차 실험을 독립적으로 수행하기 위한 기본 구조입니다.

## 핵심 원칙
- 원본 SUST-DDD 영상과 기존 video-level train/val/test split만 기본적으로 승계합니다.
- 1차 실험의 T20-S10 sequence CSV, normalization stats, 모델 checkpoint는 2차 학습 입력으로 재사용하지 않습니다.
- 2차 데이터/통계/checkpoint는 처음부터 새로 생성합니다.
- test split은 최종 모델 선택 전까지 격리합니다.
- ResNet18과 VGG16은 동시에 사용하는 backbone이 아니라 별도 비교 실험입니다.
- Context branch와 Behavior Rule branch는 서로 다른 시간 해상도를 사용합니다.

## 현재 2차 실험 기준 설계
- Context: 원본 10초 clip -> 32개 균등 sampling -> Full Face RGB 224x224x3 -> pretrained ResNet18 또는 VGG16 -> classifier 제거 -> GAP -> 512-D/frame -> LSTM(hidden=128) -> clip-level drowsy probability.
- Behavior: 같은 10초 clip -> 10 FPS(약 100 points) -> 동일 landmark source(Dlib68) -> EAR/MAR/Head Pose -> rule event.
- Fusion: Context probability + eye/yawn/head events를 후단 Decision Fusion에서 결합.

자세한 파일 이전 정책은 COPY_MANIFEST.md, 실험 설계 근거는 docs/DESIGN_NOTES.md를 참고하세요.

## Experiment 2 Progress

### STEP 1 — Source Data Validation

Status: **IMPLEMENTED — WAITING FOR MANUAL RUN**

- 목적: 2차 실험을 시작하기 전에 원본 SUST-DDD 영상, metadata, 기존 frozen split, Dlib68/YuNet asset이 정상적으로 준비되어 있는지 검증합니다.
- 원본 데이터: `data/raw/SUST Driver Drowsiness Dataset/`을 읽기 전용 입력으로 사용하며 원본 영상이나 파일명을 수정하지 않습니다.
- frozen split: `data/metadata/train.csv`, `val.csv`, `test.csv`를 그대로 사용합니다. 이 단계에서는 split을 재생성하지 않으며 test split은 통계 및 무결성 검사에만 사용합니다.
- 검증 항목: 필수 경로, 실제 CSV schema/null/중복 row, split 간 video ID 중복, metadata와 raw 영상 대응, split별 label 분포, 소수 영상의 OpenCV decode, Dlib68 및 YuNet model load를 확인합니다.
- 결과 파일: `outputs/preprocessing_v2/source_validation/` 아래에 `source_validation_report.txt`, `source_validation_summary.json`, `metadata_schema.json`, `split_integrity.json`, `raw_video_inventory.csv`가 생성됩니다.
- 기본 실행 명령: `python scripts/validate_experiment2_source.py`
- 경로 지정 예시: `python scripts/validate_experiment2_source.py --raw-dir "data/raw/SUST Driver Drowsiness Dataset" --metadata-dir "data/metadata" --output-dir "outputs/preprocessing_v2/source_validation"`
- 결과 요약: 검증 코드만 구현했으며 전체 raw dataset 검증은 아직 실행하지 않았습니다. 따라서 실제 video 수, overlap 수, label 분포 등의 숫자는 기록하지 않습니다.
- 발견된 문제: 현재 실행 결과가 없어 데이터 또는 asset 문제는 아직 판정할 수 없습니다.
- 최종 판정: 수동 실행 대기 중입니다. 실행 후 summary의 `decision`은 `READY_FOR_PREPROCESSING_AUDIT`, `READY_WITH_WARNINGS`, `SOURCE_VALIDATION_FAILED` 중 하나입니다.
- 다음 단계: STEP 1 결과를 검토한 뒤 STEP 2 preprocessing audit에서 HOG face detector → 실패 시 YuNet fallback → 모든 성공 face에 동일한 Dlib68 landmarks 적용 조합의 호환성과 품질을 소규모 샘플로 검증합니다. STEP 2 코드는 아직 구현하지 않았습니다.
