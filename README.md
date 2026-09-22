# CNN-LSTM Driver Drowsiness Detection

SUST-DDD의 약 10초 길이 운전자 영상을 이용해 졸음 상태를 탐지하는 Experiment 2 저장소다. 현재 데이터 검증부터 frozen CNN feature 생성·감사, Context Dataset/LSTM/training pipeline 구현과 두 backbone의 seed42 baseline 및 batch/dropout/weight decay/learning rate/Input LayerNorm/Label Smoothing tuning까지 완료됐다.

> **Current milestone:** STEP 6-A~6-C **COMPLETE** · seed42 Context baseline 및 hyperparameter tuning **COMPLETE** · TEST **SEALED**
> 아래 성능은 train/validation 결과이며 최종 test 성능이 아니다. Behavior rule threshold 확정과 test 평가는 아직 수행하지 않았다.

## 1. Project Overview

Experiment 2는 Experiment 1의 학습 산출물을 재사용하지 않고, 기존 video-level split만 유지한 채 데이터와 모델 산출물을 처음부터 다시 구축한다. 모델 입력은 서로 다른 시간 해상도를 갖는 두 branch로 분리한다.

- **Context branch:** 10초 clip에서 결정적으로 선택한 32개 RGB 얼굴 crop을 frozen pretrained CNN에 통과시켜 frame당 512차원 feature를 만든다. 현재 `[32,512]` frozen feature를 읽는 Dataset, LSTM baseline과 training pipeline까지 구현됐다.
- **Behavior branch:** 같은 clip을 10 Hz의 100개 canonical slot으로 표현하고 EAR, MAR와 head-pose 관련 연속값을 보존한다. 현재 `[100,6]` feature와 mask까지 구성됐다.
- 두 branch는 현재 독립적이다. ResNet18과 VGG16도 별도 실험이며 feature를 합치거나 ensemble하지 않는다.

## 2. Experiment 2 Architecture

```text
SUST-DDD video (~10 sec)
│
├── Context branch
│   ├── 32 deterministic Context slots
│   ├── RGB 224×224, SQUARE_M10 crop
│   ├── ResNet18 OR VGG16 (frozen ImageNet weights)
│   ├── 512-D per frame
│   └── stored [32,512] → LSTM classifier
│
└── Behavior branch
    ├── 100 slots @ 10 Hz
    ├── EAR / MAR
    ├── pitch_raw / pitch_centered_candidate
    └── yaw / roll
```

현재 Context baseline은 다음 구조로 구현됐으며 ResNet18/VGG16 seed42 validation run을 동일 정책으로 완료했다.

```text
Input              [B,32,512]
LSTM               input_size=512, hidden_size=128
                   num_layers=1, unidirectional, batch_first=true
                   internal dropout=0.0
Classifier         Linear(128→64) → ReLU → Linear(64→2)
Classifier dropout 0.0
Loss               CrossEntropyLoss(raw logits)
```

실제 구조와 학습 정책은 `configs/context_lstm_baseline.yaml`에서 관리한다. 자세한 실험 기준은 [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md)을 따른다.

## 3. Processing Pipeline

```text
source validation
→ detector / landmark / crop policy audit
→ 100-slot canonical preprocessing
→ train/val quality and missing-policy audit
→ Context 32-slot + Behavior 100-slot sequences
→ cross-branch integrity audit
→ frozen ResNet18 / VGG16 feature extraction
→ full feature final integrity audit
→ Context feature Dataset / LSTM baseline / training pipeline
→ seed42 train/validation baseline comparison
```

| 항목 | 동결 정책 |
|---|---|
| Face detector | YuNet primary, fallback 없음 |
| Face selection | 가장 큰 유효 bbox |
| Landmark | Dlib68, YuNet RAW bbox fitting ROI |
| Context crop | SQUARE_M10, ImageNet-mean padding, RGB 224×224 |
| Canonical timeline | 10 Hz × 10초 = 100 slots |
| Context slots | canonical timeline에서 결정적으로 선택한 32 slots |
| Context missing | `CONTEXT_C3_SHORT_GAP`, 같은 영상의 가장 가까운 원본-valid slot 재사용, 동률이면 earlier |
| Behavior missing | `BEHAVIOR_B2_COVERAGE_95`, NaN과 mask 보존, interpolation 없음 |

세부 근거는 [STEP 2](docs/experiment2_step2_preprocessing_policy.md), [STEP 3](docs/experiment2_step3_canonical_preprocessing.md), [STEP 4](docs/experiment2_step4_quality_missing_audit.md), [STEP 5](docs/experiment2_step5_sequence_construction.md) 문서에 있다.

## 4. Dataset / Split

STEP 1에서 raw 영상과 metadata 2,074개를 모두 대조했고 누락 및 split 중복은 0건이었다.

| Split | Drowsy | Not drowsy | Total |
|---|---:|---:|---:|
| Train | 683 | 769 | 1,452 |
| Validation | 146 | 165 | 311 |
| Test | 146 | 165 | 311 |
| **Total** | **975** | **1,099** | **2,074** |

- Split 단위는 video이며 train/validation/test는 중복되지 않는다.
- Subject ID mapping이 없어 unseen-driver 독립성을 보장하는 subject-wise split은 아니다.
- Test 311개는 최종 설정이 동결될 때까지 접근하지 않는다.
- STEP 3 이후의 materialization과 audit은 train/validation 1,763개만 대상으로 했다.

## 5. STEP Progress

| 단계 | 결과 | 핵심 산출물/결정 |
|---|---|---|
| STEP 1 | **COMPLETE** | 2,074개 source·metadata·split 검증 |
| STEP 2 | **FULLY CLOSED** | YuNet, Dlib68 RAW ROI, SQUARE_M10 확정 |
| STEP 3 | **COMPLETE** | train/val 1,763개, canonical 176,300 rows, Context crop 54,672개 |
| STEP 4 | **FULLY CLOSED** | Context C3 / Behavior B2 missing policy 동결 |
| STEP 5-A | **COMPLETE** | Context 1,677개 × 32 slots |
| STEP 5-B | **COMPLETE** | Behavior 1,520개 × 100 slots × 6 features |
| STEP 5-C | **COMPLETE** | BOTH 1,520, Context-only 157, Behavior-only 0, neither 86 |
| STEP 5-D1 | **COMPLETE** | 16-video ResNet18/VGG16 controlled pilot |
| STEP 5-D2 | **COMPLETE** | 전체 Context CNN feature extraction |
| STEP 5-D2R | **COMPLETE** | `n_311` numerical difference 원인 확인 |
| STEP 5-D3 | **PASS** | full artifact anomaly 0, test access 0 |
| **STEP 5** | **FULLY CLOSED** | sequence 및 frozen CNN feature 준비 완료 |
| STEP 6-A | **COMPLETE** | frozen Context feature Dataset/DataLoader |
| STEP 6-B | **COMPLETE** | 337,090-parameter Context LSTM baseline |
| STEP 6-C | **COMPLETE** | training pipeline, MLflow, seed42 baseline validation |

### STEP 1

Raw 영상 2,074개와 metadata를 전수 대조하고 기존 video-level split을 보존했다. Decode smoke test와 YuNet·Dlib68 asset load도 통과했다.

### STEP 2

Train/validation 표본으로 detector, landmark fitting ROI와 Context crop geometry를 분리해 감사했다. 최종 정책은 YuNet primary, YuNet RAW bbox에서 Dlib68 fitting, SQUARE_M10 Context crop이다. 이는 crop geometry 선택이며 모델 정확도 우위 판정이 아니다.

### STEP 3

1,763개 train/validation 영상을 100-slot canonical timeline으로 materialize했다. 영상마다 고정된 32개 Context target slot을 가지며 실패 slot은 대체하지 않고 상태·NaN·mask로 기록했다.

### STEP 4

전체 train/validation 품질과 결측 분포를 감사했다. Context C3는 결측 수 ≤8 및 최장 연속 결측 ≤4인 1,677개 영상을, Behavior B2는 valid ≥95 및 최장 YuNet 결측 ≤5인 1,520개 영상을 허용한다.

### STEP 5-A

Context 적격 1,677개를 영상당 32개 target row로 구성했다. 전체 53,664개 target 중 원본 관측은 52,965개, 같은 영상의 원본-valid source를 참조한 imputed target은 699개다. JPEG를 새로 만들지 않고 provenance와 mask를 보존했다.

### STEP 5-B

Behavior 적격 1,520개를 `[100,6]` float32 sequence로 구성했다. Feature 순서는 EAR, MAR, `pitch_raw`, `pitch_centered_candidate`, yaw, roll이며 결측은 NaN과 bool mask로 유지했다. 보간, smoothing, threshold 및 event 생성은 수행하지 않았다.

### STEP 5-C

Canonical, eligibility, Context 및 Behavior artifact를 교차 감사했다. 집합은 BOTH 1,520개, Context-only 157개, Behavior-only 0개, neither 86개이며 split·label·timeline·mask·numeric lineage mismatch는 0건이었다.

### STEP 5-D

ResNet18과 VGG16을 frozen ImageNet backbone으로 별도 실행했다. D1 pilot, D2 full extraction, D2R regression 조사와 D3 최종 무결성 감사까지 완료했다. D3에서 backbone별 52,965개 source feature와 1,677개 `[32,512]` bundle을 전수 검사했으며 numeric, mapping, imputation, policy/hash 및 cross-backbone semantic mapping anomaly는 모두 0건이었다.

### STEP 6-A

ResNet18과 VGG16 각각 train 1,382개, validation 295개의 frozen `[32,512]` float32 feature를 공급하는 Dataset/DataLoader를 구현했다. 추가 normalization 없이 저장값을 그대로 사용하며 Context-only 157개를 포함한다. Test 접근은 없다.

### STEP 6-B

입력 `[B,32,512]`를 받는 1-layer unidirectional LSTM을 구현했다. 구조는 `512→128`, classifier는 `128→64→2`와 ReLU이며 LSTM/classifier dropout은 모두 0.0이다. 출력은 softmax가 없는 raw logits이고 전체 parameter는 337,090개다. Imputed mask는 진단용으로만 보존하며 forward 입력에 결합하지 않는다.

### STEP 6-C

AdamW(`lr=0.0005`, `weight_decay=0.0001`), CrossEntropyLoss, train/validation batch 16/32, gradient clipping 1.0, ReduceLROnPlateau, early stopping과 minimum validation loss checkpoint를 사용하는 training pipeline을 구현했다. AMP와 feature normalization은 사용하지 않는다.

동일한 seed42와 동일한 학습 정책으로 얻은 **validation baseline** 결과는 다음과 같다. 이는 test 결과가 아니다.

| Backbone | Best Epoch | Val Loss | Accuracy | Macro F1 | Drowsy Recall | Early Stop |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 16 | 0.490563 | 0.7797 | 0.7784 | 0.7324 | Epoch 26 |
| VGG16 | 7 | 0.463662 | 0.8068 | 0.8061 | 0.7746 | Epoch 17 |

Seed42에서는 VGG16의 validation metric이 더 높게 관찰됐지만 단일 seed 결과이므로 backbone 우위를 확정하지 않는다. 두 backbone 모두 train loss는 계속 감소하는 반면 validation loss는 비교적 이른 시점부터 정체하거나 변동하는 경향을 보였다.

각 실험에서 하나의 변수만 바꾸어 비교한 seed42 **validation tuning** 결과는 다음과 같다.

| Backbone | Variant | Batch | Dropout | WD | LR | Train LS | Val Loss | Accuracy | Macro F1 | Drowsy Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | Baseline | 16 | 0.0 | 0.0001 | 0.0005 | 0.0 | 0.4906 | 0.7797 | 0.7784 | 0.7324 |
| VGG16 | Baseline | 16 | 0.0 | 0.0001 | 0.0005 | 0.0 | 0.4637 | 0.8068 | 0.8061 | 0.7746 |
| ResNet18 | Batch 8 | 8 | 0.0 | 0.0001 | 0.0005 | 0.0 | 0.5166 | 0.7458 | 0.7423 | 0.6549 |
| VGG16 | Batch 8 | 8 | 0.0 | 0.0001 | 0.0005 | 0.0 | 0.4895 | 0.7797 | 0.7782 | 0.7254 |
| ResNet18 | Dropout 0.2 | 16 | 0.2 | 0.0001 | 0.0005 | 0.0 | 0.4965 | 0.7864 | 0.7864 | 0.8239 |
| VGG16 | Dropout 0.2 | 16 | 0.2 | 0.0001 | 0.0005 | 0.0 | 0.5000 | 0.7627 | 0.7621 | 0.8451 |
| ResNet18 | WD 0.0005 | 16 | 0.0 | 0.0005 | 0.0005 | 0.0 | 0.4895 | 0.7898 | 0.7895 | 0.7817 |
| VGG16 | WD 0.0005 | 16 | 0.0 | 0.0005 | 0.0005 | 0.0 | 0.4983 | 0.7797 | 0.7794 | 0.7746 |
| ResNet18 | LR 0.00025 | 16 | 0.0 | 0.0001 | 0.00025 | 0.0 | 0.4523 | 0.8068 | 0.8063 | 0.7887 |
| VGG16 | LR 0.00025 | 16 | 0.0 | 0.0001 | 0.00025 | 0.0 | 0.4769 | 0.8000 | 0.7998 | 0.7958 |
| ResNet18 | Input LayerNorm | 16 | 0.0 | 0.0001 | 0.0005 | 0.0 | 0.4973 | 0.7661 | 0.7637 | 0.6901 |
| VGG16 | Input LayerNorm | 16 | 0.0 | 0.0001 | 0.0005 | 0.0 | 0.5000 | 0.7695 | 0.7683 | 0.8732 |
| ResNet18 | Label Smoothing | 16 | 0.0 | 0.0001 | 0.0005 | 0.05 | 0.4766 | 0.7763 | 0.7761 | 0.7746 |
| VGG16 | Label Smoothing | 16 | 0.0 | 0.0001 | 0.0005 | 0.05 | 0.4666 | 0.7797 | 0.7790 | 0.7535 |

- **Batch 8:** 두 backbone 모두 악화되어 제외한다.
- **Classifier dropout 0.2:** drowsy recall은 증가했지만 전체 validation 품질의 공통 개선에 실패하여 제외한다.
- **Weight decay 0.0005:** ResNet18은 소폭 개선됐지만 VGG16의 validation loss, accuracy와 Macro F1이 악화되어 공통 정책으로 채택하지 않는다.
- **Learning rate 0.00025:** ResNet18에서는 현재까지 가장 크게 개선됐지만 VGG16의 validation loss, accuracy와 Macro F1은 baseline보다 소폭 악화됐다. VGG16의 drowsy recall은 증가했으나 공통 LR은 변경하지 않는다.
- **Input LayerNorm(512):** 두 backbone 모두 validation loss, accuracy와 Macro F1이 악화되어 채택하지 않는다. ResNet18의 drowsy recall도 감소했다. VGG16은 drowsy recall이 0.7746에서 0.8732로 증가했지만 confusion matrix가 `TN 128 / FP 25 / FN 32 / TP 110`에서 `TN 103 / FP 50 / FN 18 / TP 124`로 변해 false positive가 두 배가 됐으므로 전체 validation 품질 개선으로 보지 않는다.
- **Label Smoothing 0.05:** Training CrossEntropyLoss에만 적용하고 validation loss는 기존과 동일한 smoothing 0.0으로 계산했다. ResNet18의 validation loss는 개선됐지만 accuracy와 Macro F1은 개선되지 않았고, VGG16은 validation loss와 분류 metric이 모두 악화됐다. Overconfidence 완화 가능성은 ResNet18에서 부분적으로만 관찰됐으므로 공통 정책으로 채택하지 않는다.

Input LayerNorm 후보는 mini-batch 통계에 의존하지 않고 train/test에서 같은 계산을 사용하며 recurrent model에 적용하기 쉽다는 선행연구를 근거로 선정했다. Frozen ResNet18/VGG16 512D feature의 scale/distribution 차이를 LSTM 입력 직전 `LayerNorm(512)`으로 완화하면 validation generalization이 개선될 수 있다는 가설이었다. 다만 이는 원 논문의 직접 재현이 아니며, 실제로는 두 backbone 모두 validation loss를 개선하지 못했다. LayerNorm 적용 후에도 train fitting은 진행됐지만 validation loss가 조기에 정체한 뒤 상승하는 경향이 관찰됐다. LayerNorm은 그 자체가 regularization을 목적으로 하는 기법이 아니므로 train loss 감소를 이상 현상으로 해석하지 않는다.

현재 공통 baseline은 **train/validation batch 16/32 / learning rate 0.0005 / weight decay 0.0001 / classifier dropout 0.0 / LSTM dropout 0.0 / Input LayerNorm 미적용 / label smoothing 0.0**으로 유지한다. 이번 LayerNorm 방식은 feature scale/distribution 차이가 validation 문제의 주요 원인이라는 가설을 지지하지 못했지만, feature distribution 문제 자체를 완전히 배제하지는 않는다. Label Smoothing도 backbone 공통 개선을 만들지 못했다. 모두 seed42 개발 실험이므로 최종 backbone 성능으로 해석하지 않으며 test split은 계속 sealed 상태로 유지한다.

실험 추적에는 local MLflow를 사용한다. Experiment는 `context_lstm_baseline_v1`이며 baseline, `batch8`, `dropout02`, `wd0005`, `lr00025`, `layernorm`, `labelsmooth005` run의 params, epoch metrics와 local artifacts를 backbone별로 관리한다. `mlflow.db`, `mlartifacts/`, `mlruns/`는 Git에서 제외한다.

## 6. Context Branch

| 항목 | 현재 값 |
|---|---|
| Eligible videos | 1,677 (train 1,382 / val 295) |
| Labels | drowsy 800 / not drowsy 877 |
| Context-only | 157 |
| Sequence | 32 target slots/video |
| Target rows | 53,664 |
| Original source rows | 52,965 |
| Imputed references | 699 |
| Crop input | RGB 224×224 |
| Stored CNN feature | `[32,512]` float32/video/backbone |

Context imputation은 새 이미지를 만드는 interpolation이 아니다. 같은 영상에서 이미 존재하는 가장 가까운 원본-valid Context source를 참조하며 chained imputation을 허용하지 않는다.

## 7. Behavior Branch

| 항목 | 현재 값 |
|---|---|
| Eligible videos | 1,520 (train 1,256 / val 264) |
| Labels | drowsy 736 / not drowsy 784 |
| Sequence | `[100,6]` float32/video |
| Features | EAR, MAR, pitch raw/centered candidate, yaw, roll |
| Missing representation | NaN + feature/detector/landmark/pose/behavior masks |
| Interpolation / event generation | 없음 |

Head Pose의 physical sign convention과 head-up/head-down 해석은 아직 확정되지 않았다. 저장된 연속값을 임의의 행동 의미로 재해석하지 않는다.

## 8. CNN Feature Extraction

| Backbone | Frozen weights | 512-D 구성 | Policy hash |
|---|---|---|---|
| ResNet18 | `ResNet18_Weights.IMAGENET1K_V1` | native average pooling 후 flatten | `0b8d72ab...c4cd3e` |
| VGG16 | `VGG16_Weights.IMAGENET1K_V1` | features → adaptive average pooling 1×1 → flatten | `1edc39db...0b2bc0` |

공통 조건은 RGB uint8 224×224, `/255`, ImageNet mean/std normalization, geometry transform 없음, augmentation 없음, float32, AMP off, eval/inference mode다. Full extraction batch size는 양쪽 모두 16이었다. 각 source JPEG는 backbone별로 한 번만 추론하고 target mapping에서 재사용했다.

두 backbone의 결과는 별도 artifact이며 현재 정확도 우위를 주장하지 않는다. 향후 동일한 LSTM·classifier·training protocol로 validation 성능을 비교한다.

## 9. Current Frozen Artifacts

| Artifact | 경로 | 상태 |
|---|---|---|
| Frozen split metadata | `data/metadata/{train,val,test}.csv` | 유지 |
| Canonical train/val | `data/interim/preprocessing_v2/canonical/` | 완료·read-only |
| Context sequence | `data/interim/sequences_v2/context/` | 완료·read-only |
| Behavior sequence | `data/interim/sequences_v2/behavior/` | 완료·read-only |
| ResNet18 full feature | `data/interim/features_v2/context_full/resnet18/` | 완료·read-only |
| VGG16 full feature | `data/interim/features_v2/context_full/vgg16/` | 완료·read-only |
| Cross-branch audit | `outputs/sequences_v2/cross_branch_audit/` | COMPLETE |
| D2R regression audit | `outputs/features_v2/context_cnn/full_regression_audit/` | COMPLETE |
| D3 final integrity audit | `outputs/features_v2/context_cnn/final_integrity_audit/` | PASS |

주요 동결 hash:

- Canonical policy: `29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721`
- Sequence missing policy: `f382c7900331c0be2f56b95f8fd95ed77f86dd5d4b06239da078e03d180eee4d`
- Full source mapping fingerprint: `83b4587fdb41e5ef4f02287a0de8076426910a65eefb02cf4cc032527b42e453`

대용량 원본·중간 artifact는 `.gitignore`로 제외하며 로컬에서 관리한다.

## 10. Repository Structure

```text
assets/      YuNet·Dlib68 model assets
configs/     preprocessing, sequence, feature, model, training 설정
data/
  metadata/  frozen split과 manifest
  raw/       원본 SUST-DDD 영상 (Git 제외)
  interim/   canonical, sequence, CNN feature artifact (Git 제외)
docs/        단계별 설계·정책·실행 기록
outputs/     audit summary와 검토 자료
scripts/     실행 및 감사 CLI
src/         preprocessing_v2, sequences_v2, evaluation_v2 구현
tests/       pytest 회귀 테스트
```

## 11. Environment

STEP 5-D feature extraction 및 감사에 기록된 환경:

| 항목 | 값 |
|---|---|
| OS | Windows |
| Python | 3.12.14 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| CUDA runtime | 13.0 |
| cuDNN | 91900 |
| GPU | NVIDIA GeForce RTX 3080 |
| matplotlib | 3.11.2 |

프로젝트 전용 `.venv` 사용을 권장한다. 자세한 기록은 [environment.md](docs/environment.md)와 [environment snapshot](docs/environment_snapshot.txt)을 참고한다.

## 12. Main Commands

아래는 구현 진입점을 보여 주는 참고용 명령이다. 현재 완료 artifact는 overwrite 보호 대상이며 목적 없이 재실행하지 않는다.

```powershell
# Source/split 검증
.\.venv\Scripts\python.exe scripts\validate_experiment2_source.py

# Canonical preprocessing과 sequence 구성
.\.venv\Scripts\python.exe scripts\run_canonical_preprocessing.py --help
.\.venv\Scripts\python.exe scripts\build_context_sequences.py --help
.\.venv\Scripts\python.exe scripts\build_behavior_sequences.py --help

# Cross-branch 및 CNN feature 감사
.\.venv\Scripts\python.exe scripts\run_cross_branch_sequence_audit.py --help
.\.venv\Scripts\python.exe scripts\extract_context_cnn_features.py --help
.\.venv\Scripts\python.exe scripts\audit_context_cnn_feature_regression.py --help
.\.venv\Scripts\python.exe scripts\audit_context_cnn_features.py

# Context sequence 시각화
.\.venv\Scripts\python.exe scripts\visualize_context_sequence.py --help

# Local MLflow server와 Context baseline training
.\scripts\start_mlflow.ps1
.\.venv\Scripts\python.exe scripts\train_context_lstm.py --backbone resnet18 --seed 42 --device cuda --mlflow

# 회귀 테스트
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
```

`audit_context_cnn_features.py`는 기존 final audit 출력이 있으면 덮어쓰지 않고 중단한다.

### Behavior Sequence Visual Inspection

STEP 5-B의 저장된 `[100,6]` 값과 mask를 threshold·event 해석 없이 로컬에서 검수한다. 출력은 `outputs/visualizations/behavior_sequences/`에 저장되며 Git에서 제외된다. Test split은 계속 sealed 상태다.

```powershell
# Overview를 화면에 표시
.\.venv\Scripts\python.exe scripts\visualize_behavior_sequence.py --video-id <ID> --mode overview --show

# Headless 저장
.\.venv\Scripts\python.exe scripts\visualize_behavior_sequence.py --video-id <ID> --mode overview --save --no-show

# Mask 전용 화면
.\.venv\Scripts\python.exe scripts\visualize_behavior_sequence.py --video-id <ID> --mode masks --show

# Train/validation 대표 검수 후보만 출력
.\.venv\Scripts\python.exe scripts\visualize_behavior_sequence.py --suggest-samples
```

Train/validation 대표 표본 `d_10`, `d_246`, `d_690`을 post-STEP-5 visual sanity check로 확인했다. 저장된 missing slot과 validity mask가 일치했고, NaN 구간은 보간 없이 끊겨 표시됐으며, EAR/MAR는 missing 구간 밖에서 구조적인 sequence 이상 없이 연속적으로 관찰됐다.

일부 valid head-pose frame에서는 `pitch_raw`의 큰 전환과 간헐적인 `roll` spike가 관찰됐다. 이는 sequence construction 실패가 아닌 feature-quality 주의 사항이며, wrap 또는 표현 불연속의 가능성만 기록한다. 표본에서 `pitch_centered_candidate`가 상대적으로 더 완만해 보였지만 품질 우위를 주장하지 않는다. Behavior baseline은 여섯 feature를 모두 유지한 `[100,6]` 그대로이며, head-pose 방향 의미와 rule threshold는 아직 확정하지 않았다. 검수에는 train/validation만 사용했고 test는 sealed 상태를 유지했다.

검수 시 EAR/MAR/pose 연속값의 비정상 급변, NaN과 mask의 위치, 연속 missing run, 100-slot timeline을 확인한다. 이 화면은 sanity check일 뿐 전체 품질의 통계적 증명이 아니며 PERCLOS, threshold, yawn, head-down 또는 nod 의미를 계산하지 않는다.

## 13. Experiment Rules / Leakage Protection

- Test split은 최종 configuration 선택 전까지 **SEALED**다.
- Test를 preprocessing 정책, missing policy, threshold, backbone, hyperparameter 또는 checkpoint 선택에 사용하지 않는다.
- Train/validation에서만 정책을 만들고 비교한다.
- Context와 Behavior eligibility는 독립적으로 유지한다.
- Label-dependent preprocessing과 imputation을 금지한다.
- Frozen artifact의 hash, target/source provenance와 mask를 보존한다.
- ResNet18과 VGG16 feature를 하나의 입력으로 합치지 않는다.
- LSTM 학습 중 CNN feature를 epoch마다 재추출하지 않고 저장된 `[32,512]`를 immutable input으로 사용한다.

## 14. Known Numerical Note

D1 pilot과 D2 full artifact 비교에서 양쪽 backbone 모두 16개 영상 중 15개가 exact match했고 `n_311`만 차이가 있었다. 영향 target은 24–31번이며 6개 unique source에서 발생했다.

D1의 마지막 partial batch 6장과 D2의 full batch 16장을 각각 재현했을 때 저장 vector는 해당 batch 결과와 backbone별 6/6 exact match했다. 따라서 분류는 **BENIGN FLOATING-POINT / BATCH NUMERICAL DIFFERENCE**다. Mapping, input 또는 artifact corruption이 아니며 full 재추출은 필요하지 않다. 기존 `rtol=1e-5`, `atol=1e-6`과 15/16 기록을 유지하고 16/16 PASS로 바꾸지 않는다.

향후 LSTM은 고정 저장된 D2 feature를 읽으므로 이 CNN batch-composition 차이가 epoch별 입력 변동을 만들지 않는다.

## 15. Current Status / Next Steps

```text
STEP 1   COMPLETE
STEP 2   FULLY CLOSED
STEP 3   COMPLETE
STEP 4   FULLY CLOSED
STEP 5-A COMPLETE
STEP 5-B COMPLETE
STEP 5-C COMPLETE
STEP 5-D1 COMPLETE
STEP 5-D2 COMPLETE
STEP 5-D2R COMPLETE
STEP 5-D3 PASS
STEP 5   FULLY CLOSED
STEP 6-A COMPLETE
STEP 6-B COMPLETE
STEP 6-C COMPLETE
TEST     SEALED
```

현재 단일 변수 tuning 후보는 모두 공통 정책으로 미채택했다. 아직 seed42 development stage이며 최종 3-seed validation 이전 단계다.

다음 작업은 **Train vs Validation Context Sequence / Frozen CNN Feature Audit**이다.

1. Train/validation의 missing 및 imputation 특성 비교
2. Frozen feature distribution과 temporal variation 차이 점검
3. 데이터 차이의 근거를 확인한 뒤 다음 학습 정책 결정
4. 공통 정책 동결 후 ResNet18 / VGG16 × seeds 42, 123, 2026 검증
5. 모든 선택을 동결한 뒤 최종 test를 한 번 평가

## 16. Limitations

- 현재 split은 video-level이며 subject-wise unseen-driver 독립성을 보장하지 않는다.
- 현재 LSTM 결과는 seed42 train/validation baseline 및 단일 변수 tuning에 한정되며 일반화 성능이나 최종 backbone 우위를 확정하지 않는다.
- STEP 2의 detector/crop 선택은 통제된 train/validation audit 결과이며 전체 조건에서의 절대적 최적성을 뜻하지 않는다.
- Context nearest-source reuse가 성능을 향상한다는 주장은 아직 없다.
- Behavior branch의 threshold, event 정의와 head-pose physical sign convention은 미확정이다.
- Pilot runtime과 feature 통계는 시스템 진단값이며 정확도 비교 근거가 아니다.
- Test split은 계속 sealed 상태이며 최종 test 성능은 아직 존재하지 않는다.

## 17. References

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). arXiv:1607.06450.
