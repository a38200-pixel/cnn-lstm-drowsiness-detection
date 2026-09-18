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

Status: **COMPLETED — READY_FOR_PREPROCESSING_AUDIT**

#### 목적

Experiment 2를 시작하기 전에 원본 SUST-DDD dataset, 기존 frozen split, asset, 실제 영상 decode 및 model load 상태를 검증했습니다. 2차 실험에서도 1차 실험의 frozen video-level train/validation/test split을 그대로 유지하고 split을 다시 생성하지 않습니다. 따라서 향후 1차와 2차 실험을 비교할 때 split 변화에 따른 효과를 제거할 수 있습니다.

원본 데이터 경로는 `C:\Users\AISW_203_113\Documents\GitHub\cnn-lstm-drowsiness-detection\data\raw\SUST Driver Drowsiness Dataset`이며, 원본 영상이나 metadata는 수정하지 않았습니다. metadata와 raw 영상의 실제 대응, split 무결성, 소수 영상의 decode 가능 여부, 현재 환경에서 Dlib68 및 YuNet asset의 실제 load 여부를 확인했습니다. 이 단계에서는 전체 영상 preprocessing이나 model training을 수행하지 않았습니다.

#### 결과

| Check | Result |
|---|---:|
| Raw videos | 2,074 |
| Metadata videos | 2,074 |
| Matched videos | 2,074 |
| Metadata missing from raw | 0 |
| Raw missing from metadata | 0 |
| Train videos | 1,452 |
| Validation videos | 311 |
| Test videos | 311 |
| Train ∩ Validation | 0 |
| Train ∩ Test | 0 |
| Validation ∩ Test | 0 |
| Decode smoke test | PASS — 6/6 samples |
| Dlib68 load | PASS |
| YuNet load | PASS |
| Report errors | 0 |
| Report warnings | 0 |

| Split | Drowsy | Not drowsy | Total |
|---|---:|---:|---:|
| Train | 683 (47.04%) | 769 (52.96%) | 1,452 |
| Validation | 146 (46.95%) | 165 (53.05%) | 311 |
| Test | 146 (46.95%) | 165 (53.05%) | 311 |

공식 STEP 1 artifact는 `outputs/preprocessing_v2/source_validation/`에 보존합니다.

- `source_validation_report.txt`
- `source_validation_summary.json`
- `metadata_schema.json`
- `split_integrity.json`
- `raw_video_inventory.csv`

실행 명령:

```bash
python scripts/validate_experiment2_source.py
```

#### Final Decision

`READY_FOR_PREPROCESSING_AUDIT`

#### Environment

Experiment 2는 프로젝트 전용 `.venv`를 사용합니다. 실제 Python 및 package version, Windows의 Dlib 구성 방식, dependency snapshot과 Docker 이전 계획은 [docs/environment.md](docs/environment.md)를 참고하세요.

#### Known Limitations

- 현재 split은 **video-level split**이며 subject-wise split이 아닙니다.
- 원본 데이터에 충분한 subject ID mapping이 없어 subject-wise independence를 보장할 수 없습니다.
- STEP 1 실행 중 OpenCV의 `Targets are not supported by the new graph engine for now` warning이 관찰됐지만 YuNet load는 성공했습니다. 실제 runtime 동작과 latency는 STEP 2에서 다시 확인합니다.
- STEP 1은 asset load와 제한된 decode smoke test이며 detector/landmark 조합의 실제 품질을 검증하지 않았습니다.

#### Next Step

STEP 2에서는 소규모 샘플을 대상으로 다음 preprocessing audit을 수행합니다.

`HOG primary → HOG failure → YuNet fallback → Dlib68 landmarks → EAR/MAR/HeadPose → automatic geometry validation → visual audit → detector policy 결정`

STEP 2-A 자동 audit 구현과 실행이 완료됐으며 아래에 실제 결과를 기록합니다.

### STEP 2-A — HOG → YuNet Fallback Compatibility Audit

Status: **AUTOMATIC AUDIT COMPLETE — PRIMARY DETECTOR POLICY NOT FINALIZED**

#### 목적과 데이터 범위

전체 2,074개 영상을 전처리하기 전에 train/validation의 160개 frame에서 HOG, YuNet fallback, Dlib68 조합의 호환성과 geometry 품질을 검증했습니다. 기존 frozen split은 변경하지 않았으며 **test split은 사용하지 않았습니다.**

라벨은 train/validation 및 drowsy/not_drowsy 표본 균형과 보고서 표시에만 사용합니다. detector 실행, fallback, bbox 선택, landmark, EAR/MAR, Head Pose, geometry validity에는 라벨을 전달하지 않아 같은 frame이 라벨과 무관하게 동일하게 처리되도록 구성했습니다.

#### Candidate Pipeline

`Raw frame → HOG primary → HOG 실패 시 YuNet fallback → 선택된 모든 얼굴에 Dlib68 → EAR/MAR/Head Pose → geometry validation → visual audit`

- HOG와 YuNet은 face detector만 다르며 landmark는 모두 Dlib68로 통일합니다.
- production candidate에서는 HOG 성공 시 YuNet을 실행하지 않습니다.
- 소수 HOG 성공 frame의 paired audit에서만 비교 목적으로 YuNet을 추가 실행합니다.
- multiple face는 detector 종류와 관계없이 유효 bbox 중 면적이 가장 큰 얼굴을 선택하며 검출 개수도 기록합니다.
- 3DDFA와 1차 실험의 Dlib↔3DDFA calibration은 사용하지 않습니다.
- EAR 졸음 threshold, MAR yawn threshold, Head Pose 행동 threshold는 결정하지 않습니다.
- 자동 audit 직후에도 detector 정책을 승인하지 않으며 Decision은 `WAITING_FOR_MANUAL_VISUAL_REVIEW`입니다.

#### 실행

설정과 sampling 범위만 확인:

```bash
python scripts/run_detector_landmark_audit.py --dry-run
```

소규모 audit 실행:

```bash
python scripts/run_detector_landmark_audit.py --config configs/detector_landmark_audit.yaml
```

결과는 `outputs/preprocessing_v2/detector_landmark_audit/`에 생성됩니다.

- `sample_results.csv`
- `paired_detector_results.csv`
- `audit_report.txt`
- `audit_summary.json`
- `manual_review.csv`
- `MANUAL_REVIEW_GUIDE.md`
- `visual_samples/`
- `contact_sheets/`

실행 후 contact sheet와 개별 annotation 이미지를 확인하고 `manual_review.csv`의 `bbox_ok`, `landmarks_overall_ok`, `eyes_ok`, `mouth_ok`, `nose_chin_ok`, `pose_axis_ok`, `manual_decision`, `review_note`를 직접 작성해야 합니다.

#### 현재 Decision과 다음 단계

| 항목 | 실제 결과 |
|---|---:|
| Candidate frames | 160 |
| HOG detection | 152/160 (95.00%) |
| HOG failure | 8 |
| YuNet fallback recovery | 6/8 (75.00%) |
| Combined detection | 158/160 (98.75%) |
| HOG bbox → Dlib68 | 152/152 |
| YuNet bbox → Dlib68 | 6/6 |
| HOG detector 평균 latency | 371.82 ms |
| YuNet fallback 평균 latency | 39.48 ms |
| Dlib68 평균 latency | 약 3.30 ms |

Head Pose에서는 OpenCV Euler pitch가 ±180° 부근으로 표현되어 raw pitch 중앙값이 159.40°였고, 158개 유효 pose가 모두 기존 large-pose flag 대상이 됐습니다. 12개 paired sample의 단순 pitch 절댓값 차이는 최대 357.44°였습니다. 이는 Head Pose 모델 실패로 단정할 결과가 아니라 ±180° wrap-around를 고려하지 않은 **표현 및 비교 convention 문제**입니다.

따라서 primary detector 정책은 아직 확정하지 않습니다. STEP 2-B에서 raw angle을 보존하면서 circular angular distance를 사용하고, front-centered pitch 후보 표현과 pose axis를 visual review합니다.

### STEP 2-B — HOG vs YuNet Primary Detector Comparison

Status: **IMPLEMENTED — WAITING FOR MANUAL RUN**

#### 목적

STEP 2-A의 동일한 160개 frame에서 HOG와 YuNet을 모두 실행하고, detector 이외의 조건을 동일하게 통제해 primary detector 후보를 비교합니다. Test split과 새로운 random sampling은 사용하지 않습니다.

Experiment 1 비교는 `HOG → Dlib68`과 `YuNet → 3DDFA`처럼 landmark source가 달라 EAR/MAR scale mismatch와 calibration failure가 발생했습니다. Experiment 2 비교는 `HOG → Dlib68`과 `YuNet → Dlib68`로 동일한 landmark source, EAR/MAR 공식, Head Pose estimator를 사용합니다. 따라서 이전에 YuNet을 fallback으로 제한했던 핵심 제약이 이번 비교에 그대로 적용되지는 않습니다.

또한 STEP 2-A에서 HOG 평균 latency가 약 372 ms로 측정되어 Behavior branch의 10 FPS 참고 budget인 100 ms/frame에 불리할 가능성이 확인됐습니다. 이는 자동 채택 기준이 아니며, YuNet primary의 검출·landmark·crop 품질과 함께 비교하기 위한 근거입니다.

#### Head Pose 비교 수정

- `pitch_raw`, `yaw_raw`, `roll_raw`는 그대로 보존합니다.
- detector 차이의 주요 metric으로 ±180°를 고려한 circular angular distance를 사용합니다.
- `pitch_centered_candidate = wrap_to_180(pitch_raw - 180)`을 정면≈0 분석 후보로 추가합니다.
- centered pitch의 부호와 실제 head up/down 관계는 가정하지 않으며 visual review 후 결정합니다.
- large-pose audit 표본은 raw pitch가 아니라 centered 후보의 절댓값으로 선택합니다.

#### Full Face crop preview

HOG와 YuNet bbox 모두에 25% margin, square crop, frame 밖 고정값 padding, RGB 224×224의 동일한 audit preview 정책을 적용합니다. 이는 Context model 학습 데이터나 최종 crop 정책이 아닙니다.

#### 실행 및 결과

```bash
python scripts/run_detector_primary_comparison.py --dry-run
python scripts/run_detector_primary_comparison.py --config configs/detector_primary_comparison.yaml
```

실행 결과는 STEP 2-A와 분리된 `outputs/preprocessing_v2/detector_primary_comparison/`에 생성됩니다.

- `comparison_report.txt`
- `comparison_summary.json`
- `paired_primary_results.csv`
- `manual_review.csv`
- `MANUAL_REVIEW_GUIDE.md`
- `visual_samples/`
- `contact_sheets/`

자동 비교 후에도 Decision은 `WAITING_FOR_MANUAL_PRIMARY_DETECTOR_REVIEW`입니다. Comparison image와 crop, landmark, pose axis를 수동 검토하기 전에는 HOG 또는 YuNet을 primary detector로 승인하지 않습니다.
