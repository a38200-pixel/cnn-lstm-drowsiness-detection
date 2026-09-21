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

| 단계 | 현재 상태 |
|---|---|
| STEP 1 — Source Data Validation | COMPLETE |
| STEP 2 — Preprocessing Policy Audit & Selection (2-A ~ 2-D) | **FULLY CLOSED** |
| STEP 3-A — Canonical Pipeline Implementation | **IMPLEMENTED** |
| STEP 3-B — 20-Video Pilot | **COMPLETE** |
| STEP 3-C — Full Train/Val Materialization | **COMPLETE** |
| STEP 4-A — Full Train/Val Automatic Quality & Missing Audit | **COMPLETE** |
| STEP 4-B — Missing / Quality Manual Visual Review | **COMPLETE** |
| STEP 4-C — Missing Handling Policy Selection & Freeze | **NOT STARTED** |

현재 milestone 요약: STEP 1에서 SUST-DDD 2,074개 영상과 video-level train/val/test 1,452/311/311개, split 중복 0개를 확인했다. 영상별 subject 매핑이 없어 unseen-driver 독립성은 보장하지 않는다. STEP 2에서 HOG 152/160(95.0%, 평균 372.15 ms)과 YuNet 157/160(98.125%, 평균 39.44 ms)을 통제 비교하고 YuNet primary를 확정했다. Dlib68은 YuNet 성공 157건에서 RAW/M05/M10/M15 모두 157/157 성공했으며, clipping metadata 오류는 저장 ROI 628/628개가 맞는 `REPORTING_ONLY_BUG`였다. 최종 정책은 YuNet → RAW bbox Dlib68 → EAR/MAR/Head Pose 및 SQUARE_M10 RGB 224×224, ImageNet mean padding이다. Context 선택은 모델 정확도 우위가 아닌 geometry/시각 정책 결정이다.

STEP 3-A는 영상당 10 Hz × 10초의 100개 canonical slot과 검출 독립적인 32개 Context slot, 결측 무대체, atomic bundle·resume·policy/run hash·모델 SHA-256·source fingerprint·integrity 검사를 구현했다. STEP 3-B의 20개 pilot은 2,000/2,000 decode, YuNet 1,916/2,000(95.8%), 검출 후 Dlib68 1,916/1,916, Context 613/640(95.78%) 가용 및 27개 결측이었다. 자동·사용자 시각 검토를 거쳐 STEP 3-B를 완료했고, STEP 3-C에서 train/val 1,763개 영상·176,300 canonical row·54,672 Context crop을 생성해 무결성·resume 검증을 통과했다. Pilot의 품질 분포를 전체 영상에 일반화하지 않는다. **TEST SPLIT SEALED**.

STEP 2에서 확정한 범위는 face detector, Dlib68 fitting ROI, Context CNN crop geometry입니다. 행동 임계값·시간 규칙과 모델 성능은 아직 확정되지 않았습니다. 수치, 후보별 판단, 원본 artifact의 당시 Decision 상태는 [STEP 2 preprocessing policy 상세 기록](docs/experiment2_step2_preprocessing_policy.md)에 정리했습니다.

STEP 2 시각 검토 결론은 당시 사용자가 제공한 내용입니다. STEP 2 원본 자동 report의 `WAITING_FOR_MANUAL_*` 상태와 비어 있는 해당 단계의 수동 검토 CSV 판정 칸은 그대로 보존하며, CSV에 검토 결과가 입력됐다고 주장하지 않습니다. STEP 3-B의 별도 missing Context CSV에는 이번 사용자 제공 판정을 기록했습니다.

### Experiment 2 — Frozen Preprocessing Policy after STEP 2

| 항목 | 확정 정책 |
|---|---|
| Face detector | **YuNet** (primary) |
| Landmark predictor / fitting ROI | **Dlib68 / YuNet RAW bbox** |
| Behavior features | EAR / MAR / Head Pose |
| Head Pose 비교 | Circular angular distance; raw angle 보존 |
| Context crop | **SQUARE_M10**: YuNet bbox의 width·height 각각 양쪽으로 10% 확장 → 중심 유지 정사각형 → frame 밖 ImageNet-mean padding → RGB 224×224 |
| Context model 계획 | ResNet18과 VGG16을 별도 실험으로 비교; 아직 구현·학습 전 |

```text
Frame → YuNet → YuNet RAW bbox
                   ├─ Behavior: RAW bbox → Dlib68 → EAR / MAR / Head Pose
                   └─ Context: 양쪽 10% margin → 중심 유지 square
                               → ImageNet-mean padding → RGB 224×224
                               → ResNet18 또는 VGG16 (이후 단계, 미구현)
```

### Still Unresolved After STEP 2

- `pitch_centered_candidate = wrap_to_180(pitch_raw - 180)`은 분석 표현 후보입니다. 양·음의 물리적 head-up/head-down 대응은 아직 확정하지 않았습니다.
- EAR closure, PERCLOS, MAR/yawn, head-drop/nod의 threshold와 시간 규칙은 미확정입니다.
- Sequence 생성·모델 학습은 시작하지 않았습니다. SQUARE_M10은 crop geometry 선택이지 최고 모델 정확도 입증이 아닙니다.
- 모든 STEP 2 audit은 train/validation만 사용했습니다. **TEST SPLIT SEALED**: 최종 평가 전에는 preprocessing 정책·임계값·detector·crop·rule calibration·model 선택에 test를 사용하지 않습니다.
- 기존 split은 video-level stratified split입니다. per-video subject mapping이 없어 subject-wise 독립성을 보장할 수 없으며 unseen-subject 일반화 성능을 주장하지 않습니다.

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

STEP 2에서는 소규모 샘플을 대상으로 다음 preprocessing audit을 수행했습니다.

`HOG primary → HOG failure → YuNet fallback → Dlib68 landmarks → EAR/MAR/HeadPose → automatic geometry validation → visual audit → detector policy 결정`

STEP 2-A부터 2-D까지 완료됐으며 아래에 실제 결과와 최종 정책을 기록합니다.

### STEP 2-A — Detector / Landmark Compatibility Audit

Status: **COMPLETE** — 당시 자동 audit에서는 detector 정책을 보류했고, 이후 STEP 2-B에서 YuNet primary를 확정했습니다.

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

STEP 2-A 당시에는 primary detector 정책을 확정하지 않았습니다. 이후 STEP 2-B에서 raw angle을 보존하고 circular angular distance와 front-centered pitch 분석 후보를 도입했습니다. Head Pose의 실제 up/down 부호 규약은 여전히 미확정입니다.

### STEP 2-B — HOG vs YuNet Primary Detector Comparison

Status: **COMPLETE — PRIMARY FACE DETECTOR: YUNET**

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

자동 비교 수치와 사용자 제공 visual review 결론을 종합해 **YuNet을 Experiment 2의 primary face detector로 확정**했습니다. 자동 report의 수동 검토 대기 Decision은 생성 당시 상태이며, 이후 STEP 2-C에서 Dlib68 fitting ROI, STEP 2-D에서 Context crop을 각각 확정했습니다. Head Pose sign convention과 시간축 threshold는 여전히 미확정입니다.

#### Visual Review Pack

STEP 2-B 자동 비교 결과를 바탕으로 중복 제거한 수동 검토 표본 45개와 고해상도 검토 시트 5장을 준비했습니다. YuNet-only 6개, HOG-only 1개, both-failed 2개를 모두 포함하며, 지표별 극단값 후보와 train/validation 및 drowsy/not_drowsy 균형 정상 참조 10개를 함께 제공합니다.

```bash
python scripts/build_detector_review_pack.py --dry-run
python scripts/build_detector_review_pack.py
```

생성 위치는 `outputs/preprocessing_v2/detector_primary_comparison/review_pack/`이며 다음 파일을 포함합니다.

- `review_manifest.csv`
- `review_manual.csv`
- `review_numeric_summary.md`
- `repeated_outlier_videos.csv`
- `selected_images/`
- `review_sheet_*.jpg`

배포용 묶음은 `outputs/preprocessing_v2/detector_primary_comparison/step2b_visual_review_pack.zip`입니다.

#### 실제 결과와 visual review 결론

Experiment 1의 `HOG → Dlib68` 대 `YuNet → 3DDFA` 비교와 달리, STEP 2-B는 `HOG → Dlib68`과 `YuNet → Dlib68`로 landmark model과 EAR/MAR·Head Pose 계산을 통제했습니다.

| Metric | HOG | YuNet |
|---|---:|---:|
| Detection success | 152/160 (95.0%) | 157/160 (98.125%) |
| Detector latency mean | 372.15 ms | 39.44 ms |
| Behavior pipeline mean | 376.17 ms | 43.78 ms |
| Dlib68 success after detection | 152/152 (100%) | 157/157 (100%) |

- Both success: 151
- HOG only: 1
- YuNet only: 6
- Both failed: 2
- Test split은 사용하지 않았습니다.
- YuNet bbox는 육안상 얼굴 전체를 더 자연스럽게 포함하는 사례가 많았습니다.
- HOG bbox에는 얼굴 좌우 padding이 상대적으로 큰 사례가 관찰됐습니다.
- 따라서 HOG bbox geometry를 landmark 정확도의 ground truth로 취급하지 않습니다.

같은 Dlib68 predictor를 사용해도 입력 rectangle geometry가 달라지면 landmark fitting과 EAR/MAR가 달라질 수 있습니다. EAR Pearson 약 0.60과 MAR Pearson 약 0.27은 YuNet detector 실패의 증거가 아니라, detector bbox와 Dlib68 fitting ROI가 서로 다른 정책 문제임을 보여주는 진단 신호입니다. 다음 단계에서는 YuNet detection bbox는 고정하고 Dlib68에 전달할 ROI margin만 비교합니다.

Head Pose는 raw Euler angle의 ±180° wrap 문제 때문에 circular angular difference로 비교합니다. `pitch_centered_candidate`는 분석 후보일 뿐이며 부호 규약과 generic camera model의 angle을 ground truth로 간주하지 않습니다.

STEP 2-B 당시 결론과 이후 확정 상태는 다음과 같습니다.

- Primary face detector: **YuNet** (이번 표본의 통제된 비교 결과; 모든 조건에서의 우월성을 주장하지 않음)
- 이후 STEP 2-C에서 Dlib68 fitting ROI는 RAW bbox, STEP 2-D에서 Context crop은 SQUARE_M10으로 확정
- 계속 미확정: Head Pose sign convention과 EAR/MAR 등 행동 threshold·시간 규칙

### STEP 2-C — YuNet Landmark ROI Margin Audit

Status: **COMPLETE — LANDMARK ROI: YUNET RAW BBOX → DLIB68**

YuNet detector 선택과 Dlib68 fitting rectangle을 분리해 평가합니다. STEP 2-B의 동일 160개 frame 중 YuNet 성공 157개만 사용하고, 저장된 YuNet raw detection bbox를 변경하거나 detector를 재실행하지 않습니다. HOG 값은 참고값일 뿐 ground truth가 아닙니다.

후보는 RAW(0%), M05(+5%), M10(+10%), M15(+15%)입니다. bbox width와 height에 margin을 각각 독립적으로 적용합니다. 예를 들어 `(100, 100, 300, 300)`에 5%를 적용하면 `(90, 90, 310, 310)`이 됩니다. 강제 square, HOG aspect ratio·크기 복제, bbox 중심 변경, detector ensemble은 적용하지 않습니다. frame 밖으로 나간 ROI는 clipping하고 clipping 여부와 손실 면적 비율을 기록합니다.

각 후보에서 동일 Dlib68로 landmark, EAR, MAR, raw/centered pitch, yaw, roll, geometry validity와 latency를 계산합니다. RAW↔M05/M10/M15 및 인접 margin의 normalized landmark difference도 기록합니다. 같은 영상의 인접 audit timestamp 변화량은 안정성 진단에만 사용하며 drowsy/not_drowsy 판정이나 자동 best margin 선정에는 사용하지 않습니다.

시각 자료는 `Original | RAW | +5% | +10% | +15%` 5-panel 구조이며, YuNet detection bbox와 landmark fitting ROI를 다른 색으로 표시합니다. 수동 검토자는 눈·입·전체 landmark와 pose axis를 확인하고 `preferred_landmark_roi`에 `RAW`, `M05`, `M10`, `M15`, `TIE`, `UNCERTAIN` 중 하나를 기록합니다.

```bash
python scripts/run_landmark_roi_margin_audit.py --dry-run
python scripts/run_landmark_roi_margin_audit.py --config configs/landmark_roi_margin_audit.yaml
```

실제 실행 결과는 `outputs/preprocessing_v2/landmark_roi_margin_audit/`에 보존되어 있습니다. 수동 시각 검토를 거쳐 YuNet RAW bbox를 Dlib68 fitting ROI로 선택했습니다. 이는 Context CNN crop 정책과는 별개입니다.

| 후보 | Dlib68·geometry·EAR·MAR·pose 성공 | RAW 대비 landmark 차이 평균 | EAR 평균 | MAR 평균 |
|---|---:|---:|---:|---:|
| RAW | 각 157/157 | 기준 | 0.369 | 0.148 |
| M05 | 각 157/157 | 0.0278 | 0.406 | 0.194 |
| M10 | 각 157/157 | 0.0560 | 0.455 | 0.231 |
| M15 | 각 157/157 | 0.1059 | 0.470 | 0.272 |

Margin을 늘려도 성공률은 개선되지 않았고 landmark fitting과 EAR/MAR 분포가 이동했습니다. 사용자 제공 시각 검토에서는 RAW가 눈·입·턱을 대체로 잘 따랐고, M05의 일관된 개선은 없었으며 M10/M15 일부 표본은 contour·mouth·pose 변화가 컸습니다. 따라서 **LANDMARK ROI = YUNET RAW BBOX**로 확정했습니다.

Landmark ROI margin과 Context CNN full-face crop margin은 별도 정책입니다. Context branch의 RGB 224×224 crop은 이후 STEP 2-D에서 RAW_RESIZE, SQUARE_0, SQUARE_M10, SQUARE_M20으로 독립 비교했습니다.

#### STEP 2-C Visual Review Pack

Visual review pack 준비를 완료했습니다. 기존 18개 contact sheet와 수동 검토 대상으로 선정된 35개 5-panel 이미지를 다시 계산하지 않고 후처리하여, 중복 없는 35개 표본과 고해상도 compact sheet 5장으로 재구성했습니다.

- Overview / Normal Reference: 7개
- Eye / EAR: 7개
- Mouth / MAR: 7개
- Head Pose / Landmark Change: 7개
- ROI / Edge / Difficult Cases: 7개
- detector, Dlib68, EAR/MAR, Head Pose 및 ROI margin audit 재실행 없음
- 자동 metric은 검토 표본 분류에만 사용하고 best margin은 자동 선택하지 않음

산출물은 `outputs/preprocessing_v2/landmark_roi_margin_audit/review_pack/`에 있으며, 배포용 ZIP은 `outputs/preprocessing_v2/landmark_roi_margin_audit/step2c_visual_review_pack.zip`입니다.

기존 결과의 clipping flag 불일치는 아래 별도 보정 기록을 참조하십시오. 당시 생성된 audit 및 review pack 원본은 수정하지 않았습니다.

#### STEP 2-C Clipping Metadata Correction

기존 report의 `roi_clipped`는 RAW 0/157, M05 157/157, M10 155/157, M15 157/157인데 세 확장 후보의 `clipped_fraction`은 모두 0.0이었습니다. 원인은 frame clipping 전의 **실수 좌표**와 `floor/ceil`로 정수화된 최종 ROI를 직접 비교한 boolean 계산식입니다. 정수화 차이를 경계 clipping으로 오판했으며, Dlib의 `x2-1, y2-1` 변환은 원인이 아닙니다.

이제 margin 적용 후 반개방 정수 `intended_roi`를 만든 다음 같은 좌표계의 frame-clipped ROI와 비교합니다. 두 ROI가 다를 때만 `roi_clipped=True`이며, `clipped_fraction`은 `1 - clipped_area/intended_area`입니다. 정확히 경계에 닿거나 실수 좌표를 정수화하기만 한 경우는 clipping이 아닙니다.

기존 157개 frame에 대해 train/val metadata의 frame 크기와 저장된 YuNet bbox만으로 재계산했습니다.

| Candidate | 기존 clipped | 보정 clipped | 보정 비율 |
|---|---:|---:|---:|
| RAW | 0/157 | 0/157 | 0% |
| M05 | 157/157 | 0/157 | 0% |
| M10 | 155/157 | 0/157 | 0% |
| M15 | 157/157 | 0/157 | 0% |

628개 후보의 보정 ROI 좌표가 기존 CSV에 저장된 ROI 좌표와 모두 일치합니다. 따라서 영향은 **`REPORTING_ONLY_BUG`**입니다. Dlib68 fitting, EAR/MAR, Head Pose, visual review 결과는 변경되지 않았고 ROI audit 재실행도 필요하지 않습니다. 시각 검토에서 선택한 `YuNet RAW bbox → Dlib68` 방향을 유지합니다. Context CNN crop은 이후 STEP 2-D에서 확정했고, Head Pose sign convention과 temporal rule threshold는 여전히 미확정입니다.

원본 `roi_margin_report.txt`, `roi_margin_summary.json`, `roi_margin_results.csv`는 그대로 보존하고, 보정 내역은 `outputs/preprocessing_v2/landmark_roi_margin_audit/clipping_metadata_correction/`의 report·summary·CSV에 기록했습니다. 경계 방향별 clipping, 경계 정확히 접촉, 실수 좌표 정수화, 면적 손실 및 기존 157개 결과 교차 검증을 포함한 회귀 테스트 9개를 추가했습니다. 기존 Dlib inclusive 끝점 테스트도 유지했습니다. 전체 검증은 **39 passed, 5 subtests passed**이며 `git diff --check`에서 공백 오류는 없었습니다.

Clipping 보정은 **`REPORTING_ONLY_BUG`**, **`NO ROI AUDIT RERUN REQUIRED`**입니다. 저장된 ROI 좌표 628/628개가 보정 계산과 일치하므로 실제 Dlib68 입력, EAR/MAR, Head Pose와 시각 검토 결과는 바뀌지 않았습니다. 당시 미확정이던 Context CNN crop 정책은 이후 STEP 2-D에서 독립적으로 검토해 확정했습니다.

### STEP 2-D — Context CNN Crop Policy Audit

Status: **COMPLETE — CONTEXT CROP: SQUARE_M10**

YuNet detection bbox는 고정하고, Dlib68 fitting에는 STEP 2-C에서 확정한 RAW bbox를 그대로 사용합니다. Context CNN crop ROI는 별도 분기이므로 landmark ROI margin과 같은 값을 강제하지 않습니다. 저장된 YuNet bbox를 기준으로 `RAW_RESIZE`, `SQUARE_0`, `SQUARE_M10`, `SQUARE_M20`의 RGB 224×224 audit preview를 비교합니다. 이 단계는 CNN 구현이나 학습이 아닙니다.

```text
YuNet detection bbox
├─ Dlib68 fitting ROI: RAW bbox (STEP 2-C 완료)
└─ Context CNN crop ROI: SQUARE_M10 (STEP 2-D 완료)
```

M10/M20은 bbox width와 height의 각각 10%/20%를 양쪽에 더한 뒤 긴 축을 줄이지 않고 중심 기준 정사각형으로 확장합니다. frame 밖 영역은 ROI를 줄이는 대신 ImageNet mean RGB `[124, 116, 104]`(OpenCV BGR `[104, 116, 124]`)로 padding합니다. 양 축 축소는 `INTER_AREA`, 확대가 포함되면 `INTER_LINEAR`를 사용합니다. 자동 수치는 distortion, 얼굴 점유율, padding과 시간적 변동을 찾는 진단용이며, 실제 시각 검토를 종합해 **SQUARE_M10**을 선택했습니다. CNN 구현·학습은 아직 시작하지 않았습니다.

```bash
python scripts/run_context_crop_audit.py --dry-run
python scripts/run_context_crop_audit.py --config configs/context_crop_audit.yaml
```

실제 실행 결과는 `outputs/preprocessing_v2/context_crop_audit/`에 보존돼 있습니다. 160개 train/validation frame 중 YuNet 성공 157개에 대해 crop을 비교했고, 42개 visual sample과 11개 contact sheet를 검토했습니다. 원본 자동 report의 Decision `WAITING_FOR_MANUAL_CONTEXT_CROP_REVIEW`는 생성 당시 상태입니다.

| 후보 | 핵심 진단 및 검토 결과 |
|---|---|
| RAW_RESIZE | distortion ratio 중앙값 1.426; 육안상 aspect-ratio 왜곡 반복 |
| SQUARE_0 | 얼굴 면적 비율 평균 0.704, padding 0/157; 얼굴 외곽 여유가 작음 |
| **SQUARE_M10** | 얼굴 면적 비율 평균 0.488, padding 3/157, padding 비율 평균 0.00077·최대 0.0521; 얼굴·주변 여유의 균형 |
| SQUARE_M20 | 얼굴 면적 비율 평균 0.358, padding 6/157; 배경 증가와 얼굴 비중 감소 |

인접 frame의 얼굴 면적 비율 변화 중앙값은 SQUARE_0 0.0190, M10 0.0133, M20 0.0100이었습니다. 이는 실제 얼굴 움직임을 포함하는 보조 진단이며 margin이 클수록 좋다는 근거는 아닙니다. SQUARE_M10은 crop geometry·배경량·padding·시각적 일관성을 종합해 선택했으며, 모델 정확도 우위는 아직 검증하지 않았습니다.

### Manual Review Closure

STEP 2의 시각 검토는 review pack·contact sheet를 사용한 **사용자 직접 시각 검토와 단계별 전체 정책 결정**으로 완료됐습니다. 초기 per-sample CSV 입력 계획과 실제 검토 방식이 달라, 생성된 CSV의 수동 판단 필드는 채워지지 않았습니다. 수행하지 않은 표본별 `PASS`/`FAIL`·후보 선택을 사후 생성하지 않기 위해 원본 CSV는 그대로 보존합니다. 자동 report의 `WAITING_FOR_MANUAL_*`도 생성 당시 상태입니다. 자세한 범위·관찰·결정·CSV 채움 상태는 [manual review closure 기록](outputs/preprocessing_v2/manual_review_closure/step2_manual_review_closure.md)과 [STEP 2 상세 정책 문서](docs/experiment2_step2_preprocessing_policy.md)를 참고하세요.

| 단계 | Manual review 상태 | 전체 결정 | 원본 CSV 수동 판단 입력 |
|---|---|---|---:|
| STEP 2-A | STEP 2-B decision audit로 대체 | 없음 | 0/90행 |
| STEP 2-B | 단계 수준 시각 검토 완료 | YuNet primary | 0/126행 |
| STEP 2-C | 단계 수준 시각 검토 완료 | YuNet RAW bbox → Dlib68 | 0/35행 |
| STEP 2-D | 단계 수준 시각 검토 완료 | SQUARE_M10 | 0/42행 |

이 closure는 STEP 2 frozen policy를 바꾸지 않습니다. STEP 2 수동 검토에도 test sample은 사용하지 않았고 **TEST SPLIT SEALED**를 유지합니다. 이후 STEP 3-B pilot의 실행 상태는 아래에 별도로 기록합니다.

### STEP 3 — Canonical Dataset Preprocessing

STEP 2 frozen policy는 그대로 유지하며 [STEP 3 상세 설계 및 실행 기록](docs/experiment2_step3_canonical_preprocessing.md)에 schema·결측·provenance·resume 규칙과 실제 pilot/full 결과를 정리했습니다. STEP 3-A 구현, STEP 3-B의 20-video pilot·수동 검토에 이어 STEP 3-C의 전체 train/val materialization까지 완료했습니다.

#### STEP 3-A — Pipeline Implementation

Status: **IMPLEMENTED**

10 Hz × 10초 고정 100 slot, 검출 독립적인 32개 Context slot, 영상당 순차 decode, 동일 YuNet 검출의 Behavior/Context 공유, Dlib68 RAW bbox·EAR/MAR/Head Pose, SQUARE_M10 RGB crop을 구현했습니다. 실패 slot은 대체하지 않고 NaN 및 단계별 status로 기록합니다. 영상별 bundle은 임시 경로에서 검증한 뒤 발행하며 policy hash·모델 SHA256·source fingerprint가 같은 완료 bundle만 `--resume`에서 건너뜁니다. Test split은 코드 수준에서 제외합니다.

```powershell
.venv/Scripts/python.exe scripts/run_canonical_preprocessing.py --config configs/canonical_preprocessing.yaml --dry-run
.venv/Scripts/python.exe scripts/prepare_canonical_pilot_manifest.py --dry-run
```

Metadata-only dry-run 결과는 train 1,452개와 val 311개, 총 1,763개 영상·176,300개 예상 frame row·56,416개 예상 Context slot입니다. 이는 계획 수치이며 처리 완료 수치가 아닙니다. Test 311개는 **SEALED**입니다. Pilot 후보는 train/val × drowsy/not_drowsy 각 5개, 총 20개로 고정 선택합니다.

#### STEP 3-B — 20-Video Pilot

Status: **COMPLETE**. 20-video 자동 integrity 검사와 사용자 제공 수동 시각 검토가 완료됐다.

결정적 [20개 영상 manifest](data/metadata/experiment2_step3_pilot_manifest.csv)는 train/val × drowsy/not_drowsy 각 5개로 구성했고 test는 0개입니다. 별도 `data/interim/preprocessing_v2/canonical_pilot/`에서 실제 YuNet·Dlib68 전처리를 한 번 실행했습니다. 단계별 [pilot 보고서](outputs/preprocessing_v2/canonical_preprocessing/pilot/pilot_preprocessing_report.txt)와 [구현 시각 검토 pack](outputs/preprocessing_v2/canonical_preprocessing/pilot/visual_review/)을 남겼습니다.

| 항목 | 실제 pilot 결과 |
|---|---:|
| 완료 / 처리 실패 영상 | 20 / 0 |
| Canonical row / 정상 decode | 2,000 / 2,000 |
| YuNet 성공 / 미검출 | 1,916 / 84 |
| Dlib68 성공 / 검출 후 실패 | 1,916 / 0 |
| EAR·MAR 유효 / Pose 성공 | 각 1,916 / 1,916 |
| Context 선택 / crop 가용 / 결측 | 640 / 613 / 27 |
| 저장 JPEG 무결성 / NPZ·cross-artifact 오류 | 613/613 / 0 |
| Orphan 임시 bundle / test row | 0 / 0 |
| 동일 manifest `--resume` | 신규 처리 0, 건너뜀 20, 충돌 0, detector 추론 0 |

검출 누락과 Context crop 결측은 구조 오류와 분리한 data-quality 수치입니다. 구현 시각 검토 pack은 split×label을 균형 있게 포함한 12개 영상·36개 표본·6장 sheet입니다. 사용자가 기존 6장 sheet를 검토해 frame/person alignment, RGB/BGR, JPEG corruption, SQUARE_M10 구현에 이상이 없다고 확인했습니다. 이번 pilot에서 threshold·crop·detector 정책은 변경하지 않았습니다.

Context crop 결측 27건만 따로 확인할 수 있도록 [missing-only review pack](outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/)을 생성했습니다. 대상은 train/val에서 `context_selected=True`이고 `context_crop_available=False`인 row입니다. 자동 상태는 27건 모두 정상 decode 후 `YUNET_FACE_NOT_FOUND`와 `CONTEXT_NO_FACE`였고, 검출 성공인데 crop만 없는 사례는 0건입니다. 기존 자동 [분석 보고서](outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/missing_context_report.txt)는 생성 당시 상태로 보존했습니다.

사용자가 27개 이미지를 직접 판독한 결과를 [수동 판정 CSV](outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/missing_context_manual_review.csv)에 기록했습니다. `review_id` 12~23의 12건은 강한 측면/profile, frame edge 방향 근접, 일부 코 끝 clipping으로 미검출이 비교적 납득 가능한 `EXPECTED_DETECTOR_MISS/PASS`입니다. 나머지 15건에도 상당수 yaw/profile 난이도가 있지만 얼굴 주요 구조는 육안으로 충분히 보여 `LIKELY_FALSE_NEGATIVE/CHECK_NEEDED`를 유지합니다. 이 15건의 `extreme_pose` 빈 값은 개별 미평가이며 정면 판정을 뜻하지 않습니다. Pipeline/storage issue 의심은 0건입니다. [수동 검토 summary](outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/missing_context_manual_review_summary.json)와 [별도 보고서](outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/missing_context_manual_review_report.txt)에 해석 revision 2를 반영하고, [정정 provenance](outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/manual_review_interpretation_correction.md)에 수정 전후 CSV SHA-256·사용자 후속 판독 근거를 남겼습니다. 이는 Codex의 이미지 재판독이 아니며 YuNet 교체·fallback·threshold 변경·missing frame 대체를 정당화하지 않습니다. 전체 train/val 결측 분포는 STEP 4에서 정량화합니다.

#### STEP 3-C — Full Train/Val Materialization

Status: **COMPLETE**. [Full 산출물](data/interim/preprocessing_v2/canonical/)에 train 1,452개·val 311개, 총 1,763개 영상을 STEP 3-B와 같은 policy hash `29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721` 및 동일 YuNet/Dlib 모델 SHA-256으로 처리했습니다. Test 311개는 열거나 처리하지 않았고 full 출력의 test row는 0개입니다. Pilot bundle을 복사·재사용하지 않았으며 frozen YuNet / Dlib68 RAW bbox / SQUARE_M10 정책을 변경하지 않았습니다.

| 항목 | Full train/val 실제 결과 |
|---|---:|
| 완료 / 처리 실패 영상 | 1,763 / 0 |
| Canonical row / 정상 decode | 176,300 / 176,300 |
| YuNet 성공 / 미검출 | 170,871 / 5,429 (성공률 96.9206%) |
| YuNet 성공 frame에서 Dlib68 성공 / 실패 | 170,871 / 0 (조건부 성공 100%) |
| EAR·MAR·Head Pose 유효 | 각 170,871 |
| Context 선택 / 가용 / 결측 | 56,416 / 54,672 / 1,744 (가용률 96.9087%) |
| Padding 필요 crop / multiple-face frame | 885 / 0 |
| Strict bundle·JPEG / cross-artifact 오류 | 1,763 bundle·54,672 JPEG 검사 / 0 |
| Orphan 임시 bundle / test row | 0 / 0 |
| 동일 설정 `--resume` | 신규 0, skip 1,763, 충돌 0, detector 추론 0 |

Train의 YuNet 성공/미검출은 140,815/4,385, Context 가용/결측은 45,051/1,413이다. Val은 각각 30,056/1,044와 9,621/331이다. 가용 crop의 padding fraction은 평균 0.000964, 중앙값 0, 최대 0.291262였다. 최초 실행 wall time은 약 2.716시간, 영상별 처리 평균 5.344초·중앙값 5.170초였다. 앞선 2.47시간은 pilot 기반 **simple projected runtime**이지 보장 시간이 아니었다.

[Full 요약](outputs/preprocessing_v2/canonical_preprocessing/full/full_preprocessing_summary.json), [처리 보고서](outputs/preprocessing_v2/canonical_preprocessing/full/full_preprocessing_report.txt), [엄격 무결성 보고서](outputs/preprocessing_v2/canonical_preprocessing/full/full_integrity_report.txt), [실패 목록](outputs/preprocessing_v2/canonical_preprocessing/full/failed_videos.csv), [resume 검증](outputs/preprocessing_v2/canonical_preprocessing/full/full_resume_report.txt)에 결과를 분리했다. 원본 run metadata의 `dlib_version`은 배포판 metadata 조회 실패로 `null`이며 수정하지 않았다. 확인된 `dlib.__version__=20.0.1`은 [provenance addendum](outputs/preprocessing_v2/canonical_preprocessing/full/run_provenance_addendum.json)에 별도로 기록했다. YuNet miss와 Context 결측은 처리 실패가 아닌 data-quality 관찰이다. 이번 train/val 실행에서 YuNet 성공 frame의 Dlib68은 모두 성공했으므로 관찰된 얼굴 검출·landmark 결측의 시작점은 YuNet 미검출이었다. 이는 모든 환경에서의 Dlib 정확도를 뜻하거나 Context 결측의 시각적 원인을 확정하지 않는다. STEP 2 detector 및 결측 정책은 변경하지 않았다.

STEP 4-A에서 저장된 train/val canonical metadata만 자동 분석했다. Head Pose 물리적 부호 규약, EAR closure·PERCLOS·MAR/yawn·head-drop/nod 규칙과 결측 처리 정책은 아직 미확정이다. ResNet18/VGG16 CNN과 LSTM은 학습하지 않았으며, test 311개 영상은 열기·decode·YuNet/Dlib 추론·crop 생성 모두 하지 않았다. 최종 평가 전까지 **TEST SPLIT SEALED**를 유지한다.

### STEP 4 — Full Preprocessing Quality / Missing Audit

#### STEP 4-A — Automatic Audit

Status: **COMPLETE**. [Full canonical metadata](data/interim/preprocessing_v2/canonical/metadata/)의 train 1,452개·val 311개, 총 1,763개 영상·176,300행을 read-only로 분석했다. Test row는 0개이고 policy hash는 STEP 3-C와 일치한다. 원본 영상을 열거나 detector·landmark를 재실행하지 않았다.

| 항목 | 실제 자동 관찰 |
|---|---:|
| YuNet 미검출 / 전체 frame | 5,429 / 176,300 (3.0794%) |
| Behavior 유효 frame | 170,871 / 176,300 (96.9206%) |
| Context 결측 / 선택 slot | 1,744 / 56,416 (3.0913%) |
| YuNet 성공 후 Dlib68 추가 실패 | 0 / 170,871 |
| YuNet 결측 0개 / Context 결측 0개 영상 | 1,382 / 1,425 |
| YuNet 결측 run / 최장 run | 967개 / 100 frame |
| 수치·flag 이상 / STEP 4-B 후보 | 0건 / 중복 없는 36개 영상 |

Train과 val의 YuNet 결측률은 각각 3.0200%·3.3569%, drowsy와 not-drowsy는 각각 2.2413%·3.8233%였다. 이는 기술 통계이며 집단 간 차이의 원인을 확정하지 않는다. 영상별 결측은 중앙값 0, p90 9, p99 48, 최대 100 frame이었다. 상위 5%(89개) 영상에 전체 미검출의 62.7187%가 모였다. Context 결측이 0개인 영상은 1,425개이며, 모든 결측 허용 개수 0~32에 대한 누적 가용성 표를 남겼다. 이 숫자들로 허용 임계값이나 영상 제외 정책을 정하지 않았다.

[자동 audit 보고서](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/quality_missing_report.txt), [JSON 요약](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/quality_missing_summary.json), [영상별 품질표](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/per_video_quality.csv), [36개 수동 검토 후보](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_review_candidates.csv)에 상세 수치와 근거가 있다. STEP 3-B의 27건에서는 사용자가 yaw/profile 연관성을 시각 관찰했지만, 이번 전체 자동 audit는 원본 이미지를 보지 않았다. YuNet 미검출 frame에는 해당 frame의 pose가 없으므로 full 데이터셋의 pose 원인을 단정하지 않는다. 이웃 frame yaw는 별도 보조 힌트일 뿐이다. [STEP 4 상세 기록](docs/experiment2_step4_quality_missing_audit.md)에 정의·분포·한계를 정리했다.

#### STEP 4-B — Manual Visual Review Complete

STEP 4-A의 36개 후보를 재선정하지 않고 [시각 검토 팩](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/)을 생성했다. 선택된 원본 영상 36개에서 필요한 frame 140개만 시각화 목적으로 읽었으며, 개별 검토 이미지 36장과 contact sheet 9장을 만들었다. [검토 인덱스](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_review_index.csv)가 이미지와 후보를 연결한다. 사용자가 ChatGPT 보조로 contact sheet를 검토해 확정한 review_id별 결과를 [수동 판정 CSV](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review.csv) 36행에 기록했다. Codex가 이미지를 다시 판독하거나 category를 변경하지 않았다.

수동 판정 결과는 `EXPECTED_DETECTOR_MISS` 13개, `LIKELY_FALSE_NEGATIVE` 12개, `MIXED_MISSING_PATTERN` 5개, `NORMAL_REFERENCE` 6개다. `PASS` 19개, `CHECK_NEEDED` 17개, `INVESTIGATE` 및 pipeline/storage issue 의심 0개다. `n_246`은 100/100 결측·`FULL_CLIP_MISSING`·`EXPECTED_DETECTOR_MISS`로 기록됐다. 선택된 긴 run에서는 pose/head orientation 난이도뿐 아니라 얼굴이 충분히 보이는 연속 미검출도, 고립 miss에서는 blur로 설명되는 사례와 앞뒤 frame과 비슷한 false-negative 후보가 모두 보고됐다. 따라서 yaw/profile만을 전체 결측의 원인으로 단정하지 않는다. 이 36개는 목적 선정 후보이며 전체 1,763개 영상의 무작위 표본이 아니다.

[수동 결과 요약](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review_summary.json), [해석 보고서](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review_report.txt), [출처·전후 해시](outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review_provenance.json)에 세부 내용을 남겼다. STEP 4-B는 **COMPLETE**다. 자동 메타데이터와 저장된 YuNet bbox·Context JPEG는 변경하지 않았고 YuNet·Dlib68 재추론, crop 재생성, 결측 대체도 하지 않았다. Missing policy는 **NOT SELECTED**, STEP 4-C는 **NOT STARTED**, test split은 **SEALED**다.
