# Experiment 2 STEP 2 — Preprocessing Policy Selection

## 1. Objective and evidence boundary

SUST-DDD Experiment 2의 STEP 2는 전체 데이터 전처리·학습 전에 face detector, Dlib68 fitting ROI, Context CNN crop geometry를 소규모 train/validation 표본으로 결정하는 단계다. STEP 1의 frozen video-level split을 유지했고, STEP 2-A~D에서 **test split은 사용하지 않았다**. 실험 입력은 동일한 160개 frame(32개 video, train 80·validation 80)이며, YuNet 검출 성공 frame을 사용하는 STEP 2-C/D는 157개(train 80·validation 77)다.

이 문서의 자동 수치는 아래 원본 artifact에서 확인한 값이다. 시각적 관찰과 최종 정책 선택은 이번 문서화 요청에서 사용자가 제공한 수동 검토 결론을 근거로 한다. 원본 자동 report의 `WAITING_FOR_MANUAL_*` Decision은 생성 당시 상태이며 지우거나 소급 수정하지 않는다. 수동 검토 CSV의 판정 칸은 아직 비어 있으므로, 이를 작성 완료된 증빙이라고 주장하지 않는다. 이번 작업에서는 실험·데이터 처리·학습을 다시 실행하지 않았다.

## 2. STEP 2-A — Detector / Landmark Compatibility Audit

Experiment 1은 `HOG → Dlib68`과 `YuNet → 3DDFA`로 landmark source가 달랐고 EAR/MAR scale 및 calibration 신뢰성 문제가 있었다. Experiment 2는 HOG와 YuNet 모두에 Dlib68을 적용해 이 제약을 그대로 승계하지 않고 호환성을 다시 확인했다. STEP 2-A 후보는 HOG primary, 실패 시 YuNet fallback이었다.

| 항목 | 실제 결과 |
|---|---:|
| Audit frame | 160 |
| HOG 검출 | 152/160 (95.0%) |
| HOG 실패 / YuNet 복구 | 8 / 6/8 |
| 결합 검출 | 158/160 (98.75%) |
| HOG bbox → Dlib68 / fallback YuNet bbox → Dlib68 | 152/152 / 6/6 |
| 평균 detector latency | HOG 371.82 ms / YuNet fallback 39.48 ms |

Raw Euler pitch 중앙값은 약 159.40°였고, paired raw pitch의 단순 절댓값 차이는 최대 약 357.44°였다. ±180° wrap-around를 무시한 표현·비교 convention 문제로 해석하며 Head Pose 모델 실패로 단정하지 않는다. 이 단계의 자동 Decision은 `WAITING_FOR_MANUAL_VISUAL_REVIEW`였고 primary detector는 STEP 2-B에서 결정했다.

## 3. STEP 2-B — HOG vs YuNet Primary Detector Comparison

동일한 160개 frame에서 `HOG → Dlib68`과 `YuNet → Dlib68`을 비교해 landmark predictor와 EAR/MAR·Head Pose 계산을 통제했다. 새 표본 추출이나 test 사용은 없었다.

| 항목 | HOG | YuNet |
|---|---:|---:|
| 검출 성공 | 152/160 (95.0%) | 157/160 (98.125%) |
| 평균 detector latency | 372.15 ms | 39.44 ms |
| 평균 Behavior pipeline latency | 376.17 ms | 43.78 ms |
| 검출 후 Dlib68 성공 | 152/152 | 157/157 |

검출 교집합은 both success 151, HOG only 1, YuNet only 6, both failed 2였다. 검토 pack은 선택된 이미지 45개와 sheet 5장이다. 사용자 제공 시각 검토 결론에서는 YuNet bbox가 이마부터 턱까지 얼굴을 더 자연스럽고 타이트하게 포함한 사례가 많았고, HOG는 좌우 배경 padding이 상대적으로 큰 사례가 관찰됐다. 따라서 HOG bbox를 landmark geometry의 ground truth로 간주하지 않는다.

Paired EAR Pearson은 약 0.5993, MAR Pearson은 약 0.2723이었다. 같은 Dlib68을 사용해도 입력 rectangle이 달라지면 fitting 결과가 변할 수 있다는 진단 결과다. 이 상관 차이를 YuNet 검출 오류로 단정하지 않고, detector 선택과 Dlib fitting ROI 선택을 분리해 STEP 2-C에서 검토했다.

Head Pose 비교에는 `abs(((a - b + 180) % 360) - 180)` 형태의 circular angular distance를 사용한다. Raw angle은 보존하고, `pitch_centered_candidate = wrap_to_180(pitch_raw - 180)`은 정면≈0 표현을 살피기 위한 분석 후보로만 둔다. 물리적 head-up/head-down의 최종 부호 매핑은 미확정이다.

**결론: Experiment 2 primary face detector는 YuNet.** 이번 통제된 표본에서 검출 성공률, 속도, Dlib68 호환성 및 시각적 bbox 품질을 종합한 선택이며 모든 조건에서 HOG보다 정확하다는 일반화는 아니다. 원본 report의 수동 검토 대기 Decision은 생성 당시 상태로 보존한다.

## 4. STEP 2-C — YuNet Landmark ROI Margin Audit

STEP 2-B에 저장된 YuNet bbox를 재사용해 detector를 다시 실행하지 않았다. YuNet 검출 성공 157개 frame에서 Dlib68 입력 rectangle만 RAW(0%), M05(양쪽 5%), M10(양쪽 10%), M15(양쪽 15%)로 바꿨다. 이 ROI는 Context CNN crop과 별개다.

| 후보 | Dlib68·geometry·EAR·MAR·pose 성공 | RAW 대비 normalized landmark difference 평균 | EAR 평균 | MAR 평균 |
|---|---:|---:|---:|---:|
| RAW | 각 157/157 | 기준 | 0.369 | 0.148 |
| M05 | 각 157/157 | 0.0278 | 0.406 | 0.194 |
| M10 | 각 157/157 | 0.0560 | 0.455 | 0.231 |
| M15 | 각 157/157 | 0.1059 | 0.470 | 0.272 |

Margin 추가로 성공률이 개선되지는 않았다. Margin 증가에 따라 같은 frame의 landmark와 EAR/MAR 분포가 체계적으로 이동했으며, 이는 feature 품질 향상이 아니라 Dlib68 fitting의 ROI geometry 민감성으로 해석한다. 검토 pack은 35개 5-panel 표본과 compact sheet 5장(기존 contact sheet 18장)이다. 사용자 제공 시각 검토에서는 RAW가 눈·입·턱을 대체로 잘 따라갔고, M05는 일관된 개선이 없었으며, M10/M15 일부 표본에서 contour·mouth·pose 변화가 커졌다.

**결론: LANDMARK ROI = YUNET RAW BBOX; RAW bbox → Dlib68.**

## 5. Clipping Metadata Correction

원래 `roi_clipped` 수는 RAW 0/157, M05 157/157, M10 155/157, M15 157/157이었으나 확장 후보의 `clipped_fraction`은 모두 0이었다. 원인은 margin 적용 후 실수 좌표와 `floor/ceil` 정수 ROI를 직접 비교해 정수화 차이를 frame clipping으로 오판한 flag 계산이다. Dlib inclusive endpoint 변환은 원인이 아니다.

같은 반개방 정수 좌표계에서 `intended_roi`와 frame-clipped ROI를 비교한 보정 결과는 RAW·M05·M10·M15 모두 **0/157 clipped**다. 저장된 ROI 좌표 **628/628**개가 보정 계산값과 일치했다. 영향은 `REPORTING_ONLY_BUG`이며 실제 Dlib68 입력, EAR/MAR, Head Pose, visual review는 바뀌지 않았다. **`NO ROI AUDIT RERUN REQUIRED`**. 원본 report·summary·CSV는 보존하고 보정 artifact를 별도 디렉터리에 남겼다.

## 6. STEP 2-D — Context CNN Crop Policy Audit

YuNet bbox를 고정하고 Behavior branch는 RAW bbox → Dlib68을 유지했다. Context branch용 RGB 224×224 Full Face crop만 `RAW_RESIZE`, `SQUARE_0`, `SQUARE_M10`, `SQUARE_M20`으로 비교했다. Detector·landmark 재실행 없이 YuNet 성공 157개 frame에서 audit preview를 생성했다.

- `RAW_RESIZE`: YuNet rectangle을 바로 224×224로 resize.
- `SQUARE_0`: 중심을 유지하고 긴 축을 줄이지 않는 square.
- `SQUARE_M10`/`SQUARE_M20`: bbox width·height 각각 양쪽으로 10%/20% 확장한 뒤 중심 유지 square.
- Square ROI가 frame 밖이면 ROI를 축소하지 않고 ImageNet mean RGB `[124, 116, 104]`(OpenCV BGR `[104, 116, 124]`)로 padding. 양 축 축소 시 `INTER_AREA`, 확대가 포함되면 `INTER_LINEAR`.

| 후보 | 자동 진단 | 사용자 제공 시각 검토 결론 |
|---|---|---|
| RAW_RESIZE | distortion ratio 중앙값 1.4258 | 얼굴 비율 왜곡 반복 |
| SQUARE_0 | 얼굴 면적 비율 평균 0.7038·중앙값 0.7014; padding 0/157 | 얼굴 크기는 유지되나 이마·턱·외곽 여유 부족 사례 |
| **SQUARE_M10** | 얼굴 면적 비율 평균 0.4877·중앙값 0.4850; padding 3/157; padding 비율 평균 0.000769·최대 0.0521 | 얼굴 전체와 적정 여유를 유지하고 배경 증가가 과도하지 않음 |
| SQUARE_M20 | 얼굴 면적 비율 평균 0.3584·중앙값 0.3568; padding 6/157; padding 비율 평균 0.002388·최대 0.1159 | 배경이 많아지고 얼굴 비중 감소 |

같은 video의 인접 audit frame 간 얼굴 면적 비율 변화 절댓값 중앙값은 SQUARE_0 0.0190, M10 0.0133, M20 0.0100(각 125쌍)이었다. 이는 실제 얼굴 움직임도 포함한 보조 진단으로, margin이 클수록 무조건 좋다는 뜻이 아니다. 검토 대상은 42개 이미지와 11개 contact sheet였다.

**결론: CONTEXT CROP = SQUARE_M10.** Aspect-ratio 보존, 얼굴 전체·움직임의 공간 여유, 배경량, padding 빈도 및 시각적 일관성을 종합한 preprocessing 선택이다. CNN은 아직 학습하지 않았으므로 최고 정확도나 representation 우위를 주장하지 않는다. 원본 자동 report의 `WAITING_FOR_MANUAL_CONTEXT_CROP_REVIEW`는 생성 당시 상태로 보존한다.

## 7. Final Frozen Policy

| 항목 | Experiment 2 STEP 2 확정값 |
|---|---|
| Primary face detector | **YuNet** |
| Landmark predictor | **Dlib68** |
| Landmark fitting ROI | **YuNet RAW bbox** (margin 없음) |
| Behavior features | EAR / MAR / Head Pose |
| Head Pose detector 간 비교 | Circular angular distance, raw angle 보존 |
| Context crop | **SQUARE_M10** |
| Context crop 절차 | YuNet bbox → width·height 각각 양쪽 10% margin → 중심 유지 square → frame 밖 ImageNet-mean padding → RGB 224×224 |
| Context padding RGB / OpenCV BGR | `[124, 116, 104]` / `[104, 116, 124]` |
| Context model 계획 | ResNet18과 VGG16 별도 실험; 현재 미구현·미학습 |

```text
Frame → YuNet → YuNet RAW bbox
                   ├─ Behavior: RAW bbox → Dlib68 → EAR / MAR / Head Pose
                   └─ Context: 양쪽 10% margin → 중심 유지 square
                               → ImageNet-mean padding → RGB 224×224
                               → ResNet18 또는 VGG16 (향후 단계)
```

## 8. Rejected / Not Selected Alternatives

| 후보 | 선택하지 않은 이유 |
|---|---|
| HOG primary | 이번 표본에서 검출 성공률이 낮고 latency가 높았으며 bbox 좌우 배경 padding 사례가 관찰됨 |
| YuNet bbox + 5/10/15% Dlib ROI | landmark 성공률 개선 없이 fitting·EAR/MAR·pose 변화가 증가함 |
| RAW_RESIZE Context crop | 가로세로 비율 왜곡 |
| SQUARE_0 | 사용 가능한 후보이나 M10보다 얼굴 외곽 spatial buffer가 작음 |
| SQUARE_M20 | M10보다 배경 증가·얼굴 비중 감소·padding 빈도 증가 |

이는 해당 후보들이 보편적으로 실패한다는 판정이 아니라 이번 Experiment 2의 통제된 audit에 따른 선택이다.

## 9. Remaining Open Questions and STEP 3

- Head Pose `pitch_centered_candidate`의 양·음과 실제 head-up/head-down 대응.
- EAR closure, PERCLOS, MAR/yawn, head-drop/nod threshold 및 시간 규칙.
- Canonical frame sampling, sequence 생성, missing detection 처리, CNN·LSTM 구현과 학습, ResNet18/VGG16 비교. **STEP 3 — Canonical Dataset Preprocessing: NOT STARTED.**
- 최종 평가까지 **TEST SPLIT SEALED**. Detector·crop·preprocessing 정책, threshold·rule calibration, model 선택에 test를 사용하지 않는다.

이 미확정 항목은 STEP 2의 preprocessing 정책 확정과 별개다. STEP 2 상태는 **COMPLETE**다.

## 10. Artifact Locations

| 단계 | 원본 artifact |
|---|---|
| STEP 1 | `outputs/preprocessing_v2/source_validation/` (`source_validation_summary.json`, `split_integrity.json`) |
| STEP 2-A | `outputs/preprocessing_v2/detector_landmark_audit/` (`audit_summary.json`, `audit_report.txt`) |
| STEP 2-B | `outputs/preprocessing_v2/detector_primary_comparison/` (`comparison_report.txt`, `comparison_summary.json`, `paired_primary_results.csv`, `review_pack/`) |
| STEP 2-C | `outputs/preprocessing_v2/landmark_roi_margin_audit/` (`roi_margin_report.txt`, `roi_margin_summary.json`, `roi_margin_results.csv`, `review_pack/`) |
| Clipping correction | `outputs/preprocessing_v2/landmark_roi_margin_audit/clipping_metadata_correction/` |
| STEP 2-D | `outputs/preprocessing_v2/context_crop_audit/` (`context_crop_report.txt`, `context_crop_summary.json`, `context_crop_results.csv`, `contact_sheets/`) |

## 11. Limitations

- 표본은 train/validation의 160개 frame이며, YuNet 성공 조건의 ROI/crop 비교는 157개 frame에 국한된다. 이 결과를 모든 운전자·조명·자세 조건으로 일반화하지 않는다.
- Frozen split은 **video-level stratified split**이다. 신뢰할 만한 per-video subject mapping이 없어 subject-wise independence를 보장할 수 없고 unseen-subject generalization을 주장하지 않는다.
- 시각 검토 결론은 사용자 제공 내용이며 manual review CSV 판정 칸은 미입력이다. 원본 자동 report의 대기 상태도 그대로 남아 있다.
- 행동 threshold, temporal rule, downstream 모델 정확도는 STEP 2에서 검증하거나 확정하지 않았다.
