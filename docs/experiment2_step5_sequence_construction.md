# Experiment 2 STEP 5 — Sequence Construction

## 1. Frozen Inputs

STEP 5-A는 `data/interim/preprocessing_v2/canonical/`의 train/validation metadata와 기존 Context JPEG를 읽기 전용으로 사용한다. Canonical policy hash는 `29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721`이다. Test split은 열거나 경로를 해석하지 않았다.

## 2. Sequence Missing Policy

동결 설정은 `configs/sequence_missing_policy.yaml`이며 hash는 `f382c7900331c0be2f56b95f8fd95ed77f86dd5d4b06239da078e03d180eee4d`이다. Context 정책 `CONTEXT_C3_SHORT_GAP`은 missing count ≤ 8, longest missing run ≤ 4인 영상을 허용한다. Behavior 정책 `BEHAVIOR_B2_COVERAGE_95`는 valid count ≥ 95, longest YuNet missing run ≤ 5를 요구하고 NaN과 mask를 유지하며 보간하지 않는다.

## 3. STEP 5-A Context Sequence

Train/validation 1,763개 중 Context 적격 1,677개를 32-slot sequence로 materialize했다. 86개 부적격 영상은 bundle 없이 제외 manifest에 기록했다. 결과는 `data/interim/sequences_v2/context/`, 보고서는 `outputs/sequences_v2/context_sequence/`에 있다.

## 4. Target vs Source Semantics

Target index와 timestamp는 원래 시간 위치를 유지한다. 원본 crop이 있으면 source는 target 자신이며, missing target은 같은 영상의 다른 원본-valid JPEG를 가리킨다. 따라서 대체 후에도 시간 위치와 이미지 출처가 분리되어 보존된다.

## 5. Nearest-Valid Substitution

Source는 context index 거리가 가장 가까운 원본-valid slot이다. 같은 거리이면 earlier slot을 택한다. 이미 대체된 target은 source가 될 수 없어 chained imputation은 0건이다. 영상 간 대체는 허용하지 않는다.

## 6. Masks and Provenance

각 bundle은 32행 `sequence.csv`, bool `original_context_valid_mask`와 `context_imputed_mask` 및 source/target index·거리 배열을 담은 `masks.npz`, `summary.json`, 해시가 포함된 `COMPLETE.json`으로 구성된다. JPEG는 bundle에 복사하지 않고 project-relative path로만 참조한다.

## 7. Eligibility

Context eligibility는 Behavior eligibility와 독립적이다. Behavior 적격 1,520개뿐 아니라 Context-only 157개도 포함하여 총 1,677개를 생성했다. `n_246`은 Context 32/32 missing으로 제외 manifest에만 포함된다.

## 8. Integrity Validation

각 영상의 target 0..31, mask 상보성, 원본-valid source, nearest/tie 규칙, source path 존재, bundle 파일 hash를 확인했다. 전체 53,664개 source path가 존재했고 결정적 표본 256개 JPEG가 224×224×3으로 decode되었다. Atomic publish와 동일 policy/fingerprint의 resume 검증을 사용한다.

## 9. Current Results

| 항목 | 결과 |
|---|---:|
| 입력 train/val 영상 | 1,763 |
| Context 적격 / 제외 | 1,677 / 86 |
| Sequence 행 | 53,664 |
| 원본-valid / 대체 참조 | 52,965 / 699 |
| 대체 불필요 / 필요 영상 | 1,425 / 252 |
| Past / future source | 465 / 234 |
| Chained / nearest mismatch / missing source | 0 / 0 / 0 |
| 새로 생성·복사한 JPEG | 0 |

Context-slot 대체 거리는 min 1, mean 1.2375, median 1, p90/p95 2, max 4였다. Canonical-slot 거리는 min 3, mean 3.9928, median 3, p90 6, p95 7, max 13이었고 시간 거리는 min 0.2682초, mean 0.3993초, median 0.3005초, p90 0.6034초, p95 0.7001초, max 1.3초였다.

영상별 대체 slot 분포는 `0:1425, 1:74, 2:69, 3:40, 4:25, 5:16, 6:13, 7:11, 8:4`이다. Train/validation은 1,382/295개, drowsy/not_drowsy는 800/877개다. Mask shape `(32,)`, dtype bool, per-video/global integrity, 1,677-bundle resume skip가 모두 통과했다. Canonical 변경과 test 접근은 0건이다.

STEP 5-A 완료 시 targeted test 25개와 당시 전체 test 134개 및 subtest 5개가 통과했고 `git diff --check`도 통과했다.

## 10. STEP 5-B Behavior 100-Slot Sequence Construction

Train/validation 1,763개 중 B2 적격 1,520개를 각각 100-slot로 materialize해 총 152,000행을 생성했다. 243개 부적격 영상은 bundle 없이 제외 manifest에 기록했다. `n_246`은 100/100 missing, `NO_VALID_BEHAVIOR`로 제외했다. Context eligibility는 filter로 사용하지 않았으며 Context-only 157개에는 Behavior bundle을 만들지 않았다.

Feature 순서는 `ear`, `mar`, `pitch_raw`, `pitch_centered_candidate`, `yaw`, `roll`이고 영상별 tensor shape는 `(100, 6)`, dtype은 float32다. `feature_valid_mask`는 feature별 finite 여부이며 `detector_valid_mask`, `landmark_valid_mask`, `pose_valid_mask`, frozen semantics의 `behavior_valid_mask`를 bool로 별도 보존한다. Canonical index와 timestamp도 함께 저장한다.

결측 slot을 삭제하거나 채우지 않는다. Canonical NaN과 부분 유효값을 그대로 보존하며 nearest/forward/backward/linear interpolation, zero/mean fill, smoothing, normalization, threshold 및 event 생성을 수행하지 않았다. Valid count와 frozen manifest 불일치 0, longest-run 불일치 0, required-valid NaN contradiction 0, infinity 0, canonical numeric mismatch 0이다.

Valid-count 분포는 `95:17, 96:21, 97:21, 98:34, 99:45, 100:1382`, longest YuNet missing-run 분포는 `0:1382, 1:63, 2:31, 3:19, 4:16, 5:9`이다. Train/validation은 1,256/264개, drowsy/not_drowsy는 736/784개다. 결과는 `data/interim/sequences_v2/behavior/`, 보고서는 `outputs/sequences_v2/behavior_sequence/`에 있으며 atomic bundle과 policy/source/schema hash 기반 resume을 사용한다.

Head Pose physical sign convention remains unresolved. Pitch continuous values are preserved without semantic head-down/head-up interpretation. `pitch_raw`와 `pitch_centered_candidate`는 진단용 연속값으로 함께 보존한다.

## 11. Limitations

이 단계는 sequence packaging의 구조·출처 무결성을 검증한다. Context 대체나 B2 mask가 모델 성능을 향상한다는 주장이 아니며, subject-wise 독립성도 보장하지 않는다. Behavior threshold/event와 CNN feature는 생성하지 않았다.

## 12. STEP 5 Roadmap and Next Steps

- STEP 5-A — Context 32-Slot Sequence Construction: **COMPLETE**
- STEP 5-B — Behavior 100-Slot Sequence Construction: **COMPLETE**
- STEP 5-C — Cross-Branch Sequence Integrity Audit: **NOT STARTED**
- STEP 5-D — CNN Feature Extraction: **NOT STARTED**

다음 단계는 실제 Context/Behavior sequence artifact 기준의 STEP 5-C다. 1,520 both, 157 Context-only, 0 Behavior-only, 86 neither 관계는 이 다음 단계에서 재검증하며 이번 단계에서는 cross-branch artifact를 만들지 않았다. ResNet18/VGG16 및 ImageNet normalization은 STEP 5-D 전까지 실행하지 않는다.
