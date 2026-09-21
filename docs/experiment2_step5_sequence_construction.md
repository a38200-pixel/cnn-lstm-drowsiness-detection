# Experiment 2 STEP 5 — Sequence Construction

## 1. Frozen Inputs

STEP 5-A는 `data/interim/preprocessing_v2/canonical/`의 train/validation metadata와 기존 Context JPEG를 읽기 전용으로 사용한다. Canonical policy hash는 `29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721`이다. Test split은 열거나 경로를 해석하지 않았다.

## 2. Sequence Missing Policy

동결 설정은 `configs/sequence_missing_policy.yaml`이며 hash는 `f382c7900331c0be2f56b95f8fd95ed77f86dd5d4b06239da078e03d180eee4d`이다. Context 정책 `CONTEXT_C3_SHORT_GAP`은 missing count ≤ 8, longest missing run ≤ 4인 영상을 허용한다.

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

## 10. Limitations

이 단계는 sequence mapping의 구조·출처 무결성을 검증한다. 대체 참조가 모델 성능을 향상한다는 주장이 아니며, subject-wise 독립성도 보장하지 않는다. Behavior sequence 및 모델 feature는 생성하지 않았다.

## 11. Next Steps

STEP 5-B에서 이 참조와 mask를 loader 입력으로 사용해 Context CNN feature extraction을 별도로 구현·검증한다. ResNet18/VGG16 비교, ImageNet normalization, LSTM 학습은 STEP 5-A 범위 밖이며 현재 시작하지 않았다.
