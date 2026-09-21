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

## 11. STEP 5-C Cross-Branch Sequence Integrity Audit

Canonical train/validation 1,763개, frozen eligibility, Context 1,677개 bundle, Behavior 1,520개 bundle을 실제 artifact에서 전수 교차 검증했다. 실제 집합은 BOTH 1,520, CONTEXT_ONLY 157, BEHAVIOR_ONLY 0, NEITHER 86이며 frozen eligibility와 실제 bundle mismatch는 0이다. Split/label mismatch도 0이다.

Context 53,664행은 영상별 32 target, original 52,965행, imputed reference 699행으로 재검증됐다. Source JPEG 결측, chained imputation, mask 상보성 위반, target timestamp overwrite는 모두 0이다. 모든 영상은 동일한 target canonical index pattern `0, 3, 6, 10, 13, 16, 19, 22, 26, 29, 32, 35, 38, 42, 45, 48, 51, 54, 57, 61, 64, 67, 70, 73, 77, 80, 83, 86, 89, 93, 96, 99`를 사용한다. Context target timeline and image-source provenance are intentionally separated.

Behavior 152,000행은 영상별 `(100, 6)` float32와 bool mask로 재검증됐다. 실제 mask 합은 valid 151,655, missing 345이며 valid-count/run mismatch, required-valid NaN contradiction, feature-valid mask mismatch, infinity, canonical numeric mismatch는 모두 0이다. `behavior_valid_mask`는 frozen B2의 YuNet·landmark·EAR/MAR 기준이며 pose mask와 동일하다고 가정하지 않는다.

Context-only는 train/val 126/31, drowsy/not_drowsy 64/93이고 neither는 train/val 70/16, drowsy/not_drowsy 29/57이다. `n_246`은 NEITHER이며 `NO_VALID_CONTEXT`와 `NO_VALID_BEHAVIOR`를 유지한다. 실행 전후 canonical, frozen, Context, Behavior 및 각 report tree의 fingerprint가 동일했다. Test 접근·행은 0이고 audit anomaly도 0이다. 결과는 `outputs/sequences_v2/cross_branch_audit/`에만 생성했다.

Head Pose의 `pitch_raw`와 `pitch_centered_candidate`는 저장된 연속값일 뿐이며 physical sign은 **UNRESOLVED**, head-down/head-up rule은 **NOT DEFINED** 상태다. CNN·ImageNet normalization·feature extraction·LSTM은 실행하지 않았다.

## 12. STEP 5-D CNN Feature Extraction

### STEP 5-D1 Preflight & Pilot

Context sequence loader와 16-video controlled pilot 경로를 구현했다. Pilot manifest는 `data/metadata/experiment2_step5d_cnn_feature_pilot_manifest.csv`, preflight report는 `outputs/features_v2/context_cnn/pilot/`에 있다. Context eligible 1,677개에서 deterministic하게 train/val 12/4, drowsy/not_drowsy 9/7, no-imputation/imputation-required 8/8을 선택했다. 각 backbone의 target은 16×32=512행이고 실제 unique source JPEG는 454개다. Behavior eligibility는 selection에 사용하지 않았다.

입력 transform은 canonical RGB uint8 224×224를 geometry 변경 없이 CHW float32, `/255`, ImageNet mean `[0.485, 0.456, 0.406]`과 std `[0.229, 0.224, 0.225]`로 normalize한다. Resize, center/random crop, augmentation, AMP, label-dependent transform은 없다.

ResNet18은 `torchvision.models.resnet18`의 `DEFAULT` pretrained ImageNet weight를 요구하고 `fc`를 제거한 뒤 native global average pooling 결과를 flatten하여 512D/frame으로 만든다. VGG16은 `torchvision.models.vgg16`의 `DEFAULT` pretrained weight를 요구하고 `model.features` 뒤 `AdaptiveAvgPool2d((1,1))`를 적용해 512D/frame으로 만든다. VGG classifier의 25,088D flatten, 4,096D FC, 1,000-class logits는 사용하지 않는다. 두 backbone은 alternative experiment이며 concat/average/ensemble/fusion하지 않는다.

Source JPEG path를 backbone별로 deduplicate하여 같은 source는 한 번만 inference하고 target 위치에서 같은 feature를 재사용하도록 구현했다. Original/imputed mask, target/source index와 source mapping fingerprint를 보존한다. Backbone마다 별도 semantic policy hash와 artifact root를 사용하며 device와 batch size는 hash에 포함하지 않는다. Random initialization fallback은 금지된다.

#### Environment

Dependency blocker 해결 후 실제 환경은 Python 3.12.14, torch 2.11.0+cu130, torchvision 0.26.0+cu130, torch CUDA runtime 13.0, cuDNN 91900이었다. CUDA device 1개와 NVIDIA GeForce RTX 3080을 확인했고 float32 inference가 정상 동작했다. Random-weight fallback은 없었다.

#### Controlled Pilot Design

Deterministic pilot 16개는 `d_10, n_1, d_103, n_1002, d_100, d_101, d_102, d_104, d_266, n_265, d_843, n_311, n_320, n_321, n_861, d_344`다. Train/val 12/4, drowsy/not_drowsy 9/7, no-imputation/imputation-required 8/8이고 test는 0이다. 16×32=512 target 중 실제 unique source JPEG는 454개이며, 58개 target은 기존 source feature를 재사용한다.

#### Shared Inference Settings

두 backbone 모두 CUDA, batch size 16, float32, AMP off, `model.eval()`, gradient disabled, `torch.inference_mode()`를 사용했다. Batch 16 통일은 feature 의미를 같게 만들기 위한 필수조건이 아니라 controlled runtime·VRAM·pipeline 비교에서 run-condition 차이를 줄이기 위한 것이다. 향후 LSTM training batch size와는 별개다. 입력은 RGB uint8 224×224 → float32 `/255` → ImageNet normalization이며 resize, center/random crop, augmentation은 없다.

실제 명령은 `python scripts/extract_context_cnn_features.py --mode pilot --device cuda --resnet-batch-size 16 --vgg-batch-size 16`이었다. Torchvision 표준 pretrained checkpoint `resnet18-f37072fd.pth`와 `vgg16-397923af.pth`를 사용했으며 resolved enum은 각각 `ResNet18_Weights.IMAGENET1K_V1`, `VGG16_Weights.IMAGENET1K_V1`이다.

#### ResNet18 Results

`ResNet18_Weights.IMAGENET1K_V1`을 사용하고 `fc`를 제거한 native GAP feature를 저장했다. Feature policy hash는 `0b8d72abdc9f3fbdb44934100c507ebc249ea8225d01fe1e4ca0de2236c4cd3e`이다. 16개 영상 모두 `[32,512]` float32이며 target 512행, unique source 454개다. Feature extraction은 1.5796초, model load를 포함한 total은 2.5976초, 처리량은 287.41 source images/s, peak GPU memory는 159,284,224 bytes였다.

#### VGG16 Results

`VGG16_Weights.IMAGENET1K_V1`을 사용하고 classifier 없이 `model.features → AdaptiveAvgPool2d((1,1)) → flatten` feature를 저장했다. Feature policy hash는 `1edc39db86531ea3240f767722236ab45ee89d607811770308c30b526f0b2bc0`이다. 16개 영상 모두 `[32,512]` float32이며 target 512행, unique source 454개다. Feature extraction은 1.8997초, model load를 포함한 total은 9.6234초, 처리량은 238.99 source images/s, peak GPU memory는 686,154,240 bytes였다.

#### Feature Integrity and Imputed Reuse

각 backbone에서 finite element는 262,144개이고 NaN, Inf, all-zero vector, imputed feature mismatch, wrong source mapping은 모두 0이었다. STEP 5-A target/source provenance와 실제 CNN feature reuse가 일치했다. ResNet18과 VGG16 artifact는 서로 독립적이며 backbone fusion은 없다.

#### Interpretation Limitations and Decision

Pilot에서 ResNet18이 더 짧은 extraction time과 더 낮은 peak GPU memory를 기록했지만 이는 pilot runtime observation일 뿐 정확도·feature quality·최종 backbone 우위를 의미하지 않는다. Backbone별 feature mean과 L2 norm 크기는 서로 다른 representation이므로 직접적인 품질 비교 지표가 아니다. 최종 모델 선택은 동일한 LSTM/classifier training protocol의 결과를 함께 검토해야 한다.

**STEP 5-D1: COMPLETE.** Environment, pretrained weights, `[32,512]` shape, numeric integrity, target/source reuse와 test 보호가 모두 통과했다. STEP 5-D2 full extraction은 **NOT STARTED**다.

### STEP 5-D2 Full Extraction

향후 dependency와 pretrained weight load가 검증된 뒤 Context eligible 1,677개 전체를 대상으로 실행한다. ResNet18과 VGG16은 독립 artifact를 만들며 1,520개 BOTH set으로 축소하지 않는다. 현재 full extraction은 실행하지 않았다.

### Context Sequence Classifier Baseline

STEP 5-D feature extraction 이후의 초기 Context baseline은 backbone별 `[B,32,512]` feature를 동일한 temporal/classifier 구조에 입력한다. LSTM은 `input_size=512`, `hidden_size=128`, `num_layers=1`, `batch_first=true`, `bidirectional=false`, internal dropout `0.0`이다. 마지막 hidden state `[B,128]`에 `Linear(128→64) → ReLU → Linear(64→2)`를 적용하며 classifier dropout도 `0.0`이다. `CrossEntropyLoss`에는 raw logits를 전달하므로 model 내부에 필수 softmax layer를 넣지 않는다.

Experiment 2의 초기 Context baseline은 1-layer unidirectional LSTM(hidden=128)과 128→64→2 classifier를 사용하며, LSTM 내부 및 classifier Dropout은 모두 0.0으로 시작한다. Dropout은 baseline 학습 결과에서 과적합 또는 일반화 문제가 확인될 경우 별도 ablation 대상으로 검토한다. 이는 base paper의 완전 재현을 의미하지 않으며, Dropout 0.3의 성능 이점을 가정하지 않는다. 향후 후보 0.1/0.3/0.5와 실험 범위는 아직 동결하지 않았다.

첫 ResNet18/VGG16 비교에서는 frozen split, target timestamps, missing policy, LSTM/classifier, dropout 0.0, loss, training protocol과 checkpoint criterion을 동일하게 유지하고 backbone만 바꾼다. 이 결정은 CNN→512D feature를 만드는 STEP 5-D 자체의 semantic config에는 영향을 주지 않는다. 기존 model YAML은 초기 scaffold이며 이번 documentation-only 작업에서는 수정하지 않았고, 실제 LSTM training 구현 전에 baseline과 일치하도록 별도 정정·검증해야 한다.

## 13. Limitations

이 단계는 sequence packaging의 구조·출처 무결성을 검증한다. Context 대체나 B2 mask가 모델 성능을 향상한다는 주장이 아니며, subject-wise 독립성도 보장하지 않는다. Behavior threshold/event와 CNN feature는 생성하지 않았다.

## 14. STEP 5 Roadmap and Next Steps

- STEP 5-A — Context 32-Slot Sequence Construction: **COMPLETE**
- STEP 5-B — Behavior 100-Slot Sequence Construction: **COMPLETE**
- STEP 5-C — Cross-Branch Sequence Integrity Audit: **COMPLETE**
- STEP 5-D1 — CNN Feature Extraction Preflight & Controlled Pilot: **COMPLETE**
- STEP 5-D2 — Full Context CNN Feature Extraction: **NOT STARTED**

다음 단계는 Context eligible 1,677개 전체에 대해 두 backbone을 독립적으로 materialize하는 STEP 5-D2다. Behavior eligible 1,520개로 대상을 축소하지 않으며, 현재 full extraction은 시작하지 않았다.
