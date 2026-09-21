# Experiment 2 Design Notes

## 1. 1차에서 유지할 것과 버릴 것

### 유지
- 원본 SUST-DDD 영상
- video-level train/val/test split
- split integrity 원칙
- test set 격리
- seed 고정/기록
- min validation loss checkpoint 규칙
- 1차 실험 보고서와 metrics

### 재설계
- 프레임 sampling
- Full Face crop
- RGB 입력
- sequence 구성
- missing-frame 정책
- Dataset/DataLoader
- backbone
- LSTM 입력 차원
- Rule Engine
- normalization
- optimizer/학습률

## 2. 왜 기존 NPZ를 주 입력으로 쓰지 않는가
기존 NPZ에는 10 Hz로 이미 선택된 프레임, grayscale Eye/Eye/Mouth composite, EAR/MAR, validity, timestamp만 남아 있습니다. 2차 설계는 Full Face RGB, Head Pose, 10초 전체 문맥을 사용하므로 원본 영상부터 다시 처리해야 합니다.

## 3. Face/Landmark 정책 권장안
1차의 가장 큰 전처리 문제 중 하나는 HOG face detection 실패였습니다. 반면 V2에서 3DDFA를 혼합했을 때 EAR/MAR 수치 체계가 Dlib와 일치하지 않아 calibration이 실패했습니다.

따라서 2차의 첫 기준안은 다음을 권장합니다.
- Face detector: YuNet을 우선 후보로 사용
- Landmark: 모든 유효 프레임에서 Dlib68로 통일
- 3DDFA EAR/MAR를 섞지 않음

주의: 과거의 `YuNet + 3DDFA` 99.85% 성공률을 `YuNet + Dlib68` 성능으로 간주하면 안 됩니다. 새 조합은 train/val에서 별도 preprocessing audit이 필요합니다.

## 4. 시퀀스 정책

### Context Sequence
- Sample unit: SUST-DDD 원본 10초 clip 1개 = training sample 1개
- Frames: 32
- Sampling: clip 전체에서 균등 sampling
- Input: Full Face RGB 224x224x3
- Backbone: ResNet18 또는 VGG16 (별도 실험)
- Output per frame: GAP 512-D
- Backbone output sequence: `[B, 32, 512]`
- Temporal encoder: 1-layer unidirectional LSTM, input 512, hidden 128, `batch_first=true`, internal dropout 0.0
- Temporal output: last hidden state `[B, 128]`
- Classifier: `Linear(128→64) → ReLU → Linear(64→2)`, classifier dropout 0.0
- Loss: raw logits를 사용하는 `CrossEntropyLoss`; model 내부 필수 softmax 없음

ResNet18과 VGG16의 첫 비교에서는 split, 32 target timestamps, missing policy, LSTM/classifier, dropout, loss, training protocol과 checkpoint criterion을 동일하게 유지하고 backbone만 바꾼다. Dropout 0.0은 base paper의 완전 재현이 아니라 추가 regularization 변수를 줄이기 위한 Experiment 2 초기 baseline 결정이다. Baseline 결과에서 과적합 또는 일반화 문제가 확인될 경우에만 0.1/0.3/0.5 등을 별도 controlled ablation 후보로 검토하며 아직 범위를 동결하지 않는다. 1-layer LSTM에는 internal dropout을 적용하지 않는다.

STEP 5-D1 controlled pilot에서는 두 frozen backbone 모두 CNN feature extraction batch size를 16으로 통일했다. 이는 pretrained feature의 의미를 맞추기 위한 요구가 아니라 runtime·VRAM·pipeline 비교의 run setting을 통제하기 위한 결정이다. 향후 LSTM training batch size와는 별개다.

STEP 5-D2는 STEP 5-A의 Context eligible 1,677개와 frozen `source_image_relpath`를 그대로 소비한다. Behavior eligibility로 대상을 줄이거나 nearest-neighbor source를 다시 계산하지 않는다. 53,664 target을 52,965 unique source inference로 deduplicate하고 699개 imputed target은 source feature를 정확히 재사용한다. Full artifact는 `context_full/resnet18`과 `context_full/vgg16`으로 분리하고, 원자적 publish와 provenance 일치 기반 `--resume`을 사용한다.

STEP 5-D2R에서 두 backbone 모두 `n_311`의 마지막 6개 unique source가 D1의 6장 partial batch와 D2의 16장 full batch에서 작은 수치 차이를 보였다. Source path와 target/source index는 동일했고, 각 stored vector는 동일 batch composition 재실행 결과와 exact match했다. 따라서 이는 mapping 또는 input 오류가 아니라 benign floating-point/batch numerical difference이며 full artifact를 재생성하지 않는다. Source universe마다 달라지는 `source_feature_id` 자체는 semantic mapping equality 기준으로 사용하지 않고 `source_image_relpath`와 canonical/context index를 기준으로 비교한다.

### Behavior Sequence
- Window: 동일한 10초 clip
- Sampling: 10 FPS, 약 100 timestamp
- Landmark: Dlib68
- Features: `ear`, `mar`, `pitch_raw`, `pitch_centered_candidate`, `yaw`, `roll`의 `[100,6]`
- Future use: threshold가 동결된 뒤 explicit eye/yawn/head event 또는 temporal model 입력 검토

Behavior raw continuous feature는 smoothing이나 interpolation 없이 저장된 값과 NaN/mask를 그대로 보존한다. Train/validation 대표 표본 visual sanity check에서 missing 구간과 validity mask가 일치했으며, 일부 valid head-pose frame의 `pitch_raw`에서 representation discontinuity 또는 wrap-like pattern 가능성이 있는 큰 수치 전환과 간헐적인 `roll` spike가 관찰됐다. 이는 확정된 오류나 물리 방향 해석이 아니다.

초기 baseline은 여섯 feature를 모두 변경 없이 유지한다. Pose clipping, angle unwrap, smoothing, feature exclusion 또는 centered feature 선택은 visual inspection만으로 적용하지 않으며, 필요하면 이후 validation 기반 controlled ablation으로 검증한다. Head-pose physical sign/directional semantics는 아직 unresolved 상태다.

### Stride 구분
- Training dataset: 원본 10초 clip 당 1 sample. 같은 clip에서 overlap window를 여러 개 만들지 않음.
- Real-time inference: 10초 rolling window + 별도 decision stride(예: 0.5초)를 사용할 수 있음. 이 값은 training sequence stride와 구분해서 관리.

## 5. Missing-frame 정책
- 시간축을 compaction하지 않음.
- 다른 시점 프레임을 붙여 10초 시간을 인위적으로 압축하지 않음.
- Context와 Behavior 모두 `valid/missing mask`를 metadata에 기록.
- 짧은 face-detection failure를 어떻게 대체할지는 별도 train/val audit 후 결정.
- 긴 missing gap이 있는 clip의 사용 기준은 preprocessing_v2 단계에서 사전 정의.

## 6. Label 정책
- Context model label: 원본 10초 clip label을 그대로 사용.
- 2초 sub-sequence로 label을 반복 상속하지 않음.
- Rule threshold는 clip-level drowsy/alert label만으로 정밀 event threshold라고 간주하지 않음.
- EAR/MAR/Head-Pose event calibration은 가능한 경우 train/val의 event annotation 또는 문헌 기반 threshold를 사용하고 test는 사용하지 않음.

## 7. 정규화
- RGB backbone: pretrained ImageNet normalization을 기본 후보로 사용.
- EAR/MAR/Head Pose: 필요하면 2차 train split에서만 새 통계를 계산.
- 1차 `normalization_stats.json`은 재사용 금지.

## 8. 공정 비교
- 같은 video-level split 유지
- 같은 seed set을 재사용 가능(42/43/44)하되 pretrained model initialization도 함께 기록
- checkpoint selection: minimum validation loss
- test는 최종 선택 전까지 0회 사용
- 1차와 2차 비교 시 입력/시퀀스/모델이 모두 바뀌므로 '단일 변수 ablation'이 아니라 'redesign comparison'으로 표현

## 9. 특히 주의할 위험
- subject-wise split을 만들 수 없으면 Full Face가 subject identity bias를 더 강하게 학습할 수 있음
- 10초 1-sample 정책은 near-duplicate를 줄이지만 sample 수 감소를 동반함
- YuNet+Dlib68 조합의 실제 landmark coverage는 새로 측정해야 함
- 32-frame context가 짧은 행동을 놓칠 수 있으므로 Rule stream으로 보완
- pretrained ResNet/VGG 내부 BatchNorm의 동작이 1차 GN 실험과 다르므로 train/freeze 정책을 명확히 해야 함
- Rule Engine과 Decision Fusion은 validation에 과도하게 맞추지 않도록 사전 규칙을 고정해야 함

## 10. STEP 6 Input LayerNorm Controlled Experiment

### 후보 선정 근거와 가설

Ba, Kiros, Hinton (2016)의 Layer Normalization은 Batch Normalization과 달리 mini-batch statistics에 의존하지 않고 train/test에서 동일한 normalization 계산을 사용한다. 원 논문은 recurrent neural network에 적용하기 쉬운 방법으로 제시하며 recurrent hidden-state dynamics의 안정화와 optimization/training 안정화 가능성을 보고했다.

이번 실험은 원 논문의 직접 재현이 아니다. 원 논문의 recurrent setting과 달리 여기서는 저장된 frozen CNN feature가 LSTM에 들어가기 직전에만 `LayerNorm(512)`을 한 번 적용했다. 가설은 ResNet18과 VGG16 frozen 512D feature의 scale/distribution 차이를 입력단에서 완화하면 두 backbone의 validation generalization이 공통으로 개선될 수 있다는 것이었다. 논문의 결과가 이 프로젝트의 성능 향상을 보장한다고 전제하지 않았다.

```text
Baseline
Frozen CNN feature [B,32,512] → LSTM → classifier

Controlled experiment
Frozen CNN feature [B,32,512] → LayerNorm(512) → LSTM → classifier
```

LayerNorm만 단일 변수로 변경했다. 두 backbone 모두 seed 42, train/validation batch 16/32, learning rate 0.0005, weight decay 0.0001, classifier/LSTM dropout 0.0, 동일 scheduler와 early stopping을 사용했다. AMP는 사용하지 않았고 test split은 sealed 상태를 유지했다. Frozen feature artifact와 Dataset은 변경하지 않았다.

### Validation 결과

| Backbone | Setting | Val Loss | Accuracy | Macro F1 | Drowsy Recall |
|---|---|---:|---:|---:|---:|
| ResNet18 | Baseline | 0.4906 | 0.7797 | 0.7784 | 0.7324 |
| ResNet18 | LayerNorm | 0.4973 | 0.7661 | 0.7637 | 0.6901 |
| VGG16 | Baseline | 0.4637 | 0.8068 | 0.8061 | 0.7746 |
| VGG16 | LayerNorm | 0.5000 | 0.7695 | 0.7683 | 0.8732 |

ResNet18은 validation loss, accuracy, Macro F1과 drowsy recall이 모두 악화됐다. Train loss는 계속 감소했지만 validation loss는 조기에 정체한 뒤 상승해 generalization 개선 효과가 관찰되지 않았다.

VGG16도 validation loss, accuracy와 Macro F1이 악화됐다. Drowsy recall은 증가했지만 confusion matrix가 baseline `TN 128 / FP 25 / FN 32 / TP 110`에서 LayerNorm `TN 103 / FP 50 / FN 18 / TP 124`로 변했다. False negative는 감소했으나 false positive가 25에서 50으로 증가했고 drowsy precision도 낮아졌으므로 recall 상승을 전체 validation 품질 개선으로 해석하지 않는다.

### Negative result의 의미와 결정

이번 Input LayerNorm 방식으로는 두 backbone의 공통 개선이 관찰되지 않았고 모두 전체 validation 성능이 악화됐다. 따라서 `LayerNorm(512)`은 현재 또는 향후 공통 baseline에 적용하지 않는다. 이번 결과는 frozen feature의 단순 scale/distribution 차이가 validation 문제의 주요 원인이라는 가설을 지지하지 못하지만, feature distribution 문제가 완전히 배제됐다는 뜻은 아니다.

LayerNorm 적용 후 train fitting이 계속 진행되면서 validation loss가 상승한 현상은 모순이 아니다. LayerNorm은 그 자체가 regularization을 목적으로 하는 기법이 아니므로 train loss의 지속적 감소만으로 validation generalization을 기대하지 않는다. 이 결과는 seed42 development experiment의 negative result이며 최종 backbone 성능이 아니다.

LayerNorm 실험은 미채택으로 종료한다. 다음 regularization 후보로 Label Smoothing을 검토하며, 공통 정책 동결과 ResNet18/VGG16의 seeds 42/123/2026 validation은 아직 수행 전이다. Test split은 계속 **SEALED** 상태다.

참고문헌: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
