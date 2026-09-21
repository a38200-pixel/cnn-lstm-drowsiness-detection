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

### Behavior Sequence
- Window: 동일한 10초 clip
- Sampling: 10 FPS, 약 100 timestamp
- Landmark: Dlib68
- Features: EAR, MAR, Pitch/Yaw/Roll
- Use: explicit eye/yawn/head events

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
