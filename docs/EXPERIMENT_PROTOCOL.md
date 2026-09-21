# Experiment 2 Protocol

## Baseline candidates
1. ResNet18 pretrained + GAP + LSTM128
2. VGG16 pretrained + GAP + LSTM128

두 backbone은 별도 run으로 실행하며 feature fusion하지 않습니다.

## Common Context Baseline Architecture

- CNN feature sequence: `[B, 32, 512]`
- LSTM: `input_size=512`, `hidden_size=128`, `num_layers=1`, `batch_first=true`, `bidirectional=false`, `dropout=0.0`
- Temporal representation: last hidden state `[B,128]`
- Classifier: `Linear(128→64) → ReLU → Linear(64→2)`
- Classifier dropout: `0.0`
- Loss: `CrossEntropyLoss` with raw logits; 필수 softmax layer를 model 내부에 두지 않음

Dropout 0.0은 base paper의 완전 재현을 의미하지 않으며, 두 backbone의 초기 비교에서 추가 regularization 변수를 줄이기 위한 Experiment 2 baseline 결정입니다. Baseline train/validation 결과에서 과적합 또는 일반화 문제가 확인된 뒤에만 dropout을 별도 controlled ablation으로 검토합니다. 후보 0.1/0.3/0.5와 실험 범위는 아직 동결하지 않았습니다. `num_layers=1`이므로 LSTM internal dropout도 0.0입니다.

현재 `configs/model_resnet18_lstm.yaml`과 `configs/model_vgg16_lstm.yaml`은 초기 scaffold이며 아직 실행 가능한 baseline training protocol로 동결되지 않았습니다. 향후 학습 구현 작업에서 이 문서와 일치하도록 별도 검토해야 하며, 이번 문서 변경에서는 config나 학습 코드를 수정하지 않습니다.

첫 frozen CNN feature controlled comparison에서는 ResNet18과 VGG16 모두 feature extraction batch size 16, CUDA float32, AMP off, eval mode와 `torch.inference_mode()`를 사용한다. Batch size 통일은 feature 의미를 결정하기 위한 조건이 아니라 runtime·VRAM·pipeline 비교에서 불필요한 run-condition 차이를 줄이기 위한 것이다. CNN feature extraction batch와 향후 LSTM training batch는 서로 다른 설정이며, training batch size는 별도 training protocol에서 동결한다.

## Common input
- 10-second original clip
- 32 uniformly sampled Full Face RGB frames
- 224x224x3

## Behavior rule stream
- 10 FPS
- Dlib68
- EAR / MAR / Head Pose

## Evaluation
- Primary: clip-level validation loss / accuracy / precision / recall / F1
- Stability: multi-seed history and collapse inspection
- Optional rule evaluation: event-level only when valid annotations exist
- Real-time profiling: face detection, landmark, CNN, LSTM, fusion latency separated

## Test policy
- test split remains sealed until model family, preprocessing, thresholds, and fusion policy are frozen.
