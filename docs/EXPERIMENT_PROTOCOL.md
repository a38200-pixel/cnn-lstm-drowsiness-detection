# Experiment 2 Protocol

## Baseline candidates
1. ResNet18 pretrained + GAP + LSTM128
2. VGG16 pretrained + GAP + LSTM128

두 backbone은 별도 run으로 실행하며 feature fusion하지 않습니다.

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
