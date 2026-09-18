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
