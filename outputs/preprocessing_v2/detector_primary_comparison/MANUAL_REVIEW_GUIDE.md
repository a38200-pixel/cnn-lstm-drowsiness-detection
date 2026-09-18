# STEP 2-B Primary Detector 수동 비교 가이드

자동 비교의 Decision은 `WAITING_FOR_MANUAL_PRIMARY_DETECTOR_REVIEW`입니다. 더 빠르거나 검출 성공률이 높다는 이유만으로 detector를 승인하지 않습니다.

## 1. 얼굴 검출

- 각 detector가 실제 운전자 얼굴을 놓치지 않는지 확인합니다.
- 여러 얼굴 중 배경 인물이나 잘못된 영역을 선택하지 않았는지 확인합니다.

## 2. Full Face crop

- 이마, 턱, 양쪽 눈, 입이 유지되는지 확인합니다.
- 얼굴 한쪽이 과도하게 잘리거나 배경이 지나치게 포함되지 않는지 확인합니다.
- 25% margin, square, 224 크기는 audit preview 후보이며 최종 학습 정책이 아닙니다.

## 3. Dlib68 landmark

- 눈, 입, 코, 턱 좌표가 어느 detector bbox에서 더 자연스러운지 비교합니다.
- 안경테, 눈썹, 배경에 landmark가 붙지 않았는지 확인합니다.

## 4. EAR/MAR

- 실제 눈과 입 상태가 같은데 detector 경로에 따라 값이 과도하게 달라지는지 확인합니다.
- 이 값으로 졸음이나 하품 threshold를 결정하지 않습니다.

## 5. Head Pose

- 각 pose axis가 실제 얼굴 방향과 일치하는지 확인합니다.
- detector에 따라 axis가 크게 달라지는지 확인합니다.
- `pitch_centered_candidate`의 양수/음수가 실제 head up/down과 어떻게 대응하는지 관찰합니다. 이번 단계에서는 sign을 확정하지 않습니다.

## 6. 최종 입력

- `preferred_detector`: `HOG`, `YUNET`, `TIE`, `UNCERTAIN`
- `manual_decision`: `PASS`, `FAIL`, `UNCERTAIN`
- 판단 근거는 `review_note`에 기록합니다.
