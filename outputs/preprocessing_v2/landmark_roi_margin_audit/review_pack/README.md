# STEP 2-C Compact Visual Review Pack

## 목적

YuNet detection bbox를 고정한 상태에서 Dlib68 fitting ROI만 RAW, +5%, +10%, +15%로 변경했을 때 눈·입·윤곽과 pose axis가 어떻게 달라지는지 사람이 검토하기 위한 팩입니다.

- 기존 contact sheet: 18장
- 기존 review sample: 35개
- 최종 중복 제거 sample: 35개
- 최종 compact sheet: 5장
- Decision: `WAITING_FOR_MANUAL_LANDMARK_ROI_REVIEW`

## 후보 정의

- RAW: YuNet detection bbox 그대로 사용
- M05: bbox width와 height에 각각 5% 확장
- M10: 각각 10% 확장
- M15: 각각 15% 확장

모든 panel의 YuNet detection bbox는 동일하며 Dlib68 fitting ROI만 변경됩니다. 강제 square 변환은 사용하지 않았습니다.

## Sheet 목적

- Overview / Normal Reference (7개): 변화가 작은 train/val, drowsy/not_drowsy 균형 참조
- Eye / EAR (7개): 눈 landmark와 EAR 변화가 큰 사례
- Mouth / MAR (7개): 입 landmark와 MAR 변화가 큰 사례
- Head Pose / Landmark (7개): circular pose 및 전체 landmark 변화가 큰 사례
- ROI / Edge / Difficult (7개): clipping flag, 큰 yaw, 낮은 confidence 등을 우선한 난례

자동 metric은 사람이 볼 표본을 분류하는 데만 사용했습니다. ground-truth landmark가 없으므로 RAW나 특정 margin을 best로 자동 결정하지 않습니다. `review_manual_compact.csv`의 입력 필드는 모두 비워 두었습니다.

## Known Issues

기존 audit에서 `roi_clipped` flag와 `clipped_fraction` 값의 일관성을 별도로 확인할 필요가 있습니다. YuNet 성공 157개 행에서 M05/M10/M15 중 하나 이상이 clipping flag=True이지만 기록된 fraction은 0.0입니다. 이 후처리에서는 기존 값을 수정하거나 재해석하지 않았습니다.

Landmark ROI margin과 Context CNN crop margin은 별도 정책입니다. 이 팩은 Dlib68 fitting ROI만 검토합니다.
