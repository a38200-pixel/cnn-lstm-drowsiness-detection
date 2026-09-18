# STEP 2 수동 Visual Review 가이드

자동 audit의 최종 상태는 `WAITING_FOR_MANUAL_VISUAL_REVIEW`입니다. CSV의 수동 입력 컬럼은 비워 두었으며 `PASS`, `FAIL`, `UNCERTAIN` 중 하나를 직접 기록합니다.

## bbox_ok

- bbox가 얼굴 전체를 적절히 포함하는가?
- 이마, 턱, 눈, 입이 과도하게 잘리지 않았는가?
- 배경을 지나치게 많이 포함하지 않는가?

## landmarks_overall_ok

- 68개 landmark가 얼굴 구조를 전반적으로 따르는가?

## eyes_ok

- 눈 landmark가 실제 눈꺼풀과 눈꼬리에 위치하는가?
- 안경테나 눈썹에 잘못 붙지 않았는가?
- 좌우 눈 geometry가 붕괴하지 않았는가?

## mouth_ok

- 입 landmark가 실제 입술 경계를 따르는가?
- landmark가 입 밖으로 이동하지 않았는가?

## nose_chin_ok

- Head Pose에 사용하는 코끝과 턱 landmark가 정상인가?
- 큰 yaw에서 턱 landmark가 얼굴 밖으로 이탈하지 않는가?

## pose_axis_ok

- 빨강/초록/파랑 pose axis가 실제 얼굴 방향과 대체로 일치하는가?
- axis 방향이 명백히 반대이거나 불안정하지 않은가?

## manual_decision

- `PASS`: 이후 preprocessing에 사용할 수 있음
- `FAIL`: bbox, landmark 또는 pose가 신뢰하기 어려움
- `UNCERTAIN`: 단일 이미지로 판단하기 어렵거나 추가 확인이 필요함

특히 안경테·눈썹에 붙은 eye landmark, 큰 yaw의 반대쪽 눈 붕괴, 입 밖의 mouth landmark, 얼굴 일부를 자른 bbox, 과도한 배경, 실제 방향과 다른 pose axis를 주의합니다. 라벨은 참고 표시일 뿐 판단 기준이 아닙니다.
