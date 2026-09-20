# STEP 2 Manual Visual Review Provenance Closure

이 문서는 STEP 2의 **단계별 시각 검토와 전체 정책 결정**을 기록한다. 자동 수치는 기존 audit artifact에서 확인했고, 시각 관찰 및 전체 결정은 사용자가 설명한 직접 검토·대화형 분석의 기록을 근거로 한다. 이번 closure에서 영상·이미지를 다시 분석하거나 결과를 재생성하지 않았다. STEP 2의 test sample은 사용하지 않았다.

초기 계획은 `자동 audit → 표본별 manual CSV → 최종 결정`이었다. 실제 진행 중 compact review pack/contact sheet를 보며 사용자가 직접 시각 검토하고 전체 정책을 결정하는 방식으로 전환됐다. 따라서 전체 결정은 완료됐지만 원본 표본별 CSV의 판단 칸은 채워지지 않았다. 지금 `PASS`, `YUNET`, `RAW`, `SQUARE_M10` 등을 각 행에 소급 입력하면 수행하지 않은 표본별 판정을 만든 셈이므로 **기존 CSV를 그대로 보존한다**. 자동 report의 `WAITING_FOR_MANUAL_*`도 생성 당시 상태로 보존한다.

| 단계 | 수동 검토 closure | 원본 CSV 판단 칸 | 전체 결정 |
|---|---|---:|---|
| STEP 2-A | STEP 2-B의 통제된 decision audit로 대체 | 0/90행 | 없음 |
| STEP 2-B | 단계 수준 시각 검토 완료 | 0/126행 | YuNet primary |
| STEP 2-C | 단계 수준 시각 검토 완료 | 0/35행 | YuNet RAW bbox → Dlib68 |
| STEP 2-D | 단계 수준 시각 검토 완료 | 0/42행 | SQUARE_M10 |

표본 ID·경로 등 자동 생성 필드는 채워져 있으나 위 수치는 **수동 판단 필드**의 채움 상태다. “단계 수준 완료”는 모든 표본에 독립적인 PASS/FAIL이 기록됐다는 뜻이 아니다.

## STEP 2-A — compatibility audit, decision audit로 대체

HOG primary → YuNet fallback → Dlib68의 호환성을 확인한 탐색 단계였다. Head Pose 각도 wrap 표현, detector 간 비교 표본·geometry heuristic의 한계를 확인했고, 같은 frame에서 `HOG → Dlib68`과 `YuNet → Dlib68`을 직접 비교하는 STEP 2-B로 확장했다. STEP 2-A의 별도 전체 detector 결정을 만들거나 90행 CSV를 사후 완성하지 않는다. 상태: `AUTOMATIC AUDIT COMPLETE`, `MANUAL REVIEW: SUPERSEDED_BY_STEP_2B`.

## STEP 2-B — primary detector

원본 comparison report·summary·paired results와 detector 실패·landmark/EAR/MAR/pose 극단값·정상 참조·full-face crop을 포함한 review pack을 검토 근거로 삼았다. Review pack은 45개 표본과 5개 sheet다. 사용자 제공 시각 관찰에 따르면 YuNet bbox가 이마·턱·좌우 경계를 더 자연스럽게 포함한 사례가 많았고, HOG bbox는 좌우 배경 여유가 큰 사례가 있었다. HOG 실패·YuNet 성공한 실제 얼굴 표본도 있었다. 같은 Dlib68을 사용해도 입력 bbox가 달라 EAR/MAR가 변했으며, 이를 YuNet 실패로 단정하거나 HOG bbox를 ground truth로 삼지 않았다. 자동 결과는 HOG 152/160, YuNet 157/160 검출, 평균 detector latency 약 372.15 ms 대 39.44 ms였다.

전체 수동 결정: **PRIMARY DETECTOR = YUNET**. 원본 `manual_review.csv` 126행의 판단 필드는 0행 입력이다. 상태: `MANUAL_VISUAL_REVIEW_COMPLETE`, `PER-SAMPLE STRUCTURED REVIEW: NOT FULLY RECORDED`.

## STEP 2-C — Dlib68 fitting ROI

YuNet bbox를 고정하고 RAW·M05·M10·M15를 비교했다. 35개 review 표본·compact sheet 5장과 원본 contact sheet 18장을 사용한 검토의 범위에는 정상 참조, 눈·입 feature 극단값, pose·landmark 변화, 어려운 표본이 포함됐다. 사용자 제공 관찰에 따르면 RAW가 눈·입·코·턱을 잘 따르는 사례가 많았고, M05의 일관된 개선은 없었으며 M10/M15 일부 표본은 contour·mouth·pose 변화가 컸다. 네 후보의 Dlib68 성공은 모두 157/157이었고 margin에 따라 landmark·EAR/MAR 값이 이동했다.

Clipping flag 이상은 별도 보정에서 `REPORTING_ONLY_BUG`로 확인됐다. ROI 좌표 628/628개가 일치하여 실제 Dlib 입력·feature·시각 자료는 바뀌지 않았다. `NO ROI AUDIT RERUN REQUIRED`다. 전체 수동 결정: **LANDMARK ROI = YUNET RAW BBOX**. 원본 CSV 35행의 판단 필드는 0행 입력이다.

## STEP 2-D — Context crop

사용자는 42개 visual sample과 contact sheet 11장 전체를 직접 비교했다. 제공된 관찰은 RAW_RESIZE의 반복적 얼굴 비율 왜곡, SQUARE_0의 일부 얼굴 외곽 여유 부족, SQUARE_M10의 얼굴·공간 여유·배경 균형, SQUARE_M20의 배경 증가와 얼굴 비중 감소였다. 자동 지표도 함께 참고했다: RAW_RESIZE distortion 중앙값 약 1.426, 얼굴 면적 비율 중앙값 SQUARE_0 약 0.701·M10 약 0.485·M20 약 0.357, padding 필요 M10 3/157·M20 6/157. 자동 수치만으로 best crop을 정하지 않았다.

전체 수동 결정: **CONTEXT CROP = SQUARE_M10**. 이는 preprocessing geometry 선택이며 CNN 학습 정확도 비교가 아니다. 원본 CSV 42행의 판단 필드는 0행 입력이다.

## Frozen policy / next state

| 항목 | 정책 |
|---|---|
| Face detector | YUNET |
| Landmark | DLIB68 |
| Landmark ROI | YUNET RAW BBOX |
| Context crop | SQUARE_M10, bbox 각 축 양쪽 10% margin, 중심 유지 square |
| Context size / padding | RGB 224×224 / IMAGENET MEAN |

이는 `configs/canonical_preprocessing.yaml`의 YuNet 모델·최대 유효 bbox·fallback 없음, Dlib68 predictor·RAW ROI, `square_m10`·0.10·224·ImageNet mean 설정과 일치한다. STEP 3-A는 `IMPLEMENTED`, STEP 3-B는 `WAITING FOR 20-VIDEO PILOT RUN`, STEP 3-C는 `NOT STARTED`다. **TEST SPLIT SEALED.**

구조화된 같은 기록은 `step2_manual_review_summary.json`과 `step2_manual_review_stage_summary.csv`에 있다. 원본 수동 CSV 및 audit output은 수정하지 않았다.
