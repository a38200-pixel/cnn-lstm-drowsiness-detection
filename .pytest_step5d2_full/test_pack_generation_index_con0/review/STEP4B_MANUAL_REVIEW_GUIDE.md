# STEP 4-B 수동 시각 검토 안내

## 핵심 질문

**YuNet이 놓친 구간에서 사람이 보기에 얼굴은 얼마나 식별 가능한가?**

얼굴의 profile·큰 yaw, frame edge, 얼굴 잘림, 조명, blur, 가림, 얼굴 크기, 얼굴 부재·뒤통수 여부를 확인하세요. Miss가 고립된 한 frame인지 연속 구간인지, run 전후 pose·얼굴 상태가 달라지는지, 얼굴이 충분히 보여도 miss가 지속되는지도 살펴보세요.

## 이미지 찾기와 비교

`step4b_review_index.csv`의 `review_id`와 `individual_sheet_path`가 1:1로 대응합니다. `contact_sheet_id`는 네 영상씩 묶은 전체 목차입니다. 먼저 contact sheet를 훑고 해당 individual sheet를 확대하세요. Sheet의 왼쪽은 원본 frame이며 초록 상자는 **기존 metadata에 저장된 YuNet bbox**입니다. 오른쪽은 해당 slot에 이미 저장된 SQUARE_M10 JPEG입니다. `YUNET MISS`에는 저장 bbox가 없으며, `CROP MISSING`은 새 crop으로 채우지 않았습니다.

높은 결측·긴 run은 valid-before → run-start/mid/end → valid-after를 비교하세요. 전후 frame이 없으면 표시하지 않습니다. 100/100 miss는 k000·025·050·075·099를 확인하세요. Context 결측 후보는 초·중·후 결측 slot과 가능한 저장 crop 참조를 비교하세요. 고립 miss는 앞·해당·뒤 frame을, 정상 참조는 초·중·후를 비교하세요. 선택된 일부 frame만 보여주므로 표시되지 않은 구간까지 단정하지 마세요.

## 수동 입력

`step4b_manual_review.csv`의 자동 metadata 열은 유지하고, 모든 manual 열은 **실제 이미지를 확인한 후** 직접 입력하세요. 미평가 열은 빈 칸으로 남기세요. 예/아니요 성격의 열은 `YES`/`NO`/`UNCERTAIN`을 권장합니다.

`manual_pattern` 허용 예: `PROFILE_YAW`, `FRAME_EDGE`, `PARTIAL_FACE`, `OCCLUSION`, `LOW_LIGHT`, `BLUR`, `SMALL_FACE`, `NO_FACE_OR_BACK_HEAD`, `MIXED`, `NO_OBVIOUS_DIFFICULTY`, `NORMAL_REFERENCE`, `UNCERTAIN`. 복수는 세미콜론으로 구분할 수 있습니다.

`sequence_severity`: `ISOLATED`, `SHORT_RUN`, `LONG_RUN`, `MOSTLY_MISSING`, `FULL_CLIP_MISSING`, `NORMAL_REFERENCE`. 이 값은 **사람의 시각 판단**이며 자동 run length와 별개입니다.

`manual_category`:

- `EXPECTED_DETECTOR_MISS`: 심한 profile·가림·어둠·얼굴 부재 등 사람이 봐도 검출이 어려운 조건
- `LIKELY_FALSE_NEGATIVE`: 사람이 볼 때 충분히 식별 가능한 얼굴을 YuNet이 놓친 경우
- `MIXED_MISSING_PATTERN`: 같은 영상/구간에 두 유형이 혼재
- `SOURCE_CONTENT_ISSUE`: 운전자/얼굴이 카메라 시야 밖에 있는 등 원본 내용의 문제
- `POSSIBLE_PIPELINE_ISSUE`: frame·영상 불일치, bbox/metadata 불일치, 이미지 손상 의심
- `NORMAL_REFERENCE`: 결측 없는 정상 비교 영상
- `UNCERTAIN`: 근거 부족

`review_decision`: `PASS`는 관찰된 결측이 source/detector 특성으로 설명되거나 정상 참조, `CHECK_NEEDED`는 false negative·혼합 양상 등 STEP 4-C 검토 필요, `INVESTIGATE`는 정책 결정 전 source/pipeline 이상 조사가 필요한 경우입니다. 의심 근거는 `review_note`에 적으세요.

STEP 4-B는 검토 자료와 사람의 판정을 준비하는 단계입니다. **이 CSV는 모두 미판정 상태로 생성되며, 결측 처리 정책·임계값은 여기서 바꾸지 않습니다.** Test split은 sealed입니다.
