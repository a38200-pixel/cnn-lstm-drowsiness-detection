# STEP 3-B Manual Review Interpretation Correction

## Reason

초기 입력에서 LIKELY_FALSE_NEGATIVE 15건을 '특별한 pose difficulty 없음'으로 일괄 표현한 것은 지나치게 단순했다. 사용자 후속 시각 검토에서는 상당수 결측 frame에 yaw/profile pose가 있었다. Codex가 이미지를 재판독하거나 새 per-frame pose label을 만들지는 않았다.

## Category Counts

변경 없음: EXPECTED_DETECTOR_MISS 12, LIKELY_FALSE_NEGATIVE 15, PIPELINE/STORAGE 0, UNCERTAIN 0. 총 27건.

## Corrected Interpretation

- 실제 `review_id` 12~23: strong profile, frame edge 방향에 가까운 얼굴, 일부 코 끝 clipping. YuNet miss가 납득 가능한 가장 어려운 그룹. `face_near_edge=TRUE`는 사용자 후속 관찰에 근거한다.
- 나머지 15건: 상당수에 yaw/profile pose가 있지만 얼굴과 주요 특징은 육안으로 충분히 보인다. 심한 clipping·blur·occlusion·corruption은 관찰되지 않아 likely false negative 후보를 유지한다. 개별 extreme pose 여부는 확정되지 않아 `extreme_pose`를 빈 값(미평가)으로 둔다. 빈 값은 정면이나 FALSE를 의미하지 않는다.

## Provenance

- Source: `USER_PROVIDED_VISUAL_REVIEW`, 후속 해석 정정
- Method: `DIRECT_VISUAL_INSPECTION`
- Correction date: 2026-09-20
- Before CSV SHA-256: `77064c5ae0783874497929f08b5ee454dc3a0b6fe419714c85e47a1566cec1a5`
- After CSV SHA-256: `251057001e2443c91d81f8a011485680b68262ec72f307e21a8e0f1e2d74b3e5`
- Automatic metadata preserved: `true`
- 기존 자동 summary 및 STEP 2 frozen policy는 변경하지 않았다.

## Policy Impact

NONE. YuNet primary와 Dlib68 RAW bbox·SQUARE_M10 정책 유지. Pilot 표본의 관찰을 전체 데이터셋에 일반화하지 않는다.

## Future Use

STEP 3-C 전체 train/val materialization 후 STEP 4에서 missing rate, consecutive misses, pose 관련 패턴, 영상별 집중도를 분석할 때 참고한다. Test는 sealed다.
