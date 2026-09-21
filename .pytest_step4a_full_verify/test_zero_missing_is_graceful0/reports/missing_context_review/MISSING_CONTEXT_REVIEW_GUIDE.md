# STEP 3-B Context Missing-Only 수동 검토 가이드

이 pack은 `context_selected=True`이고 `context_crop_available=False`인 pilot row만 담습니다. 왼쪽은 저장된 source index를 순차 decode한 원본 frame이고, 녹색 사각형은 YuNet bbox가 **실제로 저장된 경우에만** 그립니다. 오른쪽 `CROP MISSING`은 누락을 표시할 뿐 crop을 재생성하지 않습니다.

1. 실제 운전자 얼굴이 보이는지, 다른 물체·동승자와 혼동하지 않았는지 확인합니다.
2. YuNet bbox가 없다면 detector miss가 타당한 상황인지 판단합니다. 얼굴이 선명한데 bbox가 없다면 false negative 후보입니다.
3. bbox가 있는데도 crop이 없으면 `context_status`, 파일 경로, 저장 단계 문제를 우선 조사합니다.
4. Profile/큰 yaw, frame 가장자리, 가림·선글라스·마스크, 저조도·과노출, blur를 체크합니다.
5. 이미지의 `neighbor_hint`는 근처 *다른* frame의 yaw/edge 정보입니다. 현재 frame의 pose 정답으로 사용하지 마십시오.

`missing_context_manual_review.csv`에서 `manual_category`는 `EXPECTED_DETECTOR_MISS`, `LIKELY_FALSE_NEGATIVE`, `POSSIBLE_STORAGE_OR_PIPELINE_ISSUE`, `UNCERTAIN` 중 하나를 입력합니다. `review_decision`은 `PASS`, `CHECK_NEEDED`, `INVESTIGATE` 중 하나를 입력합니다. `detector_miss_plausible` 등 관찰 칸과 `review_note`는 사람이 직접 채웁니다. 자동 추정값을 수동 칸으로 복사하지 마십시오.

검토 목적은 STEP 3 구현·missing pattern 확인이며 YuNet threshold, SQUARE_M10, fallback 또는 졸음 행동 임계값을 재선택하는 것이 아닙니다. 수동 검토 전까지 판정은 대기 상태입니다.
