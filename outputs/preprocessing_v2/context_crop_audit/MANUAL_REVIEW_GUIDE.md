# STEP 2-D 수동 검토

RAW_RESIZE 왜곡, 정사각형 후보의 얼굴 완전성·배경·padding을 비교합니다. `preferred_context_crop`에는 RAW_RESIZE, SQUARE_0, SQUARE_M10, SQUARE_M20, TIE, UNCERTAIN 중 하나를 기록합니다. 수치나 label만으로 best crop을 정하지 않습니다.
