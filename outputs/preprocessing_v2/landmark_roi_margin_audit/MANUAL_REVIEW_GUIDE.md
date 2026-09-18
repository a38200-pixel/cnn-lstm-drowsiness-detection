# STEP 2-C Manual Review Guide

청록색은 변경하지 않은 YuNet detection bbox이고 녹색은 Dlib68 fitting ROI입니다.
RAW, M05, M10, M15의 눈·입·전체 landmark와 pose axis를 직접 비교하십시오.
`preferred_landmark_roi`에는 RAW, M05, M10, M15, TIE, UNCERTAIN 중 하나를 기록하십시오.
HOG 값이나 자동 수치를 ground truth로 취급하지 말고 최종 margin은 수동 검토 후 결정하십시오.
