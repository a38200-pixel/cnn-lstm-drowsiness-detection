# STEP 2-B Visual Review Numeric Summary

이 문서는 visual sample 선정을 위한 자동 metric 요약입니다. 이미지의 의미적 품질이나 primary detector를 자동 판정하지 않습니다.

## Detector 및 outcome

- HOG: 152/160 (95.00%)
- YuNet: 157/160 (98.12%)
- Both success: 151
- HOG only: 1
- YuNet only: 6
- Both failed: 2

## Latency

- HOG detector: mean 372.15 ms, median 364.74 ms, P95 410.97 ms
- YuNet detector: mean 39.44 ms, median 37.42 ms, P95 49.82 ms
- HOG behavior pipeline: mean 376.17 ms
- YuNet behavior pipeline: mean 43.78 ms

## Landmark 및 EAR/MAR

- HOG+Dlib68: 152/152
- YuNet+Dlib68: 157/157
- Normalized landmark diff: median 0.027922, P95 0.138505, max 0.303298
- EAR: Pearson 0.5993, Spearman 0.5534, abs diff median 0.042954, P95 0.211469
- MAR: Pearson 0.2723, Spearman 0.3928, abs diff median 0.037503, P95 0.414597

EAR/MAR correlation이 충분히 높지 않다는 사실은 어느 detector가 잘못됐다는 증거가 아닙니다. 동일 Dlib68 predictor라도 detector bbox가 landmark fitting과 파생 geometry에 영향을 줄 수 있다는 신호이므로 실제 눈·입 위치를 visual review해야 합니다.

## Circular pose difference

- Pitch: median 2.560°, P95 12.549°, max 42.967°
- Yaw: median 2.668°, P95 17.103°, max 98.012°
- Roll: median 1.325°, P95 14.209°, max 175.669°

## Crop padding

- HOG: 1/152, mean 0.000238, median 0.000000, max 0.036207, P90 0.000000, P95 0.000000
- YuNet: 13/157, mean 0.004646, median 0.000000, max 0.140649, P90 0.000000, P95 0.029986

Padding 빈도와 크기는 숫자로만 보고하며 품질의 좋고 나쁨을 자동 판정하지 않습니다.

## Review pack

- 최종 sample: 45
- Normal reference: 10
- 반복 outlier 상위 video: d_75, n_241, n_640, d_25, n_658, d_5, d_852, d_488, d_682, n_461
- Decision: `WAITING_FOR_MANUAL_PRIMARY_DETECTOR_REVIEW`
