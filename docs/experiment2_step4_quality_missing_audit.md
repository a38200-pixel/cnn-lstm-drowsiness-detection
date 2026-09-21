# Experiment 2 STEP 4 — Quality & Missing Audit

## 1. Objective

STEP 4는 자동 관찰(4-A), 선정 사례의 수동 시각 검토(4-B), 결측 처리 정책 선택·동결(4-C)로 분리한다. 이 문서는 **STEP 4-A 실제 결과**까지만 기록한다. 원본 영상 재처리, detector 재선택, missing 보정, 행동 threshold 결정 및 test 분석은 하지 않았다.

## 2. Input Canonical Dataset

STEP 3-C가 생성한 `data/interim/preprocessing_v2/canonical/metadata/`의 `canonical_frames.csv`, `canonical_videos.csv`, `run_metadata.json`만 주 입력으로 사용했다. Train 1,452개·val 311개, 총 1,763개 영상·176,300개 canonical frame이다. 영상당 100행, 검출과 무관한 Context 32 slot, test row 0개, video ID 중복 0개를 사전 확인했다. Policy hash는 STEP 3-C의 `29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721`와 같다. STEP 3-C 확정 summary의 주요 수치를 독립 재계산값과 대조했다.

`scripts/run_quality_missing_audit.py --dry-run`은 metadata 구조만 확인하고 결과를 쓰지 않는다. 실제 실행은 새 [STEP 4-A 출력 디렉터리](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/)에만 파일을 발행했다. STEP 3-C canonical dataset, pilot·manual review artifact, raw video는 수정·재실행하지 않았다.

## 3. Missing Definitions

- Detector missing: `yunet_success=False`인 canonical frame.
- Behavior valid: YuNet·Dlib68 성공이고 EAR·MAR가 모두 finite인 frame. Pose valid는 별도 집계한다.
- Context missing: `context_selected=True`이면서 `context_crop_available=False`인 32개 선택 slot 중 하나.
- YuNet missing run: 100개 canonical index에서 검출 실패가 연속되는 포함 구간.
- Context missing run: 선택된 32개 Context slot **순서**에서 crop 결측이 연속되는 구간. 100개 frame의 연속성과 다르다.

## 4. STEP 4-A Automatic Audit

| 항목 | Full train/val 실제 결과 |
|---|---:|
| 영상 / canonical frame | 1,763 / 176,300 |
| YuNet 성공 / 미검출 | 170,871 / 5,429 (결측률 3.0794%) |
| Behavior 유효 / Pose 유효 | 170,871 / 170,871 |
| YuNet 성공 후 Dlib68 추가 실패 | 0 / 170,871 |
| Context 선택 / 가용 / 결측 | 56,416 / 54,672 / 1,744 (결측률 3.0913%) |
| Padding 필요 crop / multiple-face frame | 885 / 0 |
| 수치·flag 모순 / test row | 0 / 0 |

이 수치는 [자동 보고서](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/quality_missing_report.txt)와 [JSON 요약](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/quality_missing_summary.json)에 기록했다. STEP 3-C에서 전체 JPEG strict decode가 통과했으며 이번 audit는 그 검사를 반복하지 않고 저장된 metadata만 확인했다. 이번 materialization에서 YuNet 성공 frame의 Dlib68 추가 결측은 없었지만, 이를 모든 환경의 landmark 정확도 주장으로 일반화하지 않는다.

## 5. Per-Video Missing

[영상별 품질표](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/per_video_quality.csv)는 1,763개 영상 각각의 검출·Behavior·Pose·Context 가용성, 시작/끝 결측, 최장 run 및 결측률을 담는다. YuNet 결측 수/video의 최소·중앙값은 0, 평균 3.079, p90 9, p95 약 19.9, p99 48, 최대 100이다. YuNet 결측 0개 영상은 1,382개, Behavior 100/100 유효 영상도 1,382개다. Context 결측 0개 영상은 1,425개다.

최다 결측은 train/not_drowsy의 `n_246`으로 100/100 frame 미검출·Context 32/32 결측이었다. YuNet 성공 0개 영상은 1개, 성공 50개 미만은 16개, Context 가용 0개는 1개, 16개 미만은 14개다. 이 값은 STEP 4-B 검토를 위한 진단이며 자동 제외 기준이 아니다. 상위 사례 50개씩은 `top_missing_videos.csv`, `top_context_missing_videos.csv`, `top_consecutive_missing_videos.csv`에 정렬해 두었다.

## 6. Consecutive Missing Runs

[전체 검출 결측 run](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/missing_runs.csv)은 967개다. 단일 frame run 353개, 2개 이상 연속 run 614개이며 최장은 100 frame이다. 각 run의 canonical index·timestamp·clip 시작/끝 접촉 여부·run 안의 Context 선택/결측 수를 기록했다. [길이별 histogram](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/missing_run_length_histogram.csv)은 1~최장 길이의 count와 fraction을 제공한다. 연속 길이의 위험 threshold는 정하지 않았다.

결측 run 앞뒤 최대 ±5 canonical slot의 가장 가까운 *유효 frame* yaw는 [별도 이웃 pose 힌트](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/missing_run_neighbor_pose.csv)에 분리했다. 이는 결측 frame 자신의 pose가 아니다.

## 7. Context Missing

1,744/56,416개 slot이 결측이며 영상별 결측 수의 중앙값은 0, p90 3, p99 16, 최대 32다. [정확한 0~32개 histogram](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/context_missing_count_histogram.csv)에서 결측 0개 영상은 1,425개, 32개 결측 영상은 1개다.

[Context 누적 가용성 표](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/context_missing_cumulative.csv)는 `context_missing <= N`을 **N=0~32 전체**에 대해 계산했다. 예를 들어 N=0은 1,425개, N=2는 1,568개, N=8은 1,704개, N=16은 1,749개, N=32는 1,763개다. [Behavior 누적표](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/behavior_missing_cumulative.csv)도 N=0~100 전체를 제공한다. 어느 N도 이번 단계의 허용 정책으로 선택하지 않았다.

## 8. Split / Label Comparison

| 층 | 영상 | YuNet 결측 | YuNet 결측률 | Context 결측 |
|---|---:|---:|---:|---:|
| Train | 1,452 | 4,385 / 145,200 | 3.0200% | 1,413 / 46,464 |
| Val | 311 | 1,044 / 31,100 | 3.3569% | 331 / 9,952 |
| Drowsy | 829 | 1,858 / 82,900 | 2.2413% | 604 / 26,528 |
| Not-drowsy | 934 | 3,571 / 93,400 | 3.8233% | 1,140 / 29,888 |

Val−train YuNet 결측률 차이는 약 +0.337 percentage point, Context 결측률 차이는 약 +0.285 percentage point다. Train×drowsy, train×not_drowsy, val×drowsy, val×not_drowsy의 네 층도 [split/label 표](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/split_label_quality_summary.csv)에 있다. 이는 `PREPROCESSING GAP OBSERVED` 수준의 기술 통계다. 운전자·조명·자세 구성이 섞여 있고 subject-wise ID mapping이 없으므로 label 또는 split 차이의 원인을 단정하지 않는다.

## 9. Temporal Distribution

[Canonical 0~99 slot](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/temporal_missing_by_canonical_slot.csv)과 [Context 0~31 slot](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/temporal_missing_by_context_slot.csv)의 선택 수·결측 수·결측률을 각각 기록했다. 구간은 first 10=`k0..9`, middle 80=`k10..89`, last 10=`k90..99`로 정의했다. YuNet 결측률은 차례로 2.9836%, 3.0686%, 3.2615%였다. 이는 경계 진단이며 codec 또는 pose 원인이라고 결론짓지 않는다.

## 10. Missing Concentration

YuNet 미검출 5,429개 중 상위 1%(18개) 영상에 1,195개(22.0114%), 상위 5%(89개)에 3,405개(62.7187%), 상위 10%(177개)에 4,607개(84.8591%)가 모였다. [집중도 CSV](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/missing_concentration_summary.csv)는 분모·개수를 함께 제공한다. 전체 frame 결측률만으로는 영상별 집중도를 설명할 수 없음을 보여주는 관찰이다.

## 11. Numeric Quality

YuNet 성공 frame 기준 EAR·MAR·Pitch raw·Pitch centered candidate·Yaw·Roll의 finite 수치는 각각 170,871개이며, 분포(min, mean, std, p01, p05, median, p95, p99, max)는 JSON 요약에 보존했다. YuNet bbox 너비·높이·면적 비율·confidence 분포는 [bbox 진단 JSON](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/bbox_quality_summary.json)에 있다. 성공 flag와 NaN/inf의 모순 및 명백한 수치 overflow·bbox 무효 진단은 0건이며 [이상 목록 CSV](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/numeric_quality_anomalies.csv)는 헤더만 있다. EAR/MAR/Pose 값을 행동 threshold나 event로 변환하지 않았다.

## 12. STEP 4-B Candidate Selection

[36개 후보 manifest](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_review_candidates.csv)는 영상 ID를 중복 제거하고 결정적으로 선택했다: 최다 검출 결측 8, 최장 연속 run 8, 최다 Context 결측 8, 단일 고립 결측 6, 결측 0 정상 참조 6개다. Train/drowsy 10, train/not_drowsy 10, val/drowsy 8, val/not_drowsy 8개를 포함한다. 각 row에 대표 결측 구간과 다음 단계에서 확인할 canonical index를 적었다. 이번 단계에서는 이미지를 만들거나 사람의 시각 판정을 채우지 않았다.

## 13. Limitations

STEP 3-B의 27건 사용자 시각 검토에서는 yaw/profile 관련성이 관찰됐다. 그러나 이 STEP 4-A는 전체 원본 이미지를 보지 않았으며, YuNet이 실패한 frame의 Dlib pose도 없다. 따라서 전체 5,429건의 pose 원인, Context 결측 1,744건의 시각적 유형, 적정 결측 허용 개수는 결정할 수 없다. 이웃 유효 frame yaw를 결측 frame yaw로 대체하지 않는다. Test split은 열지 않았고 최종 평가 전까지 sealed다.

## 14. Current Decision Status

- STEP 4-A: **AUTOMATIC QUALITY & MISSING AUDIT COMPLETE**
- STEP 4-B: **WAITING FOR MANUAL VISUAL REVIEW**
- STEP 4-C: **NOT STARTED**
- Missing handling policy: **NOT SELECTED**
- Frozen YuNet·Dlib68 RAW bbox·SQUARE_M10: **UNCHANGED**
- Test split: **SEALED**
