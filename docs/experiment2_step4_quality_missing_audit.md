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
- STEP 4-B: **COMPLETE**
- STEP 4-C: **NOT STARTED**
- Missing handling policy: **NOT SELECTED**
- Frozen YuNet·Dlib68 RAW bbox·SQUARE_M10: **UNCHANGED**
- Test split: **SEALED**

## STEP 4-B Manual Review Preparation

STEP 4-A의 [36개 deterministic 후보](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_review_candidates.csv)를 순서·유형 변경 없이 사용했다. 선정 유형은 최다 YuNet 결측 8개, 최장 연속 결측 8개, 최다 Context 결측 8개, 고립 단일 결측 6개, 결측 0 정상 참조 6개다. Train 20개/val 16개, drowsy 18개/not-drowsy 18개이며 `n_246`과 정상 참조 6개가 포함된다.

후보 영상마다 최대 5개 canonical frame만 선택했다. 연속 결측은 대표 run 전·시작·중간·끝·후, 전체 결측은 k000·025·050·075·099, Context 결측은 초·중·후 결측 slot과 가능한 저장 crop 비교, 고립 결측은 앞·결측·뒤, 정상 참조는 k000·050·099를 우선한다. 실제로 원본 영상 36개에서 고유 source frame 140개를 seek·decode하고 source index, frame 크기, timestamp metadata와 대조했다. 선택된 성공 frame에만 기존 YuNet bbox를 겹쳤으며 기존 Context JPEG만 읽었다. YuNet·Dlib 재실행과 Context crop 재생성은 없었다.

[시각 검토 팩](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/)에는 개별 검토 sheet 36장, 4개 영상씩 묶은 contact sheet 9장, [검토 인덱스](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_review_index.csv), [수동 판정 CSV](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review.csv), [검토 안내](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/STEP4B_MANUAL_REVIEW_GUIDE.md)가 있다. 수동 판정 CSV는 최초 생성 시 36행 모두 미판정이었으며 이후 사용자 확정 판정을 기록했다.

이 검토 팩은 원본 영상 전체나 YuNet 결측 원인에 대한 자동 판정이 아니다. 수동 결과는 다음 절에서 분리해 기록한다.

## STEP 4-B Manual Visual Review Results

사용자가 ChatGPT 보조로 contact sheet 9장의 36개 목적 선정 후보를 시각적으로 검토하고 review_id별 판정을 확정했다. Codex는 이미지를 재판독하지 않고 제공된 18개 수동 필드만 [수동 판정 CSV](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review.csv)에 기록했다. 원래 자동 metadata는 변경하지 않았다. Boolean schema에 맞지 않는 `PARTIAL`은 `TRUE`로 적고 제한된 가시성은 원문 note로 보존했으며, 정상 참조의 `N.A.`는 해당 boolean field를 빈 값으로 두었다. 모든 변환과 수정 전후 SHA-256은 [provenance](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review_provenance.json)에 있다.

| 수동 category / decision | 후보 수 |
|---|---:|
| EXPECTED_DETECTOR_MISS | 13 |
| LIKELY_FALSE_NEGATIVE | 12 |
| MIXED_MISSING_PATTERN | 5 |
| NORMAL_REFERENCE | 6 |
| SOURCE_CONTENT_ISSUE / POSSIBLE_PIPELINE_ISSUE / UNCERTAIN | 0 / 0 / 0 |
| PASS / CHECK_NEEDED / INVESTIGATE | 19 / 17 / 0 |

선정된 full-data 사례에서 강한 yaw/profile과 head pitch·roll, 일부 frame-edge·얼굴 잘림·가림·motion blur가 관찰됐다. 그러나 얼굴 주요 구조가 충분히 보이는 likely false-negative와 혼합 구간도 공존한다. `n_246`은 100/100 YuNet 결측, `FULL_CLIP_MISSING`, `EXPECTED_DETECTOR_MISS`로 판정됐으며 사용자 판정상 source corruption이나 pipeline issue는 아니다. 긴 run에서는 자세 난이도와 얼굴이 보이는 연속 미검출이 모두 보고됐고, 고립 miss에서는 일시적 blur와 앞뒤 frame에 비해 특별한 난이도가 보이지 않는 실패가 모두 보고됐다. 정상 참조 6개에서 저장 bbox·Context JPEG의 명백한 이상은 보고되지 않았다. Pipeline/storage issue 의심은 0건이다.

이 36개는 STEP 4-A의 deterministic·purposeful 후보이지 전체 1,763개 영상의 무작위 표본이 아니다. 따라서 13/12/5 등의 category 비율을 전체 detector 오류 비율로 해석하거나, 모든 YuNet 결측을 profile 때문이라고 일반화하지 않는다. [수동 요약](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review_summary.json)과 [상세 보고서](../outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review/step4b_manual_review_report.txt)에 해석 범위와 결과를 남겼다.

STEP 4-B는 **COMPLETE**다. Missing policy는 **NOT SELECTED**, STEP 4-C는 **NOT STARTED**이며 test split은 **SEALED**다. 허용 결측 개수, 영상 제외, masking·replacement·sequence 구성은 이 단계에서 결정하지 않았다.
