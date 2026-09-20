# Experiment 2 STEP 3 — Canonical Preprocessing

이 문서는 STEP 3-A 구현 설계·출력 계약과 STEP 3-B의 **실제 20-video pilot 결과**를 기록한다. STEP 3-A는 `IMPLEMENTED`, STEP 3-B는 사용자 제공 수동 검토까지 `COMPLETE`, STEP 3-C는 `NOT STARTED`다. Test split은 `SEALED`다.

현재 milestone: STEP 1에서 2,074개 source 영상과 중복 없는 video-level train/val/test 1,452/311/311개를 확정했다. Subject-wise unseen-driver 독립성은 매핑 부재로 보장하지 않는다. STEP 2는 HOG 152/160(95.0%, 약 372 ms) 대 YuNet 157/160(98.125%, 약 39 ms)의 통제 비교를 거쳐 `FULLY CLOSED`됐다. RAW/M05/M10/M15 ROI의 Dlib68 성공은 각각 157/157이며 clipping metadata 오류는 저장 ROI 628/628개가 일치한 `REPORTING_ONLY_BUG`였다. 최종 frozen 정책은 YuNet, RAW bbox Dlib68, EAR/MAR/Head Pose, SQUARE_M10 RGB 224×224 및 ImageNet mean padding이다. Context geometry 선택은 모델 정확도 우위 판정이 아니다.

## 1. Objective

Frozen video-level split의 train 1,452개·val 311개 영상에 적용할 단일 canonical pipeline을 구현한다. Test 311개는 sealed이며 구현·검증 단계에서 열거나 처리하지 않는다.

## 2. Frozen STEP 2 Policy

Face detector는 YuNet 단독, 유효 bbox 중 최대 면적 얼굴을 선택하며 HOG fallback·tracking은 없다. Landmark는 YuNet RAW bbox를 입력받는 Dlib68이다. EAR/MAR/Head Pose는 STEP 2 helper를 재사용한다. Context crop은 bbox 각 축 양쪽 10% margin 후 중심 유지 square, ImageNet mean padding, RGB uint8 224×224이다. ResNet18/VGG16, threshold, event rule은 구현하지 않았다.

## 3. Canonical 10 Hz Timeline

모든 정상 open·유효 FPS 영상에 대해 target timestamp `0.0, 0.1, …, 9.9`의 100 row를 만든다. `source_frame_index = floor(target_timestamp_sec × source_fps + 0.5)`이며, `actual_timestamp_sec = source_frame_index / source_fps`, `timestamp_error_ms = (actual-target) × 1000`으로 부호를 보존한다. 인덱스가 frame count 밖이면 clamp하지 않고 `SOURCE_INDEX_OUT_OF_RANGE`; 순차 decode 실패는 `FRAME_DECODE_FAILED`다. FPS가 0·음수·NaN·Inf면 영상 단위 실패다. Target source index를 먼저 계산하고 한 번 순차 decode하며 중복 인덱스는 검출·landmark 결과를 재사용한다.

## 4. 32 Context Slots

검출 전에 `floor(i × 99 / 31 + 0.5), i=0..31`로 끝점 포함 32개 고유 오름차순 slot을 결정한다. 검출 실패 시 인접 slot으로 대체하지 않는다. `context_selected=True`, `context_crop_available=False`로 남긴다. 일반 frame 100개 중 crop JPEG는 선택된 32개에서 YuNet 검출 성공 시에만 생성한다.

## 5. Behavior Branch

각 source frame의 YuNet 검출은 한 번만 수행하고 그 RAW bbox를 Dlib68에 제공한다. 성공 landmark에서 EAR·MAR와 raw Euler pitch/yaw/roll, 분석용 `pitch_centered_candidate`를 기록한다. Head-up/down 부호와 eye-closed/yawn/PERCLOS/nod 이벤트는 미확정이므로 생성하지 않는다. Label은 manifest·보고서·pilot 층화에만 사용하고 detector·feature 함수에 전달하지 않는다.

## 6. Context Branch

YuNet bbox가 있으면 Dlib68 성공 여부와 무관하게 SQUARE_M10 crop을 만들 수 있다. STEP 2-D의 `square_crop_roi`, `crop_geometry`, `crop_preview`, `imagenet_padding_bgr`를 재사용한다. RGB 메모리 배열을 JPEG로 쓸 때는 BGR로 변환한다. 파일명은 `context_crops/ctx_{context_slot:02d}_k{canonical_index:03d}.jpg`; 정규화 float tensor는 저장하지 않는다. Pilot의 선택적 debug preview는 최대 8개 영상에서 Context slot 0·15·31의 원본 bbox와 crop을 나란히 보여주며 정책 재선택용이 아니다.

## 7. Missing-value Philosophy

YuNet 실패면 bbox·landmark·EAR·MAR·pose는 NaN이다. Dlib68 실패면 bbox만 유지하고 landmark·EAR·MAR·pose는 NaN이며 Context crop은 가능하다. Pose만 실패하면 EAR/MAR는 유지하고 pose 숫자만 NaN이다. HOG·3DDFA fallback, 이전/다음 bbox, interpolation, forward/backward fill, nearest-valid Context 대체는 없다. `decode_status`, `detector_status`, `landmark_status`, `pose_status`, `context_status`와 `failure_flags`를 분리한다. 일부 frame의 검출 실패만으로 영상 bundle을 실패로 취급하지 않는다.

## 8. Artifact Schema

영상당 `frames.csv` 100 row에는 identity, target/source index·timestamp·FPS, decode 상태, YuNet confidence/count/bbox, Dlib 상태, EAR/MAR, raw/centered pitch·yaw·roll, Context 선택/가용성·파일 경로·padding·얼굴 면적, 단계별 시간·실패 flag가 들어간다. `landmarks.npz`에는 float32 `landmarks[100,68,2]`, bool `landmark_valid[100]`, `canonical_index[100]`, source index가 포함된다. 실패 좌표는 NaN이다. `video_summary.json`에는 유효/결측 수, duplicate index, Context 수, 처리 시간과 hash를 기록한다. Global `canonical_frames.csv`·`canonical_videos.csv`는 검증된 완료 bundle에서만 집계한다.

## 9. Per-video Atomic Bundle

Pilot은 `data/interim/preprocessing_v2/canonical_pilot/`, full은 `data/interim/preprocessing_v2/canonical/`에 각각 `per_video/train|val/<video_id>/` bundle을 둔다. `_tmp/<video_id>_<uuid>/`에 `frames.csv`, `landmarks.npz`, `context_crops/`, `video_summary.json`, `COMPLETE.json`을 만들고 integrity 검증 후 같은 파일시스템에서 최종 경로로 이동한다. 기존 final bundle은 자동 덮어쓰지 않는다. Global metadata는 각 root의 `metadata/`, 보고서는 `outputs/preprocessing_v2/canonical_preprocessing/pilot|full/`에 둔다.

## 10. Provenance / Hashes

`policy_hash`는 sampling·Context 선택·YuNet threshold 및 모델 SHA256·face selection·Dlib 모델 SHA256·feature version·ROI·crop·padding·resize·JPEG 품질을 반영한다. 출력 경로·모드·로그 옵션은 포함하지 않는다. 별도 `run_config_hash`는 모드·출력·manifest·검증 설정을 포함한다. YuNet/Dlib 파일 SHA256은 run 시작 시 한 번 계산한다. Source fingerprint는 resolved path·file size·mtime_ns이다. `run_metadata.json`에는 Python/NumPy/OpenCV/Dlib 버전, 가능하면 Git commit, 두 hash, 모델 hash, Context index와 train/val allowlist·test 미처리 상태를 남긴다.

## 11. Resume Policy

`--resume`은 final bundle·`COMPLETE.json` 존재, 동일 policy hash와 source fingerprint, 기본 integrity 통과를 모두 만족할 때만 skip한다. Hash·fingerprint 불일치, 불완전/손상 bundle은 conflict로 보고하고 덮어쓰지 않는다. Existing run metadata와 다른 정책·실행 설정도 거부한다.

## 12. Integrity Rules

완료 영상마다 100 row, canonical index 0..99, 단조 target timestamp/source index, 32개 Context 선택·slot 0..31, landmark NPZ shape/dtype, 실패 좌표 NaN을 검사한다. Crop 가용은 선택 slot·YuNet 성공을 전제로 하며 JPEG 파일 존재를 확인하고 pilot에서는 decode·224×224×3을 확인한다. Pose 실패 숫자는 NaN, Dlib 실패 EAR/MAR는 NaN이어야 한다. Global 집계에서 test/unknown split과 중복 video_id를 거부한다.

## 13. STEP 3-B Pilot Design

`scripts/prepare_canonical_pilot_manifest.py`는 metadata만 읽고 train/val × drowsy/not_drowsy 각 5개, 총 20개를 선택한다. 각 층의 video_id를 정렬하고 전체 범위의 균등 위치를 half-up 방식으로 고른다. STEP 3-B에서 영구 `data/metadata/experiment2_step3_pilot_manifest.csv`를 생성해 사용했으며 이후 selection을 바꾸지 않는다. Pilot은 correctness·schema·색상·missing·resume·atomicity 검증용이며 crop 정책 재탐색용이 아니다.

## 14. STEP 3-C Full Run Plan

Pilot 결과 검증 후 단일 프로세스 baseline으로 train/val 1,763개를 처리한다. Full root는 pilot root와 분리하고 pilot bundle을 재사용하지 않는다. Metadata-only full dry-run은 176,300 frame row·56,416 Context slot을 계획값으로 확인했다. 이 값은 처리 성과가 아니다.

## 15. Test Split Protection

Manifest와 metadata는 train/val만 허용한다. Test row는 raw path resolution 전에 거부하고, test 영상 open·decode·feature 생성·정책 선택·model selection에 사용하지 않는다. **TEST SPLIT SEALED.**

## 16. Limitations

STEP 3-B의 지정된 20개 영상만 실제 전처리했다. Full train/val과 test는 처리하지 않았다. Video-level split은 subject-wise 독립성을 보장하지 않는다. 20-video pilot의 검출·padding 분포를 전체 1,763개 영상의 품질 분포로 일반화하지 않는다. EAR/MAR·Head Pose의 행동 threshold와 Context model accuracy는 이후 단계에서 검증한다.

## STEP 3-B Actual Pilot Run

결정적 `data/metadata/experiment2_step3_pilot_manifest.csv`는 train/val × drowsy/not_drowsy 각 5개, 총 20개 고유 video_id다. 원본 경로 20/20개가 존재했고 test는 0개였다. 실행 전 `canonical_pilot/`과 full `canonical/`은 모두 없었다. `--mode pilot`로 실제 영상을 한 번 처리했으며 full root는 생성하지 않았다. Frozen YuNet / Dlib68 RAW bbox / SQUARE_M10 RGB 224×224 ImageNet mean 정책은 변경하지 않았다.

| 범위·단계 | 실제 결과 |
|---|---:|
| 완료 bundle / processing failure | 20 / 0 |
| Canonical row / 정상 decode | 2,000 / 2,000 |
| Out-of-range / decode failed / 중복 source index | 0 / 0 / 0 |
| Source FPS / reported frame count | 25–30 FPS / 250–300 frame |
| 보고 duration / signed timestamp error | 10.000–10.027초 / 약 -20~+20 ms |
| YuNet 검출 성공 / 미검출 / multiple-face frame | 1,916 / 84 / 0 |
| Dlib68 성공 / 검출 후 실패 | 1,916 / 0 |
| EAR·MAR finite / Head Pose 성공 | 각각 1,916 / 1,916 |
| Context 선택 / crop 가용 / 결측 | 640 / 613 / 27 |
| Context padding 필요 | 50/613 crop |
| 저장 JPEG 엄격 검증 / cross-artifact mismatch | 613/613 / 0 |
| Orphan 임시 bundle / test row | 0 / 0 |

YuNet 성공률은 1,916/2,000(95.8%)이며, 84개 미검출은 bundle 실패가 아닌 data-quality event다. 선택된 Context slot 27개의 crop은 대체 생성하지 않았다. 검출 후 landmark 실패는 0개였다. Detector latency는 평균 38.37 ms, 중앙값 37.46 ms, p90 42.05 ms, p95 44.40 ms다. Raw pitch 중앙값은 약 -140.28°, centered candidate 중앙값은 약 0.09°였지만 물리적 head-up/down 부호는 이번에도 결정하지 않았다.

가용 Context crop의 padding 비율은 평균 0.00477, 중앙값 0, p95 0.02004, 최대 0.25537이었다. 얼굴 면적 비율은 평균 0.47641, 중앙값 0.47399, p95 0.52712였다. Padding 존재나 YuNet miss를 이유로 STEP 2 crop/detector 정책을 변경하지 않는다.

`frames.csv`는 완료 영상마다 100행, Context 선택은 각 32개다. `landmarks.npz`의 `(100,68,2)` float32 좌표와 bool 유효성·canonical index·CSV flag를 대조한 불일치는 0개다. 가용 JPEG 613개를 전부 decode해 `uint8` 224×224×3과 CSV 파일명·개수를 확인했으며 오류 0개다. 20개 `COMPLETE.json`의 policy hash는 모두 `29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721`로 동일하고 source fingerprint가 있다. Global train/val CSV의 test row는 0개다.

첫 실행 후 동일 manifest에서 `--resume`을 한 번 실행해 신규 처리 0, 건너뜀 20, 실패·충돌 0, inference용 영상 open 0, detector frame 0을 확인했다. Resume가 첫 실행의 report와 생성 시각을 덮어쓰지 않도록 구현상 provenance 문제를 최소 수정하고 synthetic 회귀 테스트를 추가했다. 영상 20개를 재처리하지 않았다.

Pilot wall time은 run metadata 생성부터 마지막 bundle 완료까지 약 107.21초다. 영상별 처리 시간은 평균 5.045초, 중앙값 4.931초였다. 이를 1,763개 영상에 단순 선형 적용한 **simple projected runtime**은 평균 기준 약 2.47시간, 중앙값 기준 약 2.42시간이다. JPEG·집계 등 모든 overhead를 보장하는 full runtime 추정치는 아니다.

자동 [pilot 보고서](../outputs/preprocessing_v2/canonical_preprocessing/pilot/pilot_preprocessing_report.txt)와 [integrity 보고서](../outputs/preprocessing_v2/canonical_preprocessing/pilot/pilot_integrity_report.txt)는 오류 0개로 기록됐다. [시각 구현 검토 pack](../outputs/preprocessing_v2/canonical_preprocessing/pilot/visual_review/)은 train/val × drowsy/not_drowsy를 균형 있게 포함한 12개 영상, 36개 원본-bbox/저장-JPEG 비교 표본, 6장 sheet다. 사용자가 기존 6장 sheet의 frame/person alignment, RGB/BGR, JPEG, SQUARE_M10 구현 이상이 없다고 확인했다. 이 결론은 사용자 제공 시각 검토 결과이며 자동 검사 결과로 위장하지 않는다. STEP 3-C full 처리는 시작하지 않았다.

### STEP 3-B Missing-Only Context Review Pack

640개 Context 선택 slot 중 저장 crop이 없는 27개 row만 분리했다. 실제 필드명은 `context_selected=True`, `context_crop_available=False`, `context_slot`, `canonical_index`, `source_frame_index`다. Global `canonical_frames.csv`와 완료 bundle의 `frames.csv`를 대조하고, 저장된 source index의 원본 frame만 순차 decode하여 **기존 crop·feature·검출을 재생성하지 않았다**. 결과는 [별도 missing-only 디렉터리](../outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/)에 생성했고 기존 pilot 산출물은 수정하지 않았다.

| 결측 범위 | 실제 결과 |
|---|---:|
| 전체 Context 선택 / crop 결측 | 640 / 27 (4.21875%) |
| Train / validation | 12 / 15 |
| Drowsy / not_drowsy | 0 / 27 |
| 영상별 | `n_726` 12, `n_262` 7, `n_999` 5, `n_997` 3 |
| 정상 decode 후 YuNet 미검출 | 27/27 |
| YuNet 성공인데 crop만 없는 행 | 0 |
| Dlib68 / Head Pose 성공 | 각 0/27 (YuNet bbox 없음) |
| 개별 이미지 / contact sheet / 원본 frame decode 누락 | 27 / 7 / 0 |

모든 행의 자동 원인 후보는 저장 상태에 근거한 `DETECTOR_MISS`다. 자동 판정만으로 실제 얼굴이 보이는 false negative인지, 큰 profile 등으로 검출이 어려운 상황인지 알 수 없다. 근처 검출 성공 frame의 yaw/edge 값은 보조 힌트로만 표기했다. 기존 자동 summary와 report는 생성 당시 기록으로 보존했다.

### STEP 3-B Manual Missing-Context Review

사용자가 27개 결측 이미지를 직접 검토한 판정을 [수동 CSV](../outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/missing_context_manual_review.csv)의 수동 열에 기록했다. `review_id`는 행 순서가 아니라 실제 CSV ID이며, 12~23의 12건은 강한 측면/profile, frame edge 방향에 가까운 얼굴, 일부 코 끝 clipping으로 YuNet miss가 비교적 납득 가능한 `EXPECTED_DETECTOR_MISS/PASS`다. 후속 사용자 판독에 따라 이 그룹의 `face_near_edge=TRUE`로 정정했다. 나머지 15건에도 상당수 yaw/profile pose가 있지만 얼굴과 주요 특징은 육안으로 충분히 보여 `LIKELY_FALSE_NEGATIVE/CHECK_NEEDED`를 유지했다. 개별 extreme pose 여부는 확정하지 않아 `extreme_pose`를 빈 값으로 두었으며, 이는 frontal pose 판정이 아니다. Storage/pipeline issue와 uncertain은 각 0건이다.

[수동 검토 summary](../outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/missing_context_manual_review_summary.json)에는 `USER_PROVIDED_VISUAL_REVIEW`, `DIRECT_VISUAL_INSPECTION`, revision 2, pose-related 관찰, 정책 불변, 수정 전후 CSV SHA-256 및 자동 열 보존 여부를 기록했다. [별도 보고서](../outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/missing_context_manual_review_report.txt)의 correction section은 기존 12/15 판정을 유지하면서 과도하게 단순했던 pose 설명을 대체한다. [정정 provenance](../outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review/manual_review_interpretation_correction.md)는 정정 이유와 해시를 별도로 남긴다. 기존 자동 summary/report는 덮어쓰지 않았다. 이는 사용자 판정을 전사한 것이며 Codex가 이미지를 다시 판독하거나 detector를 재실행한 결과가 아니다.

20-video automatic pilot integrity와 사용자 제공 구현 시각 검토 및 27건 missing-only 검토가 완료되어 **STEP 3-B는 COMPLETE**다. Canonical preprocessing 구현의 검사 범위에서 frame/person alignment, RGB/BGR, JPEG, SQUARE_M10 오류 증거는 없었다. 27건 전반에서 yaw/profile 관련 결측 패턴이 관찰됐지만 20-video pilot을 전체 데이터셋에 일반화하지 않는다. 15건의 likely YuNet false negative는 data-quality 문제로 남으며 STEP 4에서 전체 train/val missing rate, 영상별 집중도·연속 결측·pose 관련 패턴·label 간 격차를 정량화한다. YuNet 교체, HOG fallback, threshold 조정, Context 대체는 하지 않고 STEP 2의 YuNet·Dlib68 RAW bbox·SQUARE_M10 정책을 유지한다. STEP 3-C는 `NOT STARTED`, test split은 `SEALED`다.

### Remaining Open Items and Next Step

Head Pose의 물리적 up/down 부호 규약, EAR closure·PERCLOS·MAR/yawn·head-drop/nod 행동 규칙, detector/Context 결측 처리 정책은 아직 확정하지 않았다. ResNet18/VGG16 CNN과 LSTM도 학습하지 않았다. 다음 단계 STEP 3-C는 train 1,452개와 val 311개, 총 1,763개 영상의 canonical materialization이며 test는 0개다. 이번 정정 작업에서는 STEP 3-C와 STEP 4를 시작하지 않았다.
