"""정책 해시에는 실행 경로가 들어가지 않아야 한다."""

from copy import deepcopy

from drowsiness_detection.preprocessing_v2.preprocessing_provenance import policy_payload, stable_hash


def test_policy_hash_semantics() -> None:
    config = {"sampling": {"hz": 10.0}, "context": {"margin_ratio": 0.1},
              "detector": {"model_path": "a", "score_threshold": 0.9},
              "landmark": {"roi_policy": "raw_yunet_bbox", "predictor_path": "b"},
              "outputs": {"pilot_root": "one"}, "logging": "INFO"}
    baseline = stable_hash(policy_payload(config, "yunet1", "dlib1"))
    changed = deepcopy(config)
    changed["outputs"]["pilot_root"] = "two"
    changed["logging"] = "DEBUG"
    assert stable_hash(policy_payload(changed, "yunet1", "dlib1")) == baseline
    changed["context"]["margin_ratio"] = 0.2
    assert stable_hash(policy_payload(changed, "yunet1", "dlib1")) != baseline
    assert stable_hash(policy_payload(config, "yunet2", "dlib1")) != baseline
    assert stable_hash(policy_payload(config, "yunet1", "dlib2")) != baseline
