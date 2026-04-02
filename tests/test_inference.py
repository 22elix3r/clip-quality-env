import json

import inference
from clip_quality_env.env import ClipQualityEnv as RealClipQualityEnv
from clip_quality_env.ground_truth import GTStore
from clip_quality_env.models import Action
from clip_quality_env.rubric import RubricState


def _clip(clip_id: str, **overrides):
    clip = {
        "clip_id": clip_id,
        "duration_s": 8.0,
        "fps": 24,
        "resolution": "1280x720",
        "face_area_ratio": 0.38,
        "face_confidence": 0.91,
        "head_pose_yaw_deg": 10.0,
        "head_pose_pitch_deg": 0.0,
        "motion_score": 0.10,
        "bg_complexity": "simple_room",
        "bg_complexity_score": 0.08,
        "mouth_open_ratio": 0.42,
        "blink_rate_hz": 0.25,
        "audio_snr_db": 24.0,
        "transcript_word_count": 40,
        "transcript_confidence": 0.92,
        "lighting_uniformity": 0.77,
        "occlusion_present": False,
        "environment_tag": "podcast_studio",
        "framing": "front",
    }
    clip.update(overrides)
    return clip


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_run_baseline_emits_per_clip_multi_parameter_output(tmp_path, monkeypatch):
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(
        manifest,
        [
            {**_clip("real_easy_1"), "difficulty": "easy"},
            {**_clip("real_med_1", motion_score=0.30), "difficulty": "medium"},
            {**_clip("real_hard_1", face_area_ratio=0.22, audio_snr_db=14.0), "difficulty": "hard"},
        ],
    )

    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    gt_store = GTStore(seed_path="data/seed_gt.json", state_path=str(tmp_path / "gt.json"))

    def env_factory(real_clips_path=None):
        return RealClipQualityEnv(
            gt_store=gt_store,
            rubric=rubric,
            history_path=str(tmp_path / "history.jsonl"),
            real_clips_path=real_clips_path,
        )

    monkeypatch.setattr(inference, "ClipQualityEnv", env_factory)
    monkeypatch.setattr(
        inference,
        "get_agent_action",
        lambda observation, client, model_name: Action(
            label="KEEP",
            reasoning="face_area_ratio and audio_snr_db are strong; motion_score is stable.",
            confidence=0.88,
        ),
    )

    result = inference.run_baseline(
        max_episodes=2,
        real_clips_manifest=str(manifest),
        api_base_url="http://127.0.0.1:1/v1",
        model_name="offline-baseline",
        hf_token="dummy",
    )

    assert "per_clip_grades" in result
    assert "easy" not in result and "medium" not in result and "hard" not in result
    assert result["episodes_run"] == 2
    assert result["clips_graded"] == 6
    assert result["difficulty_counts"] == {"easy": 2, "medium": 2, "hard": 2}

    first = result["per_clip_grades"][0]
    assert first["difficulty"] == "easy"
    assert first["clip_id"] == "real_easy_1"
    assert "reward" in first and "total" in first["reward"]
    assert "parameter_grading" in first
    assert "face_area_ratio" in first["parameter_grading"]
    assert "parameter_summary" in first
    assert "score_reasons" in first
    assert set(first["score_reasons"].keys()) == {"high", "mixed", "low"}
    assert len(first["score_reasons"]["high"]) >= 1
    assert "face_area_ratio" in first["parameter_grading"]
    assert "reason" in first["parameter_grading"]["face_area_ratio"]


def test_grade_manifest_clips_grades_each_clip_once(tmp_path, monkeypatch):
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(
        manifest,
        [
            {**_clip("real_easy_1"), "difficulty": "easy"},
            {**_clip("real_med_1", motion_score=0.30), "difficulty": "medium"},
            {**_clip("real_hard_1", face_area_ratio=0.22, audio_snr_db=14.0), "difficulty": "hard"},
        ],
    )

    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    gt_store = GTStore(seed_path="data/seed_gt.json", state_path=str(tmp_path / "gt.json"))
    monkeypatch.setattr(
        inference,
        "get_agent_action",
        lambda observation, client, model_name: Action(
            label="BORDERLINE",
            reasoning="face_confidence and audio_snr_db are mixed; motion_score is acceptable.",
            confidence=0.81,
        ),
    )

    result = inference.grade_manifest_clips(
        manifest_path=str(manifest),
        api_base_url="http://127.0.0.1:1/v1",
        model_name="offline-baseline",
        hf_token="dummy",
        rubric=rubric,
        gt_store=gt_store,
    )

    assert result["mode"] == "manifest_clip_grading"
    assert result["clips_graded"] == 3
    assert result["difficulty_counts"] == {"easy": 1, "medium": 1, "hard": 1}

    rows = result["per_clip_grades"]
    assert [row["clip_id"] for row in rows] == ["real_easy_1", "real_med_1", "real_hard_1"]
    for row in rows:
        assert row["step"] == 1
        assert "reward" in row and "total" in row["reward"]
        assert "parameter_grading" in row
        assert "score_reasons" in row


def test_run_baseline_respects_max_difficulty_prefix(tmp_path, monkeypatch):
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(
        manifest,
        [
            {**_clip("real_easy_1"), "difficulty": "easy"},
            {**_clip("real_med_1", motion_score=0.30), "difficulty": "medium"},
            {**_clip("real_hard_1", face_area_ratio=0.22, audio_snr_db=14.0), "difficulty": "hard"},
        ],
    )

    monkeypatch.setattr(
        inference,
        "get_agent_action",
        lambda observation, client, model_name: Action(
            label="BORDERLINE",
            reasoning="mixed quality signals with moderate confidence.",
            confidence=0.7,
        ),
    )

    result = inference.run_baseline(
        max_episodes=3,
        real_clips_manifest=str(manifest),
        api_base_url="http://127.0.0.1:1/v1",
        model_name="offline-baseline",
        hf_token="dummy",
        max_difficulty="medium",
    )

    assert result["max_difficulty"] == "medium"
    assert result["steps_per_episode"] == 2
    assert result["difficulty_counts"]["easy"] == 3
    assert result["difficulty_counts"]["medium"] == 3
    assert result["difficulty_counts"]["hard"] == 0
    assert all(item["difficulty"] in {"easy", "medium"} for item in result["per_clip_grades"])


def test_grade_manifest_clips_applies_label_calibration(tmp_path, monkeypatch):
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                **_clip("real_keep_signal", face_confidence=0.95, motion_score=0.05, audio_snr_db=26.0, occlusion_present=False),
                "difficulty": "easy",
            }
        ],
    )

    monkeypatch.setattr(
        inference,
        "get_agent_action",
        lambda observation, client, model_name: Action(
            label="BORDERLINE",
            reasoning="weak generic explanation",
            confidence=0.55,
        ),
    )

    result = inference.grade_manifest_clips(
        manifest_path=str(manifest),
        api_base_url="http://127.0.0.1:1/v1",
        model_name="offline-baseline",
        hf_token="dummy",
        max_difficulty="easy",
    )

    row = result["per_clip_grades"][0]
    assert row["model_predicted_label"] == "BORDERLINE"
    assert row["predicted_label"] in {"KEEP", "REJECT", "BORDERLINE"}
    if row["predicted_label"] != "BORDERLINE":
        assert row["label_calibrated"] is True
        assert row["calibration_note"]
