from clip_quality_env.grader import grade
from clip_quality_env.ground_truth import GTStore
from clip_quality_env.rubric import RubricState


def _clip(clip_id: str = "clip_0001") -> dict:
    return {
        "clip_id": clip_id,
        "duration_s": 8.0,
        "fps": 24,
        "resolution": "1280x720",
        "face_area_ratio": 0.38,
        "face_confidence": 0.91,
        "head_pose_yaw_deg": 10.0,
        "head_pose_pitch_deg": 0.0,
        "motion_score": 0.10,
        "bg_complexity": "solid_dark",
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


def test_perfect_keep_scores_high(tmp_path):
    gt = GTStore(seed_path="data/seed_gt.json", state_path=str(tmp_path / "gt.json"))
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    action = {
        "label": "KEEP",
        "reasoning": (
            "face_area_ratio is high and audio_snr_db is clear, "
            "so keep this clip. motion_score is low."
        ),
        "confidence": 0.9,
    }
    reward = grade(action, _clip("clip_0001"), rubric, gt)
    assert reward.total >= 0.8


def test_partial_credit_borderline(tmp_path):
    gt = GTStore(seed_path="data/seed_gt.json", state_path=str(tmp_path / "gt.json"))
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    clip = _clip("unknown_border")
    clip["face_area_ratio"] = 0.24
    clip["motion_score"] = 0.29
    clip["audio_snr_db"] = 18.0
    action = {
        "label": "KEEP",
        "reasoning": "face_area_ratio and motion_score are borderline but acceptable.",
        "confidence": 0.75,
    }
    reward = grade(action, clip, rubric, gt)
    assert 0.20 <= reward.label_score <= 0.25
