from clip_quality_env.real_clips import derive_clip_difficulty
from clip_quality_env.rubric import RubricState


def _base_clip():
    return {
        "clip_id": "x",
        "duration_s": 8.0,
        "fps": 24,
        "resolution": "1280x720",
        "face_area_ratio": 0.38,
        "face_confidence": 0.92,
        "head_pose_yaw_deg": 8.0,
        "head_pose_pitch_deg": 2.0,
        "motion_score": 0.10,
        "bg_complexity": "simple_room",
        "bg_complexity_score": 0.08,
        "mouth_open_ratio": 0.40,
        "blink_rate_hz": 0.25,
        "audio_snr_db": 24.0,
        "transcript_word_count": 40,
        "transcript_confidence": 0.93,
        "lighting_uniformity": 0.78,
        "occlusion_present": False,
        "environment_tag": "podcast_studio",
        "framing": "front",
    }


def test_derive_clip_difficulty_easy(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    clip = _base_clip()
    assert derive_clip_difficulty(clip, rubric) == "easy"


def test_derive_clip_difficulty_medium(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    clip = _base_clip()
    clip["motion_score"] = 0.30
    assert derive_clip_difficulty(clip, rubric) == "medium"


def test_derive_clip_difficulty_hard(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    clip = _base_clip()
    clip["motion_score"] = 0.30
    clip["face_area_ratio"] = 0.24
    clip["audio_snr_db"] = 17.0
    assert derive_clip_difficulty(clip, rubric) == "hard"
