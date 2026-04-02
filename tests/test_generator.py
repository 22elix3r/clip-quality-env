from clip_quality_env.generator import ClipMetaGenerator
from clip_quality_env.rubric import RubricState


def test_generator_returns_required_schema(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    gen = ClipMetaGenerator(seed=13)
    clip = gen.sample("easy", rubric)
    required = {
        "clip_id",
        "duration_s",
        "fps",
        "resolution",
        "face_area_ratio",
        "face_confidence",
        "head_pose_yaw_deg",
        "head_pose_pitch_deg",
        "motion_score",
        "bg_complexity",
        "bg_complexity_score",
        "mouth_open_ratio",
        "blink_rate_hz",
        "audio_snr_db",
        "transcript_word_count",
        "transcript_confidence",
        "lighting_uniformity",
        "occlusion_present",
        "environment_tag",
        "framing",
    }
    assert required.issubset(set(clip.keys()))


def test_hard_clip_prefix(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    gen = ClipMetaGenerator(seed=21)
    clip = gen.sample("hard", rubric)
    assert str(clip["clip_id"]).startswith("syn_hard_")
