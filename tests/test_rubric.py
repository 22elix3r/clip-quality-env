from clip_quality_env.rubric import PerformanceWindow, RubricState


def test_derive_label_reject_on_occlusion(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    clip = {"occlusion_present": True, "face_confidence": 0.95}
    assert rubric.derive_label(clip) == "REJECT"


def test_derive_label_keep_for_clean_clip(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    clip = {
        "clip_id": "x",
        "face_area_ratio": 0.40,
        "face_confidence": 0.92,
        "head_pose_yaw_deg": 8.0,
        "motion_score": 0.10,
        "bg_complexity_score": 0.08,
        "audio_snr_db": 24.0,
        "duration_s": 8.0,
        "mouth_open_ratio": 0.41,
        "lighting_uniformity": 0.78,
        "occlusion_present": False,
    }
    assert rubric.derive_label(clip) == "KEEP"


def test_recalibrate_tightens_and_versions(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    before = rubric.version
    old_min = rubric.thresholds["face_area_ratio"]["keep_min"]
    rubric.recalibrate(PerformanceWindow(easy_accuracy=0.95, medium_accuracy=0.50, hard_accuracy=0.2), current_episode=50)
    assert rubric.version == before + 1
    assert rubric.thresholds["face_area_ratio"]["keep_min"] > old_min
