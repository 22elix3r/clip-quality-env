import json

import pytest

from clip_quality_env.env import ClipQualityEnv
from clip_quality_env.generator import ClipMetaGenerator
from clip_quality_env.real_clips import load_real_clip_manifest
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


def test_load_manifest_buckets_by_difficulty(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(
        manifest,
        [
            {**_clip("real_easy_1"), "difficulty": "easy"},
            {**_clip("real_med_1", motion_score=0.30), "difficulty": "medium"},
            {**_clip("real_hard_1", face_area_ratio=0.24, motion_score=0.32, audio_snr_db=18.0), "difficulty": "hard"},
        ],
    )

    pools = load_real_clip_manifest(str(manifest), rubric)
    assert len(pools["easy"]) == 1
    assert len(pools["medium"]) == 1
    assert len(pools["hard"]) == 1
    assert pools["easy"][0]["clip_id"] == "real_easy_1"


def test_load_manifest_derives_difficulty_when_missing(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(manifest, [_clip("real_auto_easy")])

    pools = load_real_clip_manifest(str(manifest), rubric)
    assert len(pools["easy"]) == 1
    assert pools["easy"][0]["clip_id"] == "real_auto_easy"


def test_load_manifest_rejects_invalid_difficulty(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(manifest, [{**_clip("real_bad"), "difficulty": "expert"}])

    with pytest.raises(ValueError, match="Invalid difficulty"):
        load_real_clip_manifest(str(manifest), rubric)


def test_generator_prefers_real_pool_and_falls_back(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(manifest, [{**_clip("real_easy_1"), "difficulty": "easy"}])

    gen = ClipMetaGenerator(seed=9, real_clips_path=str(manifest))
    easy_clip = gen.sample("easy", rubric)
    medium_clip = gen.sample("medium", rubric)

    assert easy_clip["clip_id"] == "real_easy_1"
    assert str(medium_clip["clip_id"]).startswith("syn_med_")
    assert gen.real_clip_pool_sizes() == {"easy": 1, "medium": 0, "hard": 0}


def test_env_state_includes_real_clip_pool_info(tmp_path):
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(
        manifest,
        [
            {**_clip("real_easy_1"), "difficulty": "easy"},
            {**_clip("real_med_1", motion_score=0.30), "difficulty": "medium"},
            {**_clip("real_hard_1", face_area_ratio=0.24, motion_score=0.32, audio_snr_db=18.0), "difficulty": "hard"},
        ],
    )

    env = ClipQualityEnv(real_clips_path=str(manifest), history_path=str(tmp_path / "history.jsonl"))
    obs = env.reset()
    state = env.state()

    assert obs.step == 1
    assert obs.clip_metadata.clip_id == "real_easy_1"
    assert state["real_clips_path"] == str(manifest)
    assert state["real_clip_pool_sizes"] == {"easy": 1, "medium": 1, "hard": 1}


def test_generator_real_clip_cycle_is_deterministic(tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    manifest = tmp_path / "real_clips.jsonl"
    _write_jsonl(
        manifest,
        [
            {**_clip("real_easy_a"), "difficulty": "easy"},
            {**_clip("real_easy_b"), "difficulty": "easy"},
        ],
    )

    gen = ClipMetaGenerator(seed=1, real_clips_path=str(manifest))
    seen = [gen.sample("easy", rubric)["clip_id"] for _ in range(3)]
    assert seen[0] != seen[1]
    assert seen[2] == seen[0]
