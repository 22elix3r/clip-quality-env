from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClipMetadata(BaseModel):
    """Single clip metadata record."""

    clip_id: str
    duration_s: float = Field(..., ge=0.0)
    fps: int = Field(..., ge=1)
    resolution: str
    face_area_ratio: float = Field(..., ge=0.0, le=1.0)
    face_confidence: float = Field(..., ge=0.0, le=1.0)
    head_pose_yaw_deg: float
    head_pose_pitch_deg: float
    motion_score: float = Field(..., ge=0.0, le=1.0)
    bg_complexity: str
    bg_complexity_score: float = Field(..., ge=0.0, le=1.0)
    mouth_open_ratio: float = Field(..., ge=0.0, le=1.0)
    blink_rate_hz: float = Field(..., ge=0.0)
    audio_snr_db: float
    transcript_word_count: int = Field(..., ge=0)
    transcript_confidence: float = Field(..., ge=0.0, le=1.0)
    lighting_uniformity: float = Field(..., ge=0.0, le=1.0)
    occlusion_present: bool
    environment_tag: str
    framing: str = Field(..., min_length=1)


class HistoryItem(BaseModel):
    """Single prior step outcome in an episode."""

    step: int = Field(..., ge=1, le=3)
    clip_id: str
    label: Literal["KEEP", "BORDERLINE", "REJECT"]
    reward: float = Field(..., ge=0.0, le=1.0)


class Observation(BaseModel):
    """OpenEnv observation payload."""

    step: int = Field(..., ge=1, le=3)
    rubric_version: int = Field(..., ge=1)
    rubric_summary: str
    clip_metadata: ClipMetadata
    history: list[HistoryItem] = Field(default_factory=list)


class Action(BaseModel):
    """Agent action payload."""

    label: Literal["KEEP", "BORDERLINE", "REJECT"]
    reasoning: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class Reward(BaseModel):
    """Reward decomposition payload."""

    total: float = Field(..., ge=0.0, le=1.0)
    format_score: float = Field(..., ge=0.0, le=1.0)
    label_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_score: float = Field(..., ge=0.0, le=1.0)
