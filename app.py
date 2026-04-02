from __future__ import annotations

import os

from fastapi import FastAPI

from clip_quality_env.env import ClipQualityEnv
from clip_quality_env.models import Action, Observation


app = FastAPI(title="ClipQualityEnv", version="1.0.0")
env = ClipQualityEnv(real_clips_path=os.environ.get("REAL_CLIPS_MANIFEST"))


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "environment": "clip_quality_env"}


@app.post("/reset", response_model=Observation)
def reset() -> Observation:
    return env.reset()


@app.post("/step")
def step(action: Action) -> dict:
    obs, reward, done, info = env.step(action)
    return {
        "observation": None if done else obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return env.state()


@app.get("/render")
def render() -> dict[str, str]:
    return {"render": env.render()}
