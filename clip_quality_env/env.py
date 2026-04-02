from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

from .generator import ClipMetaGenerator
from .grader import grade
from .ground_truth import GTStore
from .models import Action, HistoryItem, Observation
from .rubric import PerformanceWindow, RubricState


class ClipQualityEnv:
    """OpenEnv-compatible Clip Quality environment."""

    difficulties = ("easy", "medium", "hard")

    def __init__(
        self,
        gt_store: GTStore | None = None,
        rubric: RubricState | None = None,
        generator: ClipMetaGenerator | None = None,
        history_path: str = "state/history.jsonl",
        real_clips_path: str | None = None,
    ) -> None:
        self.gt_store = gt_store or GTStore()
        self.rubric = rubric or RubricState()
        if generator is None:
            self.generator = ClipMetaGenerator(real_clips_path=real_clips_path)
        else:
            self.generator = generator
            if real_clips_path:
                self.generator.use_real_clips(real_clips_path, self.rubric)
        self.history_path = history_path
        self.real_clips_path = real_clips_path or self.generator.real_clips_path()
        self.episode_count = 0
        self.current_episode_history: list[dict[str, Any]] = []
        self._current_clips: list[dict[str, Any]] = []
        self._reward_windows: dict[str, deque[float]] = {
            "easy": deque(maxlen=50),
            "medium": deque(maxlen=50),
            "hard": deque(maxlen=50),
        }

    def reset(self) -> Observation:
        """Start a new episode and return step-1 observation."""
        self.episode_count += 1
        self.current_episode_history = []
        self._current_clips = [
            self.generator.sample("easy", self.rubric),
            self.generator.sample("medium", self.rubric),
            self.generator.sample("hard", self.rubric),
        ]
        return self._build_obs(step=1)

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Execute action and return (next_obs, reward, done, info).
        """
        parsed_action = action if isinstance(action, Action) else Action(**action)
        step_num = len(self.current_episode_history) + 1
        if step_num > 3:
            raise RuntimeError("Episode already complete. Call reset() before step().")

        clip = self._current_clips[step_num - 1]
        reward_breakdown = grade(parsed_action, clip, self.rubric, self.gt_store)
        expected_label = self.gt_store.lookup(str(clip["clip_id"])) or self.rubric.derive_label(clip)

        history_item = {
            "step": step_num,
            "difficulty": self.difficulties[step_num - 1],
            "clip_id": clip["clip_id"],
            "clip": clip,
            "action": parsed_action.model_dump(),
            "reward": reward_breakdown.total,
            "expected_label": expected_label,
        }
        self.current_episode_history.append(history_item)
        self._reward_windows[self.difficulties[step_num - 1]].append(float(reward_breakdown.total))

        done = step_num == 3
        info: dict[str, Any] = {
            "reward_breakdown": reward_breakdown.model_dump(),
            "expected_label": expected_label,
            "rubric_version": self.rubric.version,
        }
        if done:
            promoted = self._post_episode_update()
            info["gt_promoted"] = promoted
            next_obs = self._build_terminal_obs()
        else:
            next_obs = self._build_obs(step=step_num + 1)
        return next_obs, float(reward_breakdown.total), done, info

    def state(self) -> dict[str, Any]:
        """Return checkpointable environment state."""
        current_step = min(len(self.current_episode_history) + 1, 3)
        current_clip_id: str | None = None
        if self._current_clips and len(self.current_episode_history) < 3:
            current_clip_id = str(self._current_clips[len(self.current_episode_history)]["clip_id"])
        return {
            "episode_count": self.episode_count,
            "current_step": current_step,
            "gt_size": self.gt_store.size(),
            "rubric_version": self.rubric.version,
            "real_clips_path": self.real_clips_path,
            "real_clip_pool_sizes": self.generator.real_clip_pool_sizes(),
            "current_clip_id": current_clip_id,
            "episode_history": [
                {
                    "step": item["step"],
                    "difficulty": item["difficulty"],
                    "clip_id": item["clip_id"],
                    "label": item["action"]["label"],
                    "reward": item["reward"],
                }
                for item in self.current_episode_history
            ],
            "rubric_thresholds": self.rubric.get_thresholds_summary(),
        }

    def render(self) -> str:
        """Return concise human-readable state text."""
        state = self.state()
        return (
            f"Episode {state['episode_count']} | Step {state['current_step']}/3\n"
            f"GT Size: {state['gt_size']} | Rubric v{state['rubric_version']}\n"
            f"Current Clip: {state['current_clip_id']}"
        )

    def _build_obs(self, step: int) -> Observation:
        clip = self._current_clips[step - 1]
        history = [
            HistoryItem(
                step=item["step"],
                clip_id=item["clip_id"],
                label=item["action"]["label"],
                reward=float(item["reward"]),
            )
            for item in self.current_episode_history
        ]
        return Observation(
            step=step,
            rubric_version=self.rubric.version,
            rubric_summary=self.rubric.to_prompt_text(),
            clip_metadata=clip,
            history=history,
        )

    def _build_terminal_obs(self) -> Observation:
        last_clip = self._current_clips[-1] if self._current_clips else self.generator.sample("hard", self.rubric)
        history = [
            HistoryItem(
                step=item["step"],
                clip_id=item["clip_id"],
                label=item["action"]["label"],
                reward=float(item["reward"]),
            )
            for item in self.current_episode_history
        ]
        return Observation(
            step=3,
            rubric_version=self.rubric.version,
            rubric_summary=self.rubric.to_prompt_text(),
            clip_metadata=last_clip,
            history=history,
        )

    def _post_episode_update(self) -> bool:
        step3 = self.current_episode_history[-1]
        promoted = self.gt_store.try_promote(
            {
                "clip": step3["clip"],
                "action": step3["action"],
                "reward": step3["reward"],
                "expected_label": step3["expected_label"],
            },
            episode=self.episode_count,
        )
        if self.episode_count % 50 == 0:
            perf = self._get_performance_window()
            self.rubric.recalibrate(perf, current_episode=self.episode_count)
        self._log_episode(promoted)
        return promoted

    def _get_performance_window(self) -> PerformanceWindow:
        def avg(key: str) -> float:
            vals = self._reward_windows[key]
            if not vals:
                return 0.0
            return float(sum(vals) / len(vals))

        return PerformanceWindow(
            easy_accuracy=avg("easy"),
            medium_accuracy=avg("medium"),
            hard_accuracy=avg("hard"),
            easy_coverage=1.0 if self._reward_windows["easy"] else 0.0,
        )

    def _log_episode(self, gt_promoted: bool) -> None:
        os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
        episode_reward = (
            0.20 * self.current_episode_history[0]["reward"]
            + 0.35 * self.current_episode_history[1]["reward"]
            + 0.45 * self.current_episode_history[2]["reward"]
        )
        payload = {
            "episode": self.episode_count,
            "steps": [
                {
                    "step": item["step"],
                    "difficulty": item["difficulty"],
                    "clip_id": item["clip_id"],
                    "label": item["action"]["label"],
                    "expected_label": item["expected_label"],
                    "reward": item["reward"],
                }
                for item in self.current_episode_history
            ],
            "ep_reward": round(float(episode_reward), 6),
            "gt_promoted": gt_promoted,
            "gt_size": self.gt_store.size(),
            "rubric_version": self.rubric.version,
        }
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
