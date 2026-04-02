from __future__ import annotations

import argparse
import os
from statistics import mean

from .agent import LLMAgent
from .env import ClipQualityEnv
from .generator import ClipMetaGenerator
from .ground_truth import GTStore
from .rubric import RubricState


def run_training(episodes: int, model_name: str | None = None) -> dict[str, float]:
    rubric = RubricState("state/rubric.json")
    gt = GTStore("data/seed_gt.json", "state/ground_truth.json")
    generator = ClipMetaGenerator(real_clips_path=os.environ.get("REAL_CLIPS_MANIFEST"))
    env = ClipQualityEnv(gt, rubric, generator, history_path="state/history.jsonl")
    agent = LLMAgent(model_name=model_name)

    easy_rewards: list[float] = []
    medium_rewards: list[float] = []
    hard_rewards: list[float] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if step == 0:
                easy_rewards.append(reward)
            elif step == 1:
                medium_rewards.append(reward)
            else:
                hard_rewards.append(reward)
            step += 1
        if ep % 10 == 0:
            print(
                f"[Episode {ep}] GT={env.gt_store.size()} Rubric=v{env.rubric.version} "
                f"LastHard={hard_rewards[-1]:.3f}"
            )

    return {
        "easy": mean(easy_rewards) if easy_rewards else 0.0,
        "medium": mean(medium_rewards) if medium_rewards else 0.0,
        "hard": mean(hard_rewards) if hard_rewards else 0.0,
        "gt_size": float(env.gt_store.size()),
        "rubric_version": float(env.rubric.version),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClipQualityEnv training loop.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes.")
    parser.add_argument("--model-name", type=str, default=None, help="Override MODEL_NAME.")
    args = parser.parse_args()

    summary = run_training(episodes=args.episodes, model_name=args.model_name)
    print("Training summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
