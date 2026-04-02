#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI

from clip_quality_env.env import ClipQualityEnv
from clip_quality_env.grader import grade
from clip_quality_env.ground_truth import GTStore
from clip_quality_env.models import Action, Observation
from clip_quality_env.real_clips import DIFFICULTIES, load_real_clip_manifest
from clip_quality_env.rubric import RubricState


TIMEOUT_SECONDS = 20 * 60
BUFFER_SECONDS = 60
DEFAULT_HF_OPENAI_BASE_URL = "https://router.huggingface.co/v1"
DIFFICULTY_ORDER = ("easy", "medium", "hard")


def _load_client_config(
    api_base_url: str | None = None,
    model_name: str | None = None,
    hf_token: str | None = None,
) -> tuple[str, str, str]:
    resolved_api_base_url = api_base_url or os.environ.get("API_BASE_URL") or DEFAULT_HF_OPENAI_BASE_URL
    resolved_model_name = model_name or os.environ.get("MODEL_NAME")
    resolved_hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

    if not resolved_model_name:
        raise ValueError(
            "MODEL_NAME is required. Provide --model-name <huggingface-model-id> "
            "or set MODEL_NAME in the environment."
        )
    if not resolved_hf_token:
        raise ValueError(
            "HF_TOKEN (or OPENAI_API_KEY) is required. "
            "Provide --hf-token <token> or set HF_TOKEN in the environment."
        )
    return resolved_api_base_url, resolved_model_name, resolved_hf_token


def get_agent_action(observation: Observation, client: OpenAI, model_name: str) -> Action:
    system_prompt = (
        "You are a strict video clip quality classifier for LoRA training datasets.\n"
        "Your output must be decisive and evidence-based.\n"
        "Rules:\n"
        "1) Use KEEP when quality signals are mostly strong and no major reject signal is present.\n"
        "2) Use REJECT when any major reject signal is present (e.g., occlusion, very low face confidence, "
        "high motion/noisy audio) or reject signals dominate.\n"
        "3) Use BORDERLINE only when strong and weak signals are genuinely mixed.\n"
        "4) In reasoning, cite at least 3 concrete feature names with values from the clip and connect them "
        "to keep/reject thresholds.\n"
        "Return JSON with keys: label, reasoning, confidence."
    )
    user_prompt = (
        f"Rubric (v{observation.rubric_version}):\n{observation.rubric_summary}\n\n"
        f"Clip:\n{json.dumps(observation.clip_metadata.model_dump(), indent=2)}\n\n"
        f"History:\n{json.dumps([h.model_dump() for h in observation.history], indent=2)}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return Action(
        label=parsed.get("label", "BORDERLINE"),
        reasoning=parsed.get("reasoning", "Fallback reasoning."),
        confidence=float(parsed.get("confidence", 0.5)),
    )


def _difficulty_prefix(max_difficulty: str) -> list[str]:
    normalized = max_difficulty.strip().lower()
    if normalized not in DIFFICULTY_ORDER:
        raise ValueError(f"max_difficulty must be one of {DIFFICULTY_ORDER}")
    end_idx = DIFFICULTY_ORDER.index(normalized)
    return list(DIFFICULTY_ORDER[: end_idx + 1])


def _status_counts(clip_metadata: dict[str, Any], rubric: RubricState) -> dict[str, int]:
    counts = {"KEEP": 0, "BORDERLINE": 0, "REJECT": 0}
    for feature in rubric.thresholds:
        raw_value = clip_metadata.get(feature)
        if not isinstance(raw_value, (int, float)):
            continue
        status = rubric.get_feature_status(feature, float(raw_value))
        counts[status] = counts.get(status, 0) + 1
    if bool(clip_metadata.get("occlusion_present", False)):
        counts["REJECT"] += 1
    return counts


def _derive_deterministic_label(clip_metadata: dict[str, Any], rubric: RubricState) -> str:
    base = rubric.derive_label(clip_metadata)
    if base != "BORDERLINE":
        return base

    counts = _status_counts(clip_metadata, rubric)
    keep = int(counts["KEEP"])
    borderline = int(counts["BORDERLINE"])
    reject = int(counts["REJECT"])

    if reject >= max(2, keep + 1):
        return "REJECT"
    if keep >= 6 and reject == 0 and keep >= borderline + 2:
        return "KEEP"
    if keep >= reject + 3 and borderline <= 2:
        return "KEEP"
    if reject >= keep + 2:
        return "REJECT"
    return "BORDERLINE"


def _is_weak_reasoning(reasoning: str, rubric: RubricState) -> bool:
    text = reasoning.strip().lower()
    if len(text) < 80:
        return True
    mentioned = sum(1 for feature in rubric.thresholds if feature.lower() in text)
    if mentioned < 2:
        return True
    if "borderline" in text and ("keep" not in text and "reject" not in text):
        return True
    return False


def _synthesize_reasoning(
    *,
    label: str,
    parameter_grading: dict[str, dict[str, Any]],
    calibration_note: str | None = None,
) -> str:
    keep_reasons = [str(v["reason"]) for v in parameter_grading.values() if str(v.get("status")) == "KEEP"]
    mixed_reasons = [str(v["reason"]) for v in parameter_grading.values() if str(v.get("status")) == "BORDERLINE"]
    reject_reasons = [str(v["reason"]) for v in parameter_grading.values() if str(v.get("status")) == "REJECT"]

    if label == "KEEP":
        evidence = keep_reasons[:2]
        caution = mixed_reasons[:1]
        clauses = []
        if evidence:
            clauses.append("Strong keep evidence: " + " ".join(evidence))
        if caution:
            clauses.append("Minor caution: " + " ".join(caution))
        clauses.append("Overall KEEP because positive signals dominate and no reject-critical pattern dominates.")
    elif label == "REJECT":
        evidence = reject_reasons[:2] or mixed_reasons[:1]
        positive = keep_reasons[:1]
        clauses = []
        if evidence:
            clauses.append("Reject evidence: " + " ".join(evidence))
        if positive:
            clauses.append("Some positive signal exists: " + " ".join(positive))
        clauses.append("Overall REJECT because rejection signals outweigh acceptable quality indicators.")
    else:
        positive = keep_reasons[:1]
        caution = mixed_reasons[:2]
        negative = reject_reasons[:1]
        clauses = []
        if positive:
            clauses.append("Positive signal: " + " ".join(positive))
        if caution:
            clauses.append("Borderline signals: " + " ".join(caution))
        if negative:
            clauses.append("Negative signal: " + " ".join(negative))
        clauses.append("Overall BORDERLINE because signals are mixed and neither KEEP nor REJECT clearly dominates.")

    if calibration_note:
        clauses.append(calibration_note)
    return " ".join(clauses)


def _calibrate_action(
    raw_action: Action,
    clip_metadata: dict[str, Any],
    rubric: RubricState,
    parameter_grading: dict[str, dict[str, Any]],
) -> tuple[Action, str | None, str]:
    deterministic_label = _derive_deterministic_label(clip_metadata, rubric)
    counts = _status_counts(clip_metadata, rubric)

    calibrated_label = raw_action.label
    calibration_note: str | None = None

    if raw_action.label == "BORDERLINE" and deterministic_label != "BORDERLINE":
        calibrated_label = deterministic_label
        calibration_note = (
            f"Label calibrated from BORDERLINE to {deterministic_label} "
            f"using deterministic signal balance (keep={counts['KEEP']}, "
            f"borderline={counts['BORDERLINE']}, reject={counts['REJECT']})."
        )
    elif raw_action.label != deterministic_label:
        if deterministic_label == "REJECT" and counts["REJECT"] >= max(2, counts["KEEP"]):
            calibrated_label = "REJECT"
            calibration_note = (
                f"Label calibrated to REJECT because rejection signals dominate "
                f"(keep={counts['KEEP']}, reject={counts['REJECT']})."
            )
        elif (
            deterministic_label == "KEEP"
            and counts["KEEP"] >= 6
            and counts["REJECT"] == 0
            and counts["KEEP"] >= counts["BORDERLINE"] + 2
        ):
            calibrated_label = "KEEP"
            calibration_note = (
                f"Label calibrated to KEEP because strong keep signals dominate "
                f"(keep={counts['KEEP']}, borderline={counts['BORDERLINE']}, reject={counts['REJECT']})."
            )

    confidence = float(raw_action.confidence)
    if calibrated_label != raw_action.label:
        confidence = max(confidence, 0.72)

    reasoning = str(raw_action.reasoning)
    if _is_weak_reasoning(reasoning, rubric) or calibration_note is not None:
        reasoning = _synthesize_reasoning(
            label=calibrated_label,
            parameter_grading=parameter_grading,
            calibration_note=calibration_note,
        )

    calibrated_action = Action(
        label=calibrated_label,
        reasoning=reasoning,
        confidence=max(0.0, min(1.0, confidence)),
    )
    return calibrated_action, calibration_note, raw_action.label


def _feature_reason(feature: str, value: float, threshold: dict[str, float | str], status: str) -> str:
    mode = str(threshold["mode"])
    keep_min = float(threshold["keep_min"])
    keep_max = float(threshold["keep_max"])
    reject_min = float(threshold["reject_min"])
    reject_max = float(threshold["reject_max"])

    if mode == "higher":
        if status == "KEEP":
            return f"{feature}={value:.3g} is strong (>= {keep_min:.3g})."
        if status == "REJECT":
            return f"{feature}={value:.3g} is weak (< {reject_max:.3g})."
        return f"{feature}={value:.3g} is borderline (>= {reject_max:.3g} and < {keep_min:.3g})."

    if mode == "lower":
        if status == "KEEP":
            return f"{feature}={value:.3g} is stable (<= {keep_max:.3g})."
        if status == "REJECT":
            return f"{feature}={value:.3g} is too high (> {reject_min:.3g})."
        return f"{feature}={value:.3g} is borderline (> {keep_max:.3g} and <= {reject_min:.3g})."

    if status == "KEEP":
        return f"{feature}={value:.3g} is in the keep band [{keep_min:.3g}, {keep_max:.3g}]."
    if status == "REJECT":
        return f"{feature}={value:.3g} is outside the acceptable band [{reject_min:.3g}, {reject_max:.3g}]."
    return (
        f"{feature}={value:.3g} is near the edge of keep band [{keep_min:.3g}, {keep_max:.3g}] "
        f"within [{reject_min:.3g}, {reject_max:.3g}]."
    )


def _summarize_clip_parameters(
    clip_metadata: dict[str, Any], rubric: RubricState
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, list[Any]]]:
    thresholds = rubric.get_thresholds_summary()
    parameter_grading: dict[str, dict[str, Any]] = {}
    parameter_summary = {"keep": 0, "borderline": 0, "reject": 0}
    high_signals: list[str] = []
    mixed_signals: list[str] = []
    low_signals: list[str] = []

    for feature, threshold in thresholds.items():
        raw_value = clip_metadata.get(feature)
        if not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        status = rubric.get_feature_status(feature, value)
        reason = _feature_reason(feature, value, threshold, status)
        parameter_grading[feature] = {
            "value": round(value, 6),
            "status": status,
            "reason": reason,
        }

        if status == "KEEP":
            parameter_summary["keep"] += 1
            high_signals.append(reason)
        elif status == "REJECT":
            parameter_summary["reject"] += 1
            low_signals.append(reason)
        else:
            parameter_summary["borderline"] += 1
            mixed_signals.append(reason)

    occlusion_present = bool(clip_metadata.get("occlusion_present", False))
    if occlusion_present:
        occlusion_reason = "occlusion_present=true indicates visible obstruction."
        parameter_grading["occlusion_present"] = {
            "value": True,
            "status": "REJECT",
            "reason": occlusion_reason,
        }
        parameter_summary["reject"] += 1
        low_signals.append(occlusion_reason)
    else:
        occlusion_reason = "occlusion_present=false indicates an unobstructed frame."
        parameter_grading["occlusion_present"] = {
            "value": False,
            "status": "KEEP",
            "reason": occlusion_reason,
        }
        parameter_summary["keep"] += 1
        high_signals.append(occlusion_reason)

    dominant_features = rubric.get_dominant_features(clip_metadata)
    dominant_signals = [
        {
            "feature": feature,
            "status": str(parameter_grading[feature]["status"]),
            "reason": str(parameter_grading[feature]["reason"]),
        }
        for feature in dominant_features
        if feature in parameter_grading
    ]

    return (
        parameter_grading,
        parameter_summary,
        {
            "high": high_signals,
            "mixed": mixed_signals,
            "low": low_signals,
            "dominant": dominant_signals,
        },
    )


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for reason in reasons:
        text = reason.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _build_score_reasons(
    action: Action,
    expected_label: str,
    reward_breakdown: dict[str, float],
    parameter_signals: dict[str, list[Any]],
) -> dict[str, list[str]]:
    high: list[str] = []
    mixed: list[str] = []
    low: list[str] = []

    if float(reward_breakdown.get("format_score", 0.0)) >= 0.10:
        high.append("Action format is valid (label, confidence, reasoning provided).")
    else:
        low.append("Action format is incomplete, reducing score.")

    label_score = float(reward_breakdown.get("label_score", 0.0))
    if label_score >= 0.60:
        high.append(f"Predicted label {action.label} matches expected label {expected_label}.")
    elif label_score > 0.0:
        mixed.append(f"Predicted label {action.label} gets partial credit against expected label {expected_label}.")
    else:
        low.append(f"Predicted label {action.label} conflicts with expected label {expected_label}.")

    reasoning_score = float(reward_breakdown.get("reasoning_score", 0.0))
    if reasoning_score >= 0.20:
        high.append("Reasoning is well grounded in rubric features.")
    elif reasoning_score >= 0.10:
        mixed.append("Reasoning is partially grounded in rubric features.")
    else:
        low.append("Reasoning has weak feature grounding.")

    dominant = parameter_signals.get("dominant", [])[:2]
    for signal in dominant:
        if not isinstance(signal, dict):
            continue
        reason = str(signal.get("reason", "")).strip()
        status = str(signal.get("status", "BORDERLINE")).upper()
        if not reason:
            continue
        if status == "KEEP":
            high.append(reason)
        elif status == "REJECT":
            low.append(reason)
        else:
            mixed.append(reason)

    high.extend(parameter_signals.get("high", [])[:2])
    mixed.extend(parameter_signals.get("mixed", [])[:2])
    low.extend(parameter_signals.get("low", [])[:2])

    return {"high": _dedupe_reasons(high)[:6], "mixed": _dedupe_reasons(mixed)[:6], "low": _dedupe_reasons(low)[:6]}


def _build_clip_grade_record(
    *,
    episode: int,
    step: int,
    difficulty: str,
    clip_metadata: dict[str, Any],
    action: Action,
    expected_label: str,
    reward_breakdown: dict[str, float],
    rubric: RubricState,
    parameter_grading: dict[str, dict[str, Any]] | None = None,
    parameter_summary: dict[str, int] | None = None,
    parameter_signals: dict[str, list[Any]] | None = None,
    model_predicted_label: str | None = None,
    calibration_note: str | None = None,
) -> dict[str, Any]:
    if parameter_grading is None or parameter_summary is None or parameter_signals is None:
        parameter_grading, parameter_summary, parameter_signals = _summarize_clip_parameters(clip_metadata, rubric)
    score_reasons = _build_score_reasons(
        action=action,
        expected_label=expected_label,
        reward_breakdown=reward_breakdown,
        parameter_signals=parameter_signals,
    )
    return {
        "episode": episode,
        "step": step,
        "difficulty": difficulty,
        "clip_id": clip_metadata["clip_id"],
        "predicted_label": action.label,
        "model_predicted_label": model_predicted_label or action.label,
        "label_calibrated": bool(model_predicted_label and model_predicted_label != action.label),
        "calibration_note": calibration_note,
        "expected_label": expected_label,
        "confidence": float(action.confidence),
        "reward": reward_breakdown,
        "parameter_summary": parameter_summary,
        "parameter_grading": parameter_grading,
        "score_reasons": score_reasons,
        "agent_reasoning": action.reasoning,
        "clip_metadata": clip_metadata,
    }


def grade_manifest_clips(
    manifest_path: str,
    api_base_url: str | None = None,
    model_name: str | None = None,
    hf_token: str | None = None,
    rubric: RubricState | None = None,
    gt_store: GTStore | None = None,
    max_difficulty: str = "hard",
) -> dict[str, Any]:
    """
    Grade every clip in a real-clip manifest exactly once and return per-clip scores.
    """
    resolved_api_base_url, resolved_model_name, resolved_hf_token = _load_client_config(
        api_base_url=api_base_url,
        model_name=model_name,
        hf_token=hf_token,
    )
    client = OpenAI(base_url=resolved_api_base_url, api_key=resolved_hf_token)
    rubric_state = rubric or RubricState()
    gt = gt_store or GTStore()
    pools = load_real_clip_manifest(manifest_path, rubric_state)
    selected_difficulties = _difficulty_prefix(max_difficulty)

    per_clip_grades: list[dict[str, Any]] = []
    clip_index = 0
    for difficulty in DIFFICULTIES:
        if difficulty not in selected_difficulties:
            continue
        for clip_metadata in pools[difficulty]:
            clip_index += 1
            observation = Observation(
                step=1,
                rubric_version=rubric_state.version,
                rubric_summary=rubric_state.to_prompt_text(),
                clip_metadata=clip_metadata,
                history=[],
            )
            try:
                raw_action = get_agent_action(observation, client, resolved_model_name)
            except Exception:
                raw_action = Action(label="BORDERLINE", reasoning="Fallback due to API error.", confidence=0.5)

            parameter_grading, parameter_summary, parameter_signals = _summarize_clip_parameters(clip_metadata, rubric_state)
            action, calibration_note, model_predicted_label = _calibrate_action(
                raw_action=raw_action,
                clip_metadata=clip_metadata,
                rubric=rubric_state,
                parameter_grading=parameter_grading,
            )

            reward_breakdown = grade(action, clip_metadata, rubric_state, gt).model_dump()
            expected_label = gt.lookup(str(clip_metadata["clip_id"])) or rubric_state.derive_label(clip_metadata)
            per_clip_grades.append(
                _build_clip_grade_record(
                    episode=clip_index,
                    step=1,
                    difficulty=difficulty,
                    clip_metadata=clip_metadata,
                    action=action,
                    expected_label=expected_label,
                    reward_breakdown=reward_breakdown,
                    rubric=rubric_state,
                    parameter_grading=parameter_grading,
                    parameter_summary=parameter_summary,
                    parameter_signals=parameter_signals,
                    model_predicted_label=model_predicted_label,
                    calibration_note=calibration_note,
                )
            )

    return {
        "mode": "manifest_clip_grading",
        "manifest_path": manifest_path,
        "max_difficulty": max_difficulty,
        "clips_graded": len(per_clip_grades),
        "difficulty_counts": {difficulty: len([r for r in per_clip_grades if r["difficulty"] == difficulty]) for difficulty in DIFFICULTIES},
        "per_clip_grades": per_clip_grades,
        "gt_size": gt.size(),
        "rubric_version": rubric_state.version,
    }


def run_baseline(
    max_episodes: int | None = None,
    real_clips_manifest: str | None = None,
    api_base_url: str | None = None,
    model_name: str | None = None,
    hf_token: str | None = None,
    max_difficulty: str = "hard",
    rubric: RubricState | None = None,
    gt_store: GTStore | None = None,
    history_path: str | None = None,
) -> dict[str, Any]:
    if max_episodes is None:
        max_episodes = int(os.environ.get("MAX_EPISODES", "10"))
    if real_clips_manifest is None:
        real_clips_manifest = os.environ.get("REAL_CLIPS_MANIFEST")
    _difficulty_prefix(max_difficulty)

    api_base_url, model_name, hf_token = _load_client_config(
        api_base_url=api_base_url,
        model_name=model_name,
        hf_token=hf_token,
    )
    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    start_time = time.time()
    if rubric is None and gt_store is None and history_path is None:
        env = ClipQualityEnv(real_clips_path=real_clips_manifest)
    else:
        env = ClipQualityEnv(
            gt_store=gt_store,
            rubric=rubric,
            history_path=history_path or "state/history.jsonl",
            real_clips_path=real_clips_manifest,
        )
    difficulty_names = list(DIFFICULTY_ORDER)
    selected_difficulties = _difficulty_prefix(max_difficulty)
    per_clip_grades: list[dict[str, Any]] = []
    timeout_reached = False

    episode = 0
    while episode < max_episodes:
        elapsed = time.time() - start_time
        if elapsed > (TIMEOUT_SECONDS - BUFFER_SECONDS):
            timeout_reached = True
            break

        obs = env.reset()
        done = False
        step = 0
        while not done:
            current_obs = obs
            clip_metadata = current_obs.clip_metadata.model_dump()
            try:
                raw_action = get_agent_action(obs, client, model_name)
            except Exception:
                raw_action = Action(label="BORDERLINE", reasoning="Fallback due to API error.", confidence=0.5)

            parameter_grading, parameter_summary, parameter_signals = _summarize_clip_parameters(clip_metadata, env.rubric)
            action, calibration_note, model_predicted_label = _calibrate_action(
                raw_action=raw_action,
                clip_metadata=clip_metadata,
                rubric=env.rubric,
                parameter_grading=parameter_grading,
            )

            obs, reward, done, info = env.step(action)
            reward_breakdown = info.get(
                "reward_breakdown",
                {
                    "total": float(reward),
                    "format_score": 0.0,
                    "label_score": 0.0,
                    "reasoning_score": 0.0,
                },
            )
            expected_label = str(info.get("expected_label", "BORDERLINE"))
            difficulty = difficulty_names[step]
            if difficulty in selected_difficulties:
                per_clip_grades.append(
                    _build_clip_grade_record(
                        episode=episode + 1,
                        step=step + 1,
                        difficulty=difficulty,
                        clip_metadata=clip_metadata,
                        action=action,
                        expected_label=expected_label,
                        reward_breakdown=reward_breakdown,
                        rubric=env.rubric,
                        parameter_grading=parameter_grading,
                        parameter_summary=parameter_summary,
                        parameter_signals=parameter_signals,
                        model_predicted_label=model_predicted_label,
                        calibration_note=calibration_note,
                    )
                )
            step += 1
        episode += 1

    difficulty_counts = {
        difficulty: sum(1 for item in per_clip_grades if item["difficulty"] == difficulty)
        for difficulty in difficulty_names
    }
    return {
        "episodes_requested": max_episodes,
        "episodes_run": episode,
        "max_difficulty": max_difficulty,
        "steps_per_episode": len(selected_difficulties),
        "clips_graded": len(per_clip_grades),
        "timeout_reached": timeout_reached,
        "difficulty_counts": difficulty_counts,
        "per_clip_grades": per_clip_grades,
        "gt_size": env.gt_store.size(),
        "rubric_version": env.rubric.version,
        "real_clips_manifest": real_clips_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline agent against ClipQualityEnv.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Number of episodes (defaults to MAX_EPISODES env var or 10).",
    )
    parser.add_argument(
        "--real-clips-manifest",
        type=str,
        default=None,
        help="Path to JSON/JSONL manifest of real clip metadata.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help=(
            f"OpenAI-compatible API base URL (defaults to API_BASE_URL env or {DEFAULT_HF_OPENAI_BASE_URL} "
            "for Hugging Face Inference Providers)."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model identifier (e.g., Hugging Face model id). Falls back to MODEL_NAME env var.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token (or equivalent API key). Falls back to HF_TOKEN/OPENAI_API_KEY env vars.",
    )
    parser.add_argument(
        "--grade-all-clips-once",
        action="store_true",
        help=(
            "If set with --real-clips-manifest, grade each manifest clip exactly once. "
            "Otherwise runs baseline episodes with easy->medium->hard trajectories."
        ),
    )
    parser.add_argument(
        "--max-difficulty",
        type=str,
        default="hard",
        choices=list(DIFFICULTY_ORDER),
        help="Highest difficulty to include (easy|medium|hard). Uses cumulative ordering.",
    )
    args = parser.parse_args()

    if args.grade_all_clips_once:
        if not args.real_clips_manifest:
            raise ValueError("--grade-all-clips-once requires --real-clips-manifest")
        results = grade_manifest_clips(
            manifest_path=args.real_clips_manifest,
            api_base_url=args.api_base_url,
            model_name=args.model_name,
            hf_token=args.hf_token,
            max_difficulty=args.max_difficulty,
        )
    else:
        results = run_baseline(
            max_episodes=args.max_episodes,
            real_clips_manifest=args.real_clips_manifest,
            api_base_url=args.api_base_url,
            model_name=args.model_name,
            hf_token=args.hf_token,
            max_difficulty=args.max_difficulty,
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
