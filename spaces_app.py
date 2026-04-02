from __future__ import annotations

import json
import os
import shutil
import threading
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from clip_quality_env.ground_truth import GTStore
from clip_quality_env.rubric import RubricState
from inference import DIFFICULTY_ORDER, run_baseline
from scripts.extract_mp4_metadata import _extract_clip_metadata, _iter_videos, _load_environment_map
from scripts.extract_mp4_metadata import _CV2_IMPORT_ERROR, _NUMPY_IMPORT_ERROR


SPACES_SHARED_ROOT = Path("state/spaces_shared")
SPACES_SHARED_ROOT.mkdir(parents=True, exist_ok=True)
SHARED_RUBRIC_PATH = SPACES_SHARED_ROOT / "rubric.json"
SHARED_GT_PATH = SPACES_SHARED_ROOT / "ground_truth.json"
SHARED_HISTORY_PATH = SPACES_SHARED_ROOT / "history.jsonl"
SHARED_MANIFEST_PATH = SPACES_SHARED_ROOT / "real_clips_manifest.jsonl"
_RUN_LOCK = threading.Lock()


def _resolve_uploaded_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        return path if path.exists() else None
    candidate = getattr(value, "name", None)
    if isinstance(candidate, str):
        path = Path(candidate)
        return path if path.exists() else None
    return None


def _extract_uploaded_manifest(
    files: list[Any] | None,
    *,
    sample_fps: float,
    whisper_model: str,
    difficulty_mode: str,
    environment_tag: str | None,
    framing: str | None,
    environment_map_file: str | None,
) -> tuple[str, int, int]:
    if _NUMPY_IMPORT_ERROR is not None or _CV2_IMPORT_ERROR is not None:
        raise RuntimeError("Missing upload processing dependencies. Install numpy and opencv-python-headless.")

    if not files:
        raise ValueError("Please upload at least one clip.")

    selected_files = []
    for item in files:
        path = _resolve_uploaded_path(item)
        if path is not None:
            selected_files.append(path)
    if not selected_files:
        raise ValueError("No readable clip files were uploaded.")

    upload_root = Path(tempfile.mkdtemp(prefix="clip_quality_uploads_"))
    ingest_root = upload_root / "clips"
    ingest_root.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(selected_files, start=1):
        if not src.exists():
            continue
        target = ingest_root / f"{idx:04d}_{src.name}"
        shutil.copy2(src, target)

    candidates = _iter_videos(ingest_root)
    if not candidates:
        raise ValueError("Uploaded files contain no supported videos (.mp4/.mov/.mkv/.webm).")

    try:
        import whisper  # lazy import to avoid heavy dependency at module import time
    except Exception:
        class _NoWhisperModel:
            def transcribe(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return {"text": "", "segments": []}

        asr_model = _NoWhisperModel()
    else:
        asr_model = whisper.load_model(whisper_model)
    rubric = RubricState(path=str(upload_root / "rubric.json"))
    environment_map = _load_environment_map(environment_map_file)

    output_manifest = upload_root / "real_clips_manifest.jsonl"
    failures = 0
    with output_manifest.open("w", encoding="utf-8") as out:
        for video_path in candidates:
            try:
                row = _extract_clip_metadata(
                    video_path=video_path,
                    path_hint=video_path.relative_to(ingest_root),
                    sample_fps=sample_fps,
                    asr_model=asr_model,
                    rubric=rubric,
                    with_difficulty=(difficulty_mode == "derive"),
                    environment_tag_override=(environment_tag or None),
                    environment_map=environment_map,
                    framing_override=(framing or None),
                )
            except Exception:
                failures += 1
                continue
            out.write(json.dumps(row, ensure_ascii=True) + "\n")

    with output_manifest.open("r", encoding="utf-8") as manifest_file:
        written_rows = sum(1 for _ in manifest_file)
    if written_rows == 0:
        raise ValueError("No clips were successfully processed. Check uploads and settings.")
    return str(output_manifest), written_rows, failures


def _append_manifest_rows(new_manifest: str, target_manifest: Path) -> int:
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if target_manifest.exists():
        with target_manifest.open("r", encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                clip_id = row.get("clip_id")
                if isinstance(clip_id, str) and clip_id.strip():
                    existing_ids.add(clip_id.strip())

    appended = 0
    with target_manifest.open("a", encoding="utf-8") as out, open(new_manifest, "r", encoding="utf-8") as src:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            clip_id = row.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id.strip():
                continue
            clip_key = clip_id.strip()
            if clip_key in existing_ids:
                continue
            out.write(json.dumps(row, ensure_ascii=True) + "\n")
            existing_ids.add(clip_key)
            appended += 1
    return appended


def _shared_runtime_objects() -> tuple[RubricState, GTStore]:
    rubric = RubricState(path=str(SHARED_RUBRIC_PATH))
    gt_store = GTStore(seed_path="data/seed_gt.json", state_path=str(SHARED_GT_PATH))
    return rubric, gt_store


def _reset_shared_state() -> str:
    with _RUN_LOCK:
        for path in (SHARED_RUBRIC_PATH, SHARED_GT_PATH, SHARED_HISTORY_PATH, SHARED_MANIFEST_PATH):
            if path.exists():
                path.unlink()
    return "Shared learning state reset. Next run starts fresh from seed ground truth."


def _result_table(result: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in result.get("per_clip_grades", []):
        reward = item.get("reward", {})
        summary = item.get("parameter_summary", {})
        rows.append(
            [
                item.get("clip_id"),
                item.get("difficulty"),
                item.get("predicted_label"),
                item.get("model_predicted_label", item.get("predicted_label")),
                "yes" if bool(item.get("label_calibrated")) else "no",
                item.get("expected_label"),
                round(float(item.get("confidence", 0.0)), 4),
                round(float(reward.get("total", 0.0)), 4),
                round(float(reward.get("format_score", 0.0)), 4),
                round(float(reward.get("label_score", 0.0)), 4),
                round(float(reward.get("reasoning_score", 0.0)), 4),
                int(summary.get("keep", 0)),
                int(summary.get("borderline", 0)),
                int(summary.get("reject", 0)),
                " | ".join(item.get("score_reasons", {}).get("high", [])),
                " | ".join(item.get("score_reasons", {}).get("mixed", [])),
                " | ".join(item.get("score_reasons", {}).get("low", [])),
                item.get("calibration_note", "") or "",
                item.get("agent_reasoning", ""),
            ]
        )
    return rows


def _simple_result_table(result: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in result.get("per_clip_grades", []):
        reasons = item.get("score_reasons", {})
        label = str(item.get("predicted_label", "BORDERLINE"))
        if label == "KEEP":
            reason_text = " | ".join(reasons.get("high", [])[:2]) or str(item.get("agent_reasoning", ""))
        elif label == "REJECT":
            reason_text = " | ".join(reasons.get("low", [])[:2]) or str(item.get("agent_reasoning", ""))
        else:
            picked = reasons.get("mixed", [])[:2] or reasons.get("high", [])[:1] or reasons.get("low", [])[:1]
            reason_text = " | ".join(picked) or str(item.get("agent_reasoning", ""))

        rows.append(
            [
                item.get("clip_id"),
                label,
                reason_text.strip(),
            ]
        )
    return rows


def _run_uploaded_grading(
    clips: list[Any],
    model_name: str,
    hf_token: str,
    api_base_url: str,
    sample_fps: float,
    whisper_model: str,
    difficulty_mode: str,
    environment_tag: str,
    framing: str,
    environment_map_file: Any,
    max_difficulty: str,
    episodes: int,
) -> tuple[str, list[list[Any]], list[list[Any]], str]:
    resolved_model = model_name.strip()
    if not resolved_model:
        raise ValueError("Model name is required.")

    resolved_token = hf_token.strip() if hf_token else os.environ.get("HF_TOKEN", "").strip()
    if not resolved_token:
        raise ValueError("HF token is required (or set HF_TOKEN env var in Space secrets).")

    resolved_base = api_base_url.strip() or "https://router.huggingface.co/v1"

    map_path_obj = _resolve_uploaded_path(environment_map_file)
    map_path = str(map_path_obj) if map_path_obj is not None else None

    if max_difficulty not in DIFFICULTY_ORDER:
        raise ValueError(f"max_difficulty must be one of {DIFFICULTY_ORDER}")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    with _RUN_LOCK:
        manifest_path, written_rows, failed_rows = _extract_uploaded_manifest(
            clips,
            sample_fps=sample_fps,
            whisper_model=whisper_model,
            difficulty_mode=difficulty_mode,
            environment_tag=environment_tag,
            framing=framing,
            environment_map_file=map_path,
        )
        added_rows = _append_manifest_rows(manifest_path, SHARED_MANIFEST_PATH)
        if not SHARED_MANIFEST_PATH.exists():
            raise ValueError("Shared manifest was not initialized.")

        rubric, gt_store = _shared_runtime_objects()
        result = run_baseline(
            max_episodes=episodes,
            real_clips_manifest=str(SHARED_MANIFEST_PATH),
            api_base_url=resolved_base,
            model_name=resolved_model,
            hf_token=resolved_token,
            max_difficulty=max_difficulty,
            rubric=rubric,
            gt_store=gt_store,
            history_path=str(SHARED_HISTORY_PATH),
        )
        report_path = SPACES_SHARED_ROOT / "per_clip_grades.json"
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    table_rows = _result_table(result)
    simple_rows = _simple_result_table(result)
    summary = (
        f"Processed {written_rows} uploaded clips ({failed_rows} failed extraction, {added_rows} new in shared set). "
        f"Ran {result.get('episodes_run', 0)}/{result.get('episodes_requested', episodes)} episodes at max difficulty "
        f"{max_difficulty}. Graded {result.get('clips_graded', 0)} steps. "
        f"Learning state: GT size {result.get('gt_size', 0)}, rubric v{result.get('rubric_version', 1)}."
    )
    return summary, table_rows, simple_rows, str(report_path)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ClipQualityEnv - Multi-Clip Grader") as demo:
        gr.Markdown(
            """
            # ClipQualityEnv - Hugging Face Spaces UI
            Upload a set of clips, extract metadata, and grade each clip with multi-parameter scoring + reasons.
            """
        )

        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown(" ")
            with gr.Column(scale=2):
                reset_button = gr.Button("Reset State", variant="secondary", size="sm")

        with gr.Row():
            clips = gr.File(
                file_count="multiple",
                file_types=[".mp4", ".mov", ".mkv", ".webm"],
                type="filepath",
                label="Upload clips",
            )
            environment_map_file = gr.File(
                file_count="single",
                file_types=[".json"],
                type="filepath",
                label="Optional environment map JSON",
            )

        with gr.Row():
            model_name = gr.Textbox(
                value=os.environ.get("MODEL_NAME", "openai/gpt-oss-120b"),
                label="Hugging Face model id",
            )
            hf_token = gr.Textbox(
                value="",
                type="password",
                label="HF token (leave empty if set in Space secrets as HF_TOKEN)",
            )
            api_base_url = gr.Textbox(
                value=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
                label="OpenAI-compatible base URL",
            )

        with gr.Row():
            sample_fps = gr.Slider(minimum=1.0, maximum=5.0, value=2.0, step=0.5, label="Frame sample FPS")
            whisper_model = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                value="small",
                label="Whisper model",
            )
            difficulty_mode = gr.Dropdown(
                choices=["derive", "none"],
                value="derive",
                label="Difficulty labeling",
            )

        with gr.Row():
            max_difficulty = gr.Dropdown(
                choices=list(DIFFICULTY_ORDER),
                value="hard",
                label="Max grading difficulty",
                info="easy -> easy only; medium -> easy+medium; hard -> easy+medium+hard",
            )
            episodes = gr.Slider(
                minimum=1,
                maximum=100,
                value=10,
                step=1,
                label="Episodes to run",
            )

        with gr.Row():
            environment_tag = gr.Textbox(
                value="",
                label="Optional fixed environment_tag override for all clips",
            )
            framing = gr.Textbox(
                value="",
                label="Optional fixed framing override for all clips",
            )

        run_button = gr.Button("Grade Clips", variant="primary")
        summary = gr.Textbox(label="Run Summary", interactive=False)
        simple_results_table = gr.Dataframe(
            headers=[
                "clip_id",
                "label",
                "reason",
            ],
            datatype=[
                "str",
                "str",
                "str",
            ],
            wrap=True,
            label="Simple per-clip grading",
        )
        results_table = gr.Dataframe(
            headers=[
                "clip_id",
                "difficulty",
                "predicted_label",
                "model_predicted_label",
                "label_calibrated",
                "expected_label",
                "confidence",
                "total_score",
                "format_score",
                "label_score",
                "reasoning_score",
                "keep_signals",
                "borderline_signals",
                "reject_signals",
                "high_reasons",
                "mixed_reasons",
                "low_reasons",
                "calibration_note",
                "agent_reasoning",
            ],
            datatype=[
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "str",
                "str",
                "str",
                "str",
                "str",
            ],
            wrap=True,
            label="Deatailed per-clip grading",
        )
        report_file = gr.File(label="Download full JSON report")
        reset_status = gr.Textbox(label="Reset status", interactive=False)

        run_button.click(
            fn=_run_uploaded_grading,
            inputs=[
                clips,
                model_name,
                hf_token,
                api_base_url,
                sample_fps,
                whisper_model,
                difficulty_mode,
                environment_tag,
                framing,
                environment_map_file,
                max_difficulty,
                episodes,
            ],
            outputs=[summary, simple_results_table, results_table, report_file],
        )
        reset_button.click(
            fn=_reset_shared_state,
            inputs=[],
            outputs=[reset_status],
        )

    return demo


demo = build_demo()
demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
    )
