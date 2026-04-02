# ClipQualityEnv

## 1. Environment Description and Motivation

ClipQualityEnv is an **OpenEnv-compatible RL environment** for a real production task: curating a high-quality talking-head clip dataset for LoRA training.  
At each episode, an agent classifies clip metadata into `KEEP`, `BORDERLINE`, or `REJECT` across a fixed trajectory of **easy → medium → hard** tasks.

The environment is designed to continuously improve with use:
- **Ground truth expansion:** high-confidence, high-reward hard clips are promoted into the GT store.
- **Rubric calibration:** thresholds tighten as recent performance improves.

This co-evolution creates a non-stationary but controlled training signal similar to real-world data curation workflows.

## 2. Action and Observation Space Definitions

### Observation (`Observation`)
| Field | Type | Description |
|---|---|---|
| `step` | `int` (1..3) | Current step in episode |
| `rubric_version` | `int` | Versioned rubric state |
| `rubric_summary` | `str` | Human-readable grading rubric |
| `clip_metadata` | `ClipMetadata` | Metadata for current clip |
| `history` | `list[HistoryItem]` | Prior step outcomes in same episode |

`clip_metadata` includes quality signals such as `face_area_ratio`, `face_confidence`, `motion_score`, `bg_complexity_score`, `audio_snr_db`, `lighting_uniformity`, `occlusion_present`, etc.

### Action (`Action`)
| Field | Type | Constraint |
|---|---|---|
| `label` | `str` | One of `KEEP`, `BORDERLINE`, `REJECT` |
| `reasoning` | `str` | Non-empty |
| `confidence` | `float` | 0.0 to 1.0 |

### Reward (`Reward`)
Deterministic reward in `[0.0, 1.0]`:
- `format_score` (0.10 max)
- `label_score` (0.60 max)
- `reasoning_score` (0.30 max)
- `total = format_score + label_score + reasoning_score`

`step(action)` returns `(observation, reward, done, info)`, and `state()` returns the full checkpointable environment state.

## 3. Task Descriptions with Difficulty Levels

| Task ID | Difficulty | Objective |
|---|---|---|
| `easy_classification` | Easy | Classify clips with clearly dominant quality signals |
| `medium_classification` | Medium | Classify clips with one or two borderline features |
| `hard_classification` | Hard | Classify clips with conflicting/ambiguous quality signals |

Each episode runs exactly 3 steps in this order: **easy → medium → hard**.

## 4. Setup and Usage Instructions

### Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run tests
```bash
python -m pytest -q
```

### Run the OpenEnv-compatible API server
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

Available endpoints:
- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /render`

### Run baseline inference (`inference.py`)
Set required variables:
- `MODEL_NAME` (Hugging Face model id)
- `HF_TOKEN` (or `OPENAI_API_KEY`)

Optional:
- `API_BASE_URL` (defaults to `https://router.huggingface.co/v1`)

Example:
```bash
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

CLI-only equivalent (no env vars required for base URL/model/token):
```bash
python inference.py \
  --model-name meta-llama/Llama-3.3-70B-Instruct \
  --hf-token $HF_TOKEN
```

`inference.py` now returns **per-clip grading output** (not only mean-by-difficulty):
- `per_clip_grades`: one record per graded clip (episode + step + clip_id)
- `reward`: deterministic multi-parameter score breakdown (`format_score`, `label_score`, `reasoning_score`, `total`)
- `parameter_grading`: rubric status + reason for each rubric feature on that clip
- `score_reasons`: concise high/mixed/low explanations for why the clip scored the way it did

To grade every clip in a manifest exactly once (instead of episode sampling):
```bash
python inference.py \
  --real-clips-manifest data/real_clips_manifest.jsonl \
  --grade-all-clips-once \
  --model-name openai/gpt-oss-120b \
  --hf-token $HF_TOKEN
```

### Attach real clips (metadata manifest) and run agent
Use a JSONL (or JSON) manifest where each row contains full `ClipMetadata` fields and optional `difficulty`.

Example `data/real_clips_manifest.jsonl` row:
```json
{
  "clip_id": "real_0001",
  "duration_s": 8.4,
  "fps": 30,
  "resolution": "1920x1080",
  "face_area_ratio": 0.34,
  "face_confidence": 0.93,
  "head_pose_yaw_deg": 9.0,
  "head_pose_pitch_deg": 1.5,
  "motion_score": 0.18,
  "bg_complexity": "simple_room",
  "bg_complexity_score": 0.11,
  "mouth_open_ratio": 0.39,
  "blink_rate_hz": 0.24,
  "audio_snr_db": 23.1,
  "transcript_word_count": 45,
  "transcript_confidence": 0.94,
  "lighting_uniformity": 0.76,
  "occlusion_present": false,
  "environment_tag": "podcast_studio",
  "framing": "front",
  "difficulty": "easy"
}
```

Run baseline on real clips:
```bash
python inference.py --real-clips-manifest data/real_clips_manifest.jsonl --max-episodes 20
```

Or via environment variable (used by `app.py` and `train.py` too):
```bash
export REAL_CLIPS_MANIFEST=data/real_clips_manifest.jsonl
python inference.py
```

Validation behavior:
- Rows are validated with the same `ClipMetadata` Pydantic model.
- `difficulty` must be `easy|medium|hard` if present.
- If `difficulty` is omitted, it is derived from rubric ambiguity.
- Malformed rows fail fast with explicit errors.

### Extract metadata from `.mp4` clips
Use the provided extractor to generate a manifest from raw videos.

Install extractor dependencies:
```bash
pip install -r requirements_extractor.txt
```

Run extraction:
```bash
python scripts/extract_mp4_metadata.py \
  --input-dir /path/to/mp4_clips \
  --output data/real_clips_manifest.jsonl \
  --sample-fps 2 \
  --difficulty-mode derive \
  --whisper-model small
```

If your MediaPipe build does not expose `mp.solutions.face_mesh`, the extractor now auto-falls back to a heuristic face-feature mode and still produces valid manifest rows.

`environment_tag` is inferred from directory/file naming keywords when possible; otherwise it defaults to `unknown_env`.
`framing` captures shot composition/view (for example: `front`, `left`, `right`, `closeup`, `offgaze`) and is inferred from path hints + pose when not overridden.

If all clips are from one known setup, set it explicitly:
```bash
python scripts/extract_mp4_metadata.py \
  --input-dir /path/to/mp4_clips \
  --output data/real_clips_manifest.jsonl \
  --environment-tag office \
  --framing front
```

For mixed environments, provide a per-clip mapping file:
```json
{
  "clip_001.mp4": "office",
  "clip_002.mp4": "street_walk",
  "clip_003": "podcast_studio"
}
```

Then run:
```bash
python scripts/extract_mp4_metadata.py \
  --input-dir /path/to/mp4_clips \
  --output data/real_clips_manifest.jsonl \
  --environment-map data/environment_map.json
```

Then run the environment on extracted clips:
```bash
python inference.py --real-clips-manifest data/real_clips_manifest.jsonl --max-episodes 20
```

### Hugging Face Spaces UI (multi-clip upload + grading)

This repository includes a Spaces-ready Gradio app: `spaces_app.py`.

It lets users:
- Upload multiple clips directly in the UI
- Optionally upload an environment map JSON
- Extract metadata from uploaded clips
- Persist learning across runs (shared rubric/GT/manifest state) until reset
- Grade with configurable episode count and max difficulty (`easy|medium|hard`)
- Use cumulative difficulty output (easy; easy+medium; easy+medium+hard)
- Apply calibrated labels/reasoning to avoid weak BORDERLINE-only behavior
- Show two output views:
  - **Deatailed per-clip grading** (full metrics and score breakdown)
  - **Simple per-clip grading** (`clip_id`, `KEEP/REJECT/BORDERLINE`, concise reason)
- Download full JSON report

Space controls:
- **Episodes to run**: how many training/grading episodes execute each run.
- **Max grading difficulty**:
  - `easy` -> outputs easy only
  - `medium` -> outputs easy + medium
  - `hard` -> outputs easy + medium + hard
- **Reset Learning State**: clears shared learned state and starts fresh from seed GT.

Run locally:
```bash
python spaces_app.py
```

Deploy on Hugging Face Spaces (Gradio SDK):
- Set **App file** to `spaces_app.py`
- Add secret `HF_TOKEN` (or provide token in UI input)
- (Optional) set `MODEL_NAME` / `API_BASE_URL` in Space variables

Deploy on Hugging Face Spaces (Docker SDK):
- Keep Space README header as `sdk: docker`
- The included `Dockerfile` now starts `spaces_app.py` by default
- Add secret `HF_TOKEN` in Space settings
- (Optional) set `MODEL_NAME` / `API_BASE_URL`

### Docker
```bash
docker build -t clip-quality-env .
docker run --rm -p 7860:7860 clip-quality-env
```

To run API server with real clips:
```bash
docker run --rm -p 7860:7860 \
  -e APP_MODE=api \
  -e REAL_CLIPS_MANIFEST=/app/data/real_clips_manifest.jsonl \
  clip-quality-env
```

## 5. Baseline Output

`inference.py` prints a JSON report with run-level metadata and per-clip grades.

Reproduce (offline fallback):
```bash
API_BASE_URL=http://127.0.0.1:1/v1 MODEL_NAME=offline-baseline HF_TOKEN=dummy MAX_EPISODES=10 python inference.py
```
