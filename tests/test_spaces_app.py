from __future__ import annotations

import json

from spaces_app import _result_table, _simple_result_table


def test_result_table_flattens_per_clip_rows():
    result = {
        "per_clip_grades": [
            {
                "clip_id": "clip_a",
                "difficulty": "easy",
                "predicted_label": "KEEP",
                "expected_label": "KEEP",
                "confidence": 0.92,
                "reward": {
                    "total": 0.95,
                    "format_score": 0.1,
                    "label_score": 0.6,
                    "reasoning_score": 0.25,
                },
                "parameter_summary": {"keep": 8, "borderline": 1, "reject": 1},
                "score_reasons": {
                    "high": ["good framing"],
                    "mixed": ["audio borderline"],
                    "low": ["slight occlusion"],
                },
                "agent_reasoning": "Strong overall quality with one caveat.",
            }
        ]
    }

    rows = _result_table(result)
    assert len(rows) == 1
    first = rows[0]
    assert first[0] == "clip_a"
    assert first[1] == "easy"
    assert first[2] == "KEEP"
    assert first[7] == 0.95
    assert "good framing" in first[14]
    assert "audio borderline" in first[15]
    assert "slight occlusion" in first[16]


def test_append_manifest_rows_deduplicates_clip_ids(tmp_path):
    from spaces_app import _append_manifest_rows

    existing_manifest = tmp_path / "existing.jsonl"
    incoming_manifest = tmp_path / "incoming.jsonl"

    existing_manifest.write_text(
        json.dumps({"clip_id": "clip_1"}) + "\n" + json.dumps({"clip_id": "clip_2"}) + "\n",
        encoding="utf-8",
    )
    incoming_manifest.write_text(
        json.dumps({"clip_id": "clip_2"}) + "\n" + json.dumps({"clip_id": "clip_3"}) + "\n",
        encoding="utf-8",
    )

    appended = _append_manifest_rows(str(incoming_manifest), existing_manifest)
    assert appended == 1

    rows = [
        json.loads(line)
        for line in existing_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["clip_id"] for row in rows] == ["clip_1", "clip_2", "clip_3"]


def test_simple_result_table_contains_only_three_columns():
    result = {
        "per_clip_grades": [
            {
                "clip_id": "clip_simple_1",
                "predicted_label": "BORDERLINE",
                "score_reasons": {
                    "high": ["good lighting"],
                    "mixed": ["audio_snr_db borderline"],
                    "low": [],
                },
                "agent_reasoning": "Mixed cues.",
            }
        ]
    }
    rows = _simple_result_table(result)
    assert len(rows) == 1
    first = rows[0]
    assert len(first) == 3
    assert first[0] == "clip_simple_1"
    assert first[1] == "BORDERLINE"
    assert "audio_snr_db borderline" in first[2]
