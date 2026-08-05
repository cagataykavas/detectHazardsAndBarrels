import json
from pathlib import Path

from crisis_vision.demo import generate_synthetic_frames
from crisis_vision.pipeline import analyze_frames

ASSETS = Path(__file__).resolve().parents[1] / "hazmats" / "hazmats"


def test_demo_writes_explainable_artifacts(tmp_path):
    source = ASSETS / "flammable-solid.png"
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / source.name).write_bytes(source.read_bytes())
    output = tmp_path / "output"

    summary = analyze_frames(
        generate_synthetic_frames(source, 6),
        fps=24.0,
        template_dir=template_dir,
        output_dir=output,
        max_frames=4,
        write_video=False,
        input_metadata={"kind": "synthetic-test"},
    )
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]

    assert summary == json.loads((output / "summary.json").read_text())
    assert summary["source_frames_seen"] == 4
    assert summary["frames_processed"] == 4
    assert summary["kind_observations"]["barrel"] >= 8
    assert summary["kind_observations"]["hazmat"] >= 4
    assert summary["artifacts"]["video"] is None
    assert (output / "preview.jpg").stat().st_size > 0
    assert len(events) == 4
    assert events[0]["schema_version"] == "1.0"
    assert summary["safety_note"]
