"""Command-line interface for crisis-scene video processing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crisis_vision.config import CrisisConfig
from crisis_vision.demo import generate_synthetic_frames
from crisis_vision.pipeline import analyze_frames, analyze_video
from crisis_vision.track_stability import (
    TrackAuditInputError,
    TrackStabilityPolicy,
    audit_track_stability,
)

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "hazmats" / "hazmats"


def json_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "frame_event": {
            "frame_index": "zero-based source frame index",
            "timestamp_ms": "derived from source FPS",
            "processing_ms": "wall-clock detector and tracker time",
            "detections": [
                {
                    "track_id": "semantic label plus sequence number",
                    "first_observation": "boolean",
                    "label": "template name, red_barrel, or blue_barrel",
                    "kind": "hazmat or barrel",
                    "confidence": "heuristic score in [0, 1]",
                    "bbox": "source-pixel rectangle",
                    "centroid_normalized": "source position in [0, 1]",
                    "polygon": "four points for validated placards; empty for barrels",
                    "evidence": "measurements, thresholds, and association reason",
                }
            ],
            "diagnostics": "scene keypoints, masks, and candidate counts",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crisis-vision",
        description="Explainable hazmat placard and colored-barrel tracking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze a real video")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--templates", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--max-frames", type=int)
    analyze.add_argument("--no-video", action="store_true")

    demo = subparsers.add_parser("demo", help="run the generated integration scene")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_DIR)
    demo.add_argument("--template-name", default="flammable-solid")
    demo.add_argument("--frames", type=int, default=48)
    demo.add_argument("--fps", type=float, default=24.0)
    demo.add_argument("--config", type=Path)
    demo.add_argument("--no-video", action="store_true")

    audit = subparsers.add_parser(
        "audit-tracks", help="audit temporal stability in an events.jsonl artifact"
    )
    audit.add_argument("--events", required=True, type=Path)
    audit.add_argument("--min-observations", type=int, default=3)
    audit.add_argument("--max-frame-gap", type=int, default=8)
    audit.add_argument("--min-mean-confidence", type=float, default=0.45)
    audit.add_argument("--max-confidence-drop", type=float, default=0.35)
    audit.add_argument("--max-normalized-speed-per-frame", type=float, default=0.12)
    audit.add_argument(
        "--require-pass",
        action="store_true",
        help="return exit code 3 when a valid artifact violates the policy",
    )

    subparsers.add_parser("explain-schema", help="print the JSON event contract")
    return parser


def _load_config(path: Path | None) -> CrisisConfig:
    config = CrisisConfig.from_json(path) if path else CrisisConfig()
    config.validate()
    return config


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Event artifact does not exist: {path}")
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TrackAuditInputError(f"events.jsonl line {line_number} is empty")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrackAuditInputError(
                    f"events.jsonl line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(event, dict):
                raise TrackAuditInputError(
                    f"events.jsonl line {line_number} must contain an object"
                )
            events.append(event)
    return events


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "explain-schema":
            print(json.dumps(json_contract(), indent=2, sort_keys=True))
            return 0
        if args.command == "audit-tracks":
            policy = TrackStabilityPolicy(
                min_observations=args.min_observations,
                max_frame_gap=args.max_frame_gap,
                min_mean_confidence=args.min_mean_confidence,
                max_confidence_drop=args.max_confidence_drop,
                max_normalized_speed_per_frame=args.max_normalized_speed_per_frame,
            )
            report = audit_track_stability(_load_events(args.events), policy)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 3 if args.require_pass and not report["passed"] else 0
        config = _load_config(args.config)
        if args.command == "analyze":
            summary = analyze_video(
                args.input,
                template_dir=args.templates,
                output_dir=args.output,
                config=config,
                max_frames=args.max_frames,
                write_video=not args.no_video,
            )
        else:
            template_path = args.templates / f"{args.template_name}.png"
            if not template_path.is_file():
                raise FileNotFoundError(f"Named demo template does not exist: {template_path}")
            summary = analyze_frames(
                generate_synthetic_frames(template_path, args.frames),
                fps=args.fps,
                template_dir=args.templates,
                output_dir=args.output,
                config=config,
                write_video=not args.no_video,
                input_metadata={
                    "kind": "synthetic",
                    "generator": "crisis_vision.demo.generate_synthetic_frames",
                    "frames_requested": args.frames,
                    "placard_template": args.template_name,
                    "ground_truth_metrics": False,
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
