"""Headless frame orchestration and reproducible artifact bundles."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from crisis_vision.config import CrisisConfig
from crisis_vision.detection import BarrelDetector, PlacardDetector
from crisis_vision.models import FrameResult
from crisis_vision.templates import load_templates
from crisis_vision.tracking import CentroidTracker


class FrameProcessor:
    def __init__(self, template_dir: str | Path, config: CrisisConfig | None = None) -> None:
        self.config = config or CrisisConfig()
        self.config.validate()
        sift = cv2.SIFT_create()
        self.templates = load_templates(template_dir, sift)
        self.barrel_detector = BarrelDetector(self.config)
        self.placard_detector = PlacardDetector(self.config, self.templates)
        self.tracker = CentroidTracker(self.config)

    def process(self, frame: NDArray[np.uint8], frame_index: int, fps: float) -> FrameResult:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty BGR image with shape (height, width, 3)")
        started = time.perf_counter()
        barrel_detections, barrel_diagnostics = self.barrel_detector.detect(frame)
        placard_detections, placard_diagnostics = self.placard_detector.detect(frame)
        detections = self.tracker.update(barrel_detections + placard_detections)
        return FrameResult(
            frame_index=frame_index,
            timestamp_ms=frame_index * 1000 / max(fps, 0.001),
            detections=tuple(detections),
            processing_ms=(time.perf_counter() - started) * 1000,
            diagnostics={
                "barrel_candidates": len(barrel_detections),
                "placard_candidates": len(placard_detections),
                **barrel_diagnostics,
                **placard_diagnostics,
            },
        )

    @staticmethod
    def annotate(frame: NDArray[np.uint8], result: FrameResult) -> NDArray[np.uint8]:
        canvas = frame.copy()
        colors = {
            "red_barrel": (40, 40, 245),
            "blue_barrel": (245, 130, 40),
            "hazmat": (40, 230, 230),
        }
        for detection in result.detections:
            color = colors["hazmat"] if detection.kind == "hazmat" else colors[detection.label]
            bbox = detection.bbox
            cv2.rectangle(
                canvas,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                color,
                2,
            )
            if detection.polygon:
                polygon = np.array(
                    [[point.x, point.y] for point in detection.polygon], dtype=np.int32
                ).reshape(-1, 1, 2)
                cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_AA)
            label = f"{detection.track_id} {detection.confidence:.2f}"
            cv2.putText(
                canvas,
                label,
                (bbox.x, max(18, bbox.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        status = (
            f"frame={result.frame_index} detections={len(result.detections)} "
            f"processing={result.processing_ms:.1f}ms"
        )
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 610), 32), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            status,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        return canvas


def analyze_frames(
    frames: Iterable[NDArray[np.uint8]],
    *,
    fps: float,
    template_dir: str | Path,
    output_dir: str | Path,
    config: CrisisConfig | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    processor = FrameProcessor(template_dir, config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    events_path = destination / "events.jsonl"
    summary_path = destination / "summary.json"
    preview_path = destination / "preview.jpg"
    video_path = destination / "annotated.mp4"

    writer: cv2.VideoWriter | None = None
    track_observations: Counter[str] = Counter()
    kind_observations: Counter[str] = Counter()
    unique_tracks: set[str] = set()
    frames_processed = 0
    source_frames_seen = 0
    processing_total_ms = 0.0
    last_annotated: NDArray[np.uint8] | None = None
    wall_started = time.perf_counter()

    with events_path.open("w", encoding="utf-8") as handle:
        for frame_index, frame in enumerate(frames):
            if max_frames is not None and source_frames_seen >= max_frames:
                break
            source_frames_seen += 1
            if frame_index % processor.config.frame_stride != 0:
                continue
            result = processor.process(frame, frame_index, fps)
            annotated = processor.annotate(frame, result)
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            if write_video and writer is None:
                height, width = annotated.shape[:2]
                candidate = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps / processor.config.frame_stride,
                    (width, height),
                )
                if candidate.isOpened():
                    writer = candidate
                else:
                    candidate.release()
            if writer is not None:
                writer.write(annotated)

            for detection in result.detections:
                if detection.track_id:
                    track_observations[detection.track_id] += 1
                    unique_tracks.add(detection.track_id)
                kind_observations[detection.kind] += 1
            processing_total_ms += result.processing_ms
            frames_processed += 1
            last_annotated = annotated

    if writer is not None:
        writer.release()
    elif video_path.exists():
        video_path.unlink()
    if frames_processed == 0 or last_annotated is None:
        raise ValueError("No frames were processed")
    if not cv2.imwrite(str(preview_path), last_annotated):
        raise OSError(f"Could not write preview image: {preview_path}")

    wall_seconds = time.perf_counter() - wall_started
    video_written = video_path.exists() and video_path.stat().st_size > 0
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "source_frames_seen": source_frames_seen,
        "frames_processed": frames_processed,
        "fps_reported": round(float(fps), 4),
        "templates_loaded": [template.name for template in processor.templates],
        "unique_track_count": len(unique_tracks),
        "track_observations": dict(sorted(track_observations.items())),
        "kind_observations": dict(sorted(kind_observations.items())),
        "mean_processing_ms": round(processing_total_ms / frames_processed, 4),
        "wall_seconds": round(wall_seconds, 4),
        "throughput_fps": round(frames_processed / max(wall_seconds, 0.0001), 4),
        "configuration": processor.config.to_dict(),
        "input": input_metadata or {"kind": "frame_iterable"},
        "artifacts": {
            "events": events_path.name,
            "summary": summary_path.name,
            "preview": preview_path.name,
            "video": video_path.name if video_written else None,
        },
        "metric_note": (
            "Counts are heuristic observations from this run, not an accuracy benchmark."
        ),
        "safety_note": "Not certified for operational emergency-response decisions.",
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _capture_frames(capture: cv2.VideoCapture) -> Iterator[NDArray[np.uint8]]:
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


def analyze_video(
    input_path: str | Path,
    *,
    template_dir: str | Path,
    output_dir: str | Path,
    config: CrisisConfig | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
) -> dict[str, Any]:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open input video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    metadata = {
        "kind": "video",
        "path": str(source),
        "reported_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    try:
        return analyze_frames(
            _capture_frames(capture),
            fps=fps,
            template_dir=template_dir,
            output_dir=output_dir,
            config=config,
            max_frames=max_frames,
            write_video=write_video,
            input_metadata=metadata,
        )
    finally:
        capture.release()
