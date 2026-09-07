"""Serializable models for detections and frame events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _round(value: float) -> float:
    return round(float(value), 4)


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": _round(self.x), "y": _round(self.y)}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def centroid(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    kind: str
    confidence: float
    bbox: BoundingBox
    centroid_normalized: Point
    polygon: tuple[Point, ...] = ()
    track_id: str | None = None
    first_observation: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_observation": self.first_observation,
            "label": self.label,
            "kind": self.kind,
            "confidence": _round(self.confidence),
            "bbox": self.bbox.to_dict(),
            "centroid_normalized": self.centroid_normalized.to_dict(),
            "polygon": [point.to_dict() for point in self.polygon],
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    timestamp_ms: float
    detections: tuple[Detection, ...]
    processing_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_index": self.frame_index,
            "timestamp_ms": _round(self.timestamp_ms),
            "processing_ms": _round(self.processing_ms),
            "detections": [detection.to_dict() for detection in self.detections],
            "diagnostics": self.diagnostics,
        }
