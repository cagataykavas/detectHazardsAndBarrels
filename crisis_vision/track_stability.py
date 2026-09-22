"""Fail-closed temporal stability audit for emitted detection tracks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class TrackAuditInputError(ValueError):
    """Raised when an event artifact cannot support a trustworthy audit."""


@dataclass(frozen=True, slots=True)
class TrackStabilityPolicy:
    """Release thresholds for track-level temporal evidence."""

    min_observations: int = 3
    max_frame_gap: int = 8
    min_mean_confidence: float = 0.45
    max_confidence_drop: float = 0.35
    max_normalized_speed_per_frame: float = 0.12

    def validate(self) -> None:
        if self.min_observations < 2:
            raise TrackAuditInputError("min_observations must be at least 2")
        if self.max_frame_gap < 1:
            raise TrackAuditInputError("max_frame_gap must be positive")
        for name, value in (
            ("min_mean_confidence", self.min_mean_confidence),
            ("max_confidence_drop", self.max_confidence_drop),
            ("max_normalized_speed_per_frame", self.max_normalized_speed_per_frame),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise TrackAuditInputError(f"{name} must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "min_observations": self.min_observations,
            "max_frame_gap": self.max_frame_gap,
            "min_mean_confidence": self.min_mean_confidence,
            "max_confidence_drop": self.max_confidence_drop,
            "max_normalized_speed_per_frame": self.max_normalized_speed_per_frame,
        }


@dataclass(frozen=True, slots=True)
class _Observation:
    frame_index: int
    label: str
    kind: str
    confidence: float
    x: float
    y: float
    first_observation: bool


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackAuditInputError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TrackAuditInputError(f"{path} must be finite")
    return result


def _parse_detection(detection: Any, frame_index: int, index: int) -> tuple[str, _Observation]:
    path = f"frame[{frame_index}].detections[{index}]"
    if not isinstance(detection, Mapping):
        raise TrackAuditInputError(f"{path} must be an object")

    track_id = detection.get("track_id")
    label = detection.get("label")
    kind = detection.get("kind")
    first_observation = detection.get("first_observation")
    if not isinstance(track_id, str) or not track_id.strip():
        raise TrackAuditInputError(f"{path}.track_id must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise TrackAuditInputError(f"{path}.label must be a non-empty string")
    if not isinstance(kind, str) or not kind.strip():
        raise TrackAuditInputError(f"{path}.kind must be a non-empty string")
    if not isinstance(first_observation, bool):
        raise TrackAuditInputError(f"{path}.first_observation must be boolean")

    confidence = _finite_number(detection.get("confidence"), f"{path}.confidence")
    if not 0 <= confidence <= 1:
        raise TrackAuditInputError(f"{path}.confidence must be in [0, 1]")
    centroid = detection.get("centroid_normalized")
    if not isinstance(centroid, Mapping):
        raise TrackAuditInputError(f"{path}.centroid_normalized must be an object")
    x = _finite_number(centroid.get("x"), f"{path}.centroid_normalized.x")
    y = _finite_number(centroid.get("y"), f"{path}.centroid_normalized.y")
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise TrackAuditInputError(f"{path}.centroid_normalized must be in [0, 1]")

    return track_id, _Observation(
        frame_index=frame_index,
        label=label,
        kind=kind,
        confidence=confidence,
        x=x,
        y=y,
        first_observation=first_observation,
    )


def _collect(events: Iterable[Mapping[str, Any]]) -> tuple[dict[str, list[_Observation]], int]:
    tracks: dict[str, list[_Observation]] = defaultdict(list)
    previous_frame = -1
    frame_count = 0
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TrackAuditInputError(f"events[{event_index}] must be an object")
        frame_index = event.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise TrackAuditInputError(f"events[{event_index}].frame_index must be non-negative")
        if frame_index <= previous_frame:
            raise TrackAuditInputError("frame_index values must be strictly increasing")
        previous_frame = frame_index
        detections = event.get("detections")
        if not isinstance(detections, list):
            raise TrackAuditInputError(f"events[{event_index}].detections must be a list")
        seen_in_frame: set[str] = set()
        for detection_index, detection in enumerate(detections):
            track_id, observation = _parse_detection(detection, frame_index, detection_index)
            if track_id in seen_in_frame:
                raise TrackAuditInputError(
                    f"track {track_id!r} occurs more than once in frame {frame_index}"
                )
            seen_in_frame.add(track_id)
            tracks[track_id].append(observation)
        frame_count += 1
    if frame_count == 0:
        raise TrackAuditInputError("at least one frame event is required")
    if not tracks:
        raise TrackAuditInputError("at least one tracked detection is required")
    return dict(tracks), frame_count


def _round(value: float) -> float:
    return round(value, 6)


def _audit_track(
    track_id: str, observations: list[_Observation], policy: TrackStabilityPolicy
) -> dict[str, Any]:
    first = observations[0]
    gaps: list[int] = []
    speeds: list[float] = []
    confidence_drops: list[float] = []
    for previous, current in zip(observations, observations[1:], strict=False):
        gap = current.frame_index - previous.frame_index
        gaps.append(gap)
        speeds.append(math.hypot(current.x - previous.x, current.y - previous.y) / gap)
        confidence_drops.append(max(0.0, previous.confidence - current.confidence))

    mean_confidence = sum(item.confidence for item in observations) / len(observations)
    max_gap = max(gaps, default=0)
    max_speed = max(speeds, default=0.0)
    max_drop = max(confidence_drops, default=0.0)
    reasons: list[str] = []
    if len(observations) < policy.min_observations:
        reasons.append("insufficient_observations")
    if max_gap > policy.max_frame_gap:
        reasons.append("frame_gap_exceeded")
    if mean_confidence < policy.min_mean_confidence:
        reasons.append("mean_confidence_below_minimum")
    if max_drop > policy.max_confidence_drop:
        reasons.append("confidence_drop_exceeded")
    if max_speed > policy.max_normalized_speed_per_frame:
        reasons.append("normalized_speed_exceeded")
    if not first.first_observation:
        reasons.append("missing_track_origin")
    if any(item.first_observation for item in observations[1:]):
        reasons.append("repeated_track_origin")
    if any(item.label != first.label or item.kind != first.kind for item in observations[1:]):
        reasons.append("track_identity_changed")

    return {
        "track_id": track_id,
        "label": first.label,
        "kind": first.kind,
        "passed": not reasons,
        "reasons": reasons,
        "observation_count": len(observations),
        "first_frame": first.frame_index,
        "last_frame": observations[-1].frame_index,
        "max_frame_gap": max_gap,
        "mean_confidence": _round(mean_confidence),
        "max_confidence_drop": _round(max_drop),
        "max_normalized_speed_per_frame": _round(max_speed),
    }


def audit_track_stability(
    events: Iterable[Mapping[str, Any]],
    policy: TrackStabilityPolicy | None = None,
) -> dict[str, Any]:
    """Audit serialized frame events and return deterministic JSON-ready evidence."""

    active_policy = policy or TrackStabilityPolicy()
    active_policy.validate()
    tracks, frame_count = _collect(events)
    results = [
        _audit_track(track_id, tracks[track_id], active_policy) for track_id in sorted(tracks)
    ]
    reason_counts: dict[str, int] = defaultdict(int)
    for result in results:
        for reason in result["reasons"]:
            reason_counts[reason] += 1
    failed = sum(not result["passed"] for result in results)
    return {
        "schema_version": "track-stability/1.0",
        "passed": failed == 0,
        "policy": active_policy.to_dict(),
        "frame_count": frame_count,
        "track_count": len(results),
        "passed_track_count": len(results) - failed,
        "failed_track_count": failed,
        "reason_counts": dict(sorted(reason_counts.items())),
        "tracks": results,
    }
