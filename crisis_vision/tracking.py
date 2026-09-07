"""Greedy, class-aware centroid tracking with visible association evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from crisis_vision.config import CrisisConfig
from crisis_vision.models import Detection, Point


@dataclass(slots=True)
class _Track:
    entity_key: str
    position: Point
    missing_frames: int = 0


class CentroidTracker:
    def __init__(self, config: CrisisConfig) -> None:
        self.config = config
        self._tracks: dict[str, _Track] = {}
        self._next_id: dict[str, int] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        for track in self._tracks.values():
            track.missing_frames += 1

        assigned_track_ids: set[str] = set()
        results: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            entity_key = f"{detection.kind}:{detection.label}"
            compatible = [
                (track_id, track)
                for track_id, track in self._tracks.items()
                if track.entity_key == entity_key and track_id not in assigned_track_ids
            ]
            nearest_id: str | None = None
            nearest_distance = self.config.track_max_distance
            for track_id, track in compatible:
                distance = self._distance(track.position, detection.centroid_normalized)
                if distance <= nearest_distance:
                    nearest_id = track_id
                    nearest_distance = distance

            if nearest_id is None:
                sequence = self._next_id.get(entity_key, 1)
                self._next_id[entity_key] = sequence + 1
                prefix = (
                    f"hazmat_{detection.label}" if detection.kind == "hazmat" else detection.label
                )
                track_id = f"{prefix}_{sequence}"
                first_observation = True
                association = "new_track"
            else:
                track_id = nearest_id
                first_observation = False
                association = "nearest_same_label_centroid"

            self._tracks[track_id] = _Track(
                entity_key=entity_key,
                position=detection.centroid_normalized,
                missing_frames=0,
            )
            assigned_track_ids.add(track_id)
            results.append(
                replace(
                    detection,
                    track_id=track_id,
                    first_observation=first_observation,
                    evidence={
                        **detection.evidence,
                        "association_rule": association,
                        "association_distance": round(nearest_distance, 4)
                        if nearest_id is not None
                        else None,
                        "maximum_association_distance": self.config.track_max_distance,
                    },
                )
            )

        for track_id in list(self._tracks):
            if self._tracks[track_id].missing_frames > self.config.track_max_missing:
                del self._tracks[track_id]
        return results

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)
