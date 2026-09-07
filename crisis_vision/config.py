"""Validated configuration for crisis-scene detection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CrisisConfig:
    barrel_min_area_ratio: float = 0.006
    barrel_max_area_ratio: float = 0.12
    barrel_min_aspect_ratio: float = 0.2
    barrel_max_aspect_ratio: float = 0.85
    barrel_min_solidity: float = 0.55

    placard_ratio_test: float = 0.72
    placard_min_matches: int = 12
    placard_ransac_threshold: float = 5.0
    placard_min_inlier_ratio: float = 0.55
    placard_min_area_ratio: float = 0.001
    placard_max_area_ratio: float = 0.35
    placard_nms_iou: float = 0.4

    track_max_distance: float = 0.18
    track_max_missing: int = 8
    frame_stride: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CrisisConfig:
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> CrisisConfig:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Configuration root must be a JSON object")
        return cls.from_mapping(payload)

    def validate(self) -> None:
        unit_fields = (
            "barrel_min_area_ratio",
            "barrel_max_area_ratio",
            "barrel_min_solidity",
            "placard_ratio_test",
            "placard_min_inlier_ratio",
            "placard_min_area_ratio",
            "placard_max_area_ratio",
            "placard_nms_iou",
            "track_max_distance",
        )
        for name in unit_fields:
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be greater than 0 and at most 1")
        if self.barrel_min_area_ratio >= self.barrel_max_area_ratio:
            raise ValueError("barrel area-ratio bounds are reversed")
        if self.barrel_min_aspect_ratio >= self.barrel_max_aspect_ratio:
            raise ValueError("barrel aspect-ratio bounds are reversed")
        if self.placard_min_area_ratio >= self.placard_max_area_ratio:
            raise ValueError("placard area-ratio bounds are reversed")
        if self.placard_min_matches < 4:
            raise ValueError("placard_min_matches must be at least 4 for homography")
        if self.placard_ransac_threshold <= 0:
            raise ValueError("placard_ransac_threshold must be positive")
        if self.track_max_missing < 0:
            raise ValueError("track_max_missing must not be negative")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
