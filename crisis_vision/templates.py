"""Template discovery and SIFT feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TemplateFeature:
    name: str
    path: Path
    image: NDArray[np.uint8]
    keypoints: tuple[Any, ...]
    descriptors: NDArray[np.float32]


def load_templates(directory: str | Path, sift: Any) -> tuple[TemplateFeature, ...]:
    """Load readable PNG/JPEG templates that contain usable SIFT descriptors."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {root}")

    paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    templates: list[TemplateFeature] = []
    skipped: list[str] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            skipped.append(path.name)
            continue
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) < 4:
            skipped.append(path.name)
            continue
        templates.append(
            TemplateFeature(
                name=path.stem,
                path=path,
                image=image,
                keypoints=tuple(keypoints),
                descriptors=np.asarray(descriptors, dtype=np.float32),
            )
        )

    if not templates:
        detail = f"; skipped: {', '.join(skipped)}" if skipped else ""
        raise ValueError(f"No usable image templates found in {root}{detail}")
    return tuple(templates)
