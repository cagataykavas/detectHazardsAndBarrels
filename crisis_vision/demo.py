"""Generated crisis scene for repeatable onboarding and smoke tests."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def generate_synthetic_frames(
    template_path: str | Path,
    frame_count: int = 48,
    *,
    width: int = 960,
    height: int = 540,
) -> Iterator[NDArray[np.uint8]]:
    """Yield a labeled synthetic warehouse scene containing two barrels and a placard."""

    if frame_count < 6:
        raise ValueError("Synthetic demo requires at least six frames")
    if width < 480 or height < 320:
        raise ValueError("Synthetic frame dimensions must be at least 480 x 320")
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None or template.size == 0:
        raise FileNotFoundError(f"Demo template could not be loaded: {template_path}")
    template_height, template_width = template.shape[:2]
    source_quad = np.float32(
        [
            [0, 0],
            [template_width - 1, 0],
            [template_width - 1, template_height - 1],
            [0, template_height - 1],
        ]
    )
    template_mask = np.full((template_height, template_width), 255, dtype=np.uint8)

    for index in range(frame_count):
        phase = index / max(frame_count - 1, 1)
        frame = np.full((height, width, 3), (82, 88, 92), dtype=np.uint8)
        cv2.rectangle(frame, (0, int(height * 0.66)), (width, height), (62, 68, 72), -1)
        for x_value in range(0, width, 120):
            cv2.line(frame, (x_value, 0), (x_value, int(height * 0.66)), (72, 77, 81), 2)
        cv2.line(frame, (0, int(height * 0.66)), (width, int(height * 0.66)), (150, 150, 150), 3)

        barrel_specs = (
            (
                int(210 + 45 * math.sin(phase * math.tau)),
                int(height * 0.86),
                78,
                148,
                (28, 35, 220),
            ),
            (
                int(750 + 38 * math.sin(phase * math.tau + 1.3)),
                int(height * 0.84),
                82,
                142,
                (220, 90, 28),
            ),
        )
        for center_x, base_y, barrel_width, barrel_height, color in barrel_specs:
            top_y = base_y - barrel_height
            cv2.rectangle(
                frame,
                (center_x - barrel_width // 2, top_y),
                (center_x + barrel_width // 2, base_y),
                color,
                -1,
            )
            cv2.ellipse(
                frame,
                (center_x, top_y),
                (barrel_width // 2, 10),
                0,
                0,
                360,
                color,
                -1,
            )
            cv2.ellipse(
                frame,
                (center_x, base_y),
                (barrel_width // 2, 10),
                0,
                0,
                360,
                color,
                -1,
            )
            cv2.rectangle(
                frame,
                (center_x - barrel_width // 2, top_y),
                (center_x + barrel_width // 2, base_y),
                (30, 30, 30),
                2,
            )

        placard_size = 178
        placard_x = int(390 + 38 * math.sin(phase * math.tau * 0.8))
        placard_y = int(105 + 20 * math.sin(phase * math.tau * 1.2))
        skew = int(10 * math.sin(phase * math.tau))
        destination_quad = np.float32(
            [
                [placard_x + skew, placard_y],
                [placard_x + placard_size, placard_y + 5],
                [placard_x + placard_size - skew, placard_y + placard_size],
                [placard_x, placard_y + placard_size - 4],
            ]
        )
        homography = cv2.getPerspectiveTransform(source_quad, destination_quad)
        warped = cv2.warpPerspective(template, homography, (width, height))
        warped_mask = cv2.warpPerspective(template_mask, homography, (width, height))
        frame[warped_mask > 0] = warped[warped_mask > 0]

        cv2.putText(
            frame,
            "SYNTHETIC SCENE - NOT OPERATIONAL FOOTAGE",
            (18, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        yield frame
