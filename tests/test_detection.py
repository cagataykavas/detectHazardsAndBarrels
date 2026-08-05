from pathlib import Path

import cv2
import numpy as np

from crisis_vision.config import CrisisConfig
from crisis_vision.demo import generate_synthetic_frames
from crisis_vision.detection import BarrelDetector, PlacardDetector
from crisis_vision.templates import load_templates

ASSETS = Path(__file__).resolve().parents[1] / "hazmats" / "hazmats"


def test_barrel_rules_are_resolution_independent():
    frame = np.full((540, 960, 3), 80, dtype=np.uint8)
    cv2.rectangle(frame, (120, 260), (200, 440), (25, 30, 220), -1)
    cv2.rectangle(frame, (680, 270), (765, 445), (220, 80, 25), -1)

    detections, diagnostics = BarrelDetector(CrisisConfig()).detect(frame)

    assert {detection.label for detection in detections} == {"red_barrel", "blue_barrel"}
    assert all(detection.evidence["decision_rule"] for detection in detections)
    assert diagnostics["mask_pixels"]["red_barrel"] > 0


def test_generated_placard_passes_sift_and_ransac(tmp_path):
    source = ASSETS / "flammable-solid.png"
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / source.name).write_bytes(source.read_bytes())
    sift = cv2.SIFT_create()
    templates = load_templates(template_dir, sift)
    frame = next(generate_synthetic_frames(source, 6))

    detections, diagnostics = PlacardDetector(CrisisConfig(), templates).detect(frame)

    assert [detection.label for detection in detections] == ["flammable-solid"]
    assert detections[0].evidence["inlier_ratio"] >= CrisisConfig().placard_min_inlier_ratio
    assert diagnostics["templates_evaluated"] == 1
