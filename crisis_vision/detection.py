"""Scale-aware barrel segmentation and validated placard matching."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from crisis_vision.config import CrisisConfig
from crisis_vision.models import BoundingBox, Detection, Point
from crisis_vision.templates import TemplateFeature


def _contours(mask: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(found)


def _normalized_centroid(bbox: BoundingBox, width: int, height: int) -> Point:
    center = bbox.centroid
    return Point(center.x / max(width, 1), center.y / max(height, 1))


class BarrelDetector:
    """Detect vertically oriented red and blue barrel-shaped regions."""

    _RANGES = {
        "red_barrel": (
            (np.array([0, 85, 55]), np.array([12, 255, 255])),
            (np.array([165, 85, 55]), np.array([179, 255, 255])),
        ),
        "blue_barrel": ((np.array([92, 85, 40]), np.array([142, 255, 255])),),
    }

    def __init__(self, config: CrisisConfig) -> None:
        self.config = config

    def detect(self, frame: NDArray[np.uint8]) -> tuple[list[Detection], dict[str, Any]]:
        height, width = frame.shape[:2]
        frame_area = float(height * width)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        detections: list[Detection] = []
        mask_pixels: dict[str, int] = {}

        for label, ranges in self._RANGES.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_pixels[label] = cv2.countNonZero(mask)

            for contour in _contours(mask):
                area = float(cv2.contourArea(contour))
                x, y, box_width, box_height = cv2.boundingRect(contour)
                if box_height <= 0:
                    continue
                area_ratio = area / frame_area
                aspect_ratio = box_width / float(box_height)
                hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
                solidity = min(area / hull_area, 1.0)
                matches = (
                    self.config.barrel_min_area_ratio
                    <= area_ratio
                    <= self.config.barrel_max_area_ratio
                    and self.config.barrel_min_aspect_ratio
                    <= aspect_ratio
                    <= self.config.barrel_max_aspect_ratio
                    and solidity >= self.config.barrel_min_solidity
                )
                if not matches:
                    continue
                bbox = BoundingBox(x, y, box_width, box_height)
                aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 0.58) / 0.9)
                solidity_score = max(
                    0.0,
                    (solidity - self.config.barrel_min_solidity)
                    / max(1 - self.config.barrel_min_solidity, 0.01),
                )
                confidence = min(0.98, 0.5 + 0.25 * aspect_score + 0.25 * solidity_score)
                detections.append(
                    Detection(
                        label=label,
                        kind="barrel",
                        confidence=confidence,
                        bbox=bbox,
                        centroid_normalized=_normalized_centroid(bbox, width, height),
                        evidence={
                            "decision_rule": "hsv_vertical_solid_region",
                            "area_px": round(area, 2),
                            "area_ratio": round(area_ratio, 6),
                            "aspect_ratio": round(aspect_ratio, 4),
                            "solidity": round(solidity, 4),
                            "thresholds": {
                                "area_ratio": [
                                    self.config.barrel_min_area_ratio,
                                    self.config.barrel_max_area_ratio,
                                ],
                                "aspect_ratio": [
                                    self.config.barrel_min_aspect_ratio,
                                    self.config.barrel_max_aspect_ratio,
                                ],
                                "minimum_solidity": self.config.barrel_min_solidity,
                            },
                        },
                    )
                )
        return detections, {"mask_pixels": mask_pixels}


class PlacardDetector:
    """Match templates, validate the homography, then suppress overlapping candidates."""

    def __init__(self, config: CrisisConfig, templates: tuple[TemplateFeature, ...]) -> None:
        self.config = config
        self.templates = templates
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect(self, frame: NDArray[np.uint8]) -> tuple[list[Detection], dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scene_keypoints, scene_descriptors = self.sift.detectAndCompute(gray, None)
        if scene_descriptors is None or len(scene_keypoints) < 4:
            return [], {"scene_keypoints": len(scene_keypoints), "templates_evaluated": 0}

        height, width = frame.shape[:2]
        frame_area = float(height * width)
        candidates: list[Detection] = []
        for template in self.templates:
            pairs = self.matcher.knnMatch(template.descriptors, scene_descriptors, k=2)
            good = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < self.config.placard_ratio_test * second.distance
            ]
            if len(good) < self.config.placard_min_matches:
                continue

            source_points = np.float32(
                [template.keypoints[match.queryIdx].pt for match in good]
            ).reshape(-1, 1, 2)
            destination_points = np.float32(
                [scene_keypoints[match.trainIdx].pt for match in good]
            ).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(
                source_points,
                destination_points,
                cv2.RANSAC,
                self.config.placard_ransac_threshold,
            )
            if homography is None or inlier_mask is None:
                continue
            inliers = int(inlier_mask.ravel().sum())
            inlier_ratio = inliers / len(good)
            if inlier_ratio < self.config.placard_min_inlier_ratio:
                continue

            template_height, template_width = template.image.shape[:2]
            corners = np.float32(
                [
                    [0, 0],
                    [template_width - 1, 0],
                    [template_width - 1, template_height - 1],
                    [0, template_height - 1],
                ]
            ).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
            if not np.isfinite(transformed).all():
                continue
            polygon_int = np.round(transformed).astype(np.int32).reshape(-1, 1, 2)
            if not cv2.isContourConvex(polygon_int):
                continue
            polygon_area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
            area_ratio = polygon_area / frame_area
            if not (
                self.config.placard_min_area_ratio
                <= area_ratio
                <= self.config.placard_max_area_ratio
            ):
                continue
            x, y, box_width, box_height = cv2.boundingRect(polygon_int)
            if box_width <= 0 or box_height <= 0:
                continue
            bbox = BoundingBox(x, y, box_width, box_height)
            match_score = min(len(good) / max(self.config.placard_min_matches * 3, 1), 1.0)
            confidence = min(0.99, 0.35 + 0.35 * inlier_ratio + 0.3 * match_score)
            candidates.append(
                Detection(
                    label=template.name,
                    kind="hazmat",
                    confidence=confidence,
                    bbox=bbox,
                    centroid_normalized=_normalized_centroid(bbox, width, height),
                    polygon=tuple(
                        Point(float(x_value), float(y_value)) for x_value, y_value in transformed
                    ),
                    evidence={
                        "decision_rule": "sift_ratio_plus_ransac",
                        "template": template.path.name,
                        "template_keypoints": len(template.keypoints),
                        "good_matches": len(good),
                        "minimum_matches": self.config.placard_min_matches,
                        "inliers": inliers,
                        "inlier_ratio": round(inlier_ratio, 4),
                        "minimum_inlier_ratio": self.config.placard_min_inlier_ratio,
                        "polygon_area_ratio": round(area_ratio, 6),
                        "ratio_test": self.config.placard_ratio_test,
                        "ransac_reprojection_threshold": self.config.placard_ransac_threshold,
                    },
                )
            )

        kept = self._non_maximum_suppression(candidates)
        return kept, {
            "scene_keypoints": len(scene_keypoints),
            "templates_evaluated": len(self.templates),
            "placard_candidates_before_nms": len(candidates),
        }

    def _non_maximum_suppression(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(
                self._iou(candidate.bbox, existing.bbox) < self.config.placard_nms_iou
                for existing in kept
            ):
                kept.append(candidate)
        return kept

    @staticmethod
    def _iou(first: BoundingBox, second: BoundingBox) -> float:
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = first.width * first.height + second.width * second.height - intersection
        return intersection / union if union > 0 else 0.0
