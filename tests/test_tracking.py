from crisis_vision.config import CrisisConfig
from crisis_vision.models import BoundingBox, Detection, Point
from crisis_vision.tracking import CentroidTracker


def _barrel(x: float) -> Detection:
    return Detection(
        label="red_barrel",
        kind="barrel",
        confidence=0.8,
        bbox=BoundingBox(10, 20, 40, 80),
        centroid_normalized=Point(x, 0.5),
    )


def test_nearby_same_label_detection_keeps_identity():
    tracker = CentroidTracker(CrisisConfig())
    first = tracker.update([_barrel(0.2)])[0]
    second = tracker.update([_barrel(0.22)])[0]

    assert first.track_id == second.track_id
    assert first.first_observation is True
    assert second.first_observation is False
    assert second.evidence["association_rule"] == "nearest_same_label_centroid"
