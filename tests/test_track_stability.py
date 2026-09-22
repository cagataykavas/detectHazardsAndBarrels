import json

import pytest

from crisis_vision.cli import main
from crisis_vision.track_stability import (
    TrackAuditInputError,
    TrackStabilityPolicy,
    audit_track_stability,
)


def _detection(
    *,
    track_id: str = "red_barrel_1",
    label: str = "red_barrel",
    confidence: float = 0.8,
    x: float = 0.2,
    first: bool = False,
) -> dict:
    return {
        "track_id": track_id,
        "label": label,
        "kind": "barrel",
        "confidence": confidence,
        "centroid_normalized": {"x": x, "y": 0.5},
        "first_observation": first,
    }


def _events(*detections: dict, frames: tuple[int, ...] | None = None) -> list[dict]:
    frame_indexes = frames or tuple(range(len(detections)))
    return [
        {"frame_index": frame, "detections": [detection]}
        for frame, detection in zip(frame_indexes, detections, strict=True)
    ]


def test_stable_track_passes_with_deterministic_evidence():
    report = audit_track_stability(
        _events(
            _detection(first=True, x=0.20, confidence=0.82),
            _detection(x=0.23, confidence=0.79),
            _detection(x=0.27, confidence=0.81),
        )
    )

    assert report["passed"] is True
    assert report["reason_counts"] == {}
    assert report["tracks"][0] == {
        "track_id": "red_barrel_1",
        "label": "red_barrel",
        "kind": "barrel",
        "passed": True,
        "reasons": [],
        "observation_count": 3,
        "first_frame": 0,
        "last_frame": 2,
        "max_frame_gap": 1,
        "mean_confidence": 0.806667,
        "max_confidence_drop": 0.03,
        "max_normalized_speed_per_frame": 0.04,
    }


def test_all_temporal_policy_violations_are_reported():
    policy = TrackStabilityPolicy(
        min_observations=3,
        max_frame_gap=2,
        min_mean_confidence=0.75,
        max_confidence_drop=0.2,
        max_normalized_speed_per_frame=0.05,
    )
    report = audit_track_stability(
        _events(
            _detection(first=True, x=0.1, confidence=0.9),
            _detection(x=0.6, confidence=0.4),
            frames=(0, 4),
        ),
        policy,
    )

    assert report["passed"] is False
    assert report["tracks"][0]["reasons"] == [
        "insufficient_observations",
        "frame_gap_exceeded",
        "mean_confidence_below_minimum",
        "confidence_drop_exceeded",
        "normalized_speed_exceeded",
    ]


@pytest.mark.parametrize(
    ("detections", "reason"),
    [
        ((_detection(), _detection()), "missing_track_origin"),
        (
            (_detection(first=True), _detection(first=True), _detection()),
            "repeated_track_origin",
        ),
        (
            (_detection(first=True), _detection(), _detection(label="blue_barrel")),
            "track_identity_changed",
        ),
    ],
)
def test_track_lifecycle_contract_is_audited(detections, reason):
    report = audit_track_stability(_events(*detections))

    assert report["passed"] is False
    assert reason in report["tracks"][0]["reasons"]


def test_tracks_and_reason_counts_are_sorted():
    first = _events(
        _detection(track_id="z", first=True),
        _detection(track_id="z"),
        _detection(track_id="z"),
    )
    second = _events(
        _detection(track_id="a", first=True, confidence=0.1),
        _detection(track_id="a", confidence=0.1),
        _detection(track_id="a", confidence=0.1),
    )
    combined = [
        {
            "frame_index": index,
            "detections": first[index]["detections"] + second[index]["detections"],
        }
        for index in range(3)
    ]

    report = audit_track_stability(combined)

    assert [track["track_id"] for track in report["tracks"]] == ["a", "z"]
    assert report["reason_counts"] == {"mean_confidence_below_minimum": 1}


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"frame_index": 0, "detections": []}],
        [{"frame_index": 1, "detections": []}, {"frame_index": 1, "detections": []}],
        _events(_detection(first=True, confidence=float("nan"))),
        _events(_detection(first=True, x=1.1)),
        [
            {
                "frame_index": 0,
                "detections": [
                    _detection(first=True),
                    _detection(first=True),
                ],
            }
        ],
    ],
)
def test_malformed_or_insufficient_artifacts_fail_closed(events):
    with pytest.raises(TrackAuditInputError):
        audit_track_stability(events)


def test_invalid_policy_fails_closed():
    with pytest.raises(TrackAuditInputError, match="min_observations"):
        audit_track_stability(
            _events(_detection(first=True), _detection()),
            TrackStabilityPolicy(min_observations=1),
        )


def test_cli_returns_distinct_policy_exit_code(tmp_path, capsys):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(event) for event in _events(_detection(first=True))) + "\n"
    )

    exit_code = main(["audit-tracks", "--events", str(events_path), "--require-pass"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 3
    assert report["passed"] is False


def test_cli_rejects_invalid_jsonl(tmp_path, capsys):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("not-json\n")

    exit_code = main(["audit-tracks", "--events", str(events_path)])

    assert exit_code == 2
    assert "not valid JSON" in capsys.readouterr().err
