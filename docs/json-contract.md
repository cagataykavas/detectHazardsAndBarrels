# JSON contract

`events.jsonl` contains one independent JSON object per processed frame. Consumers can
stream it line by line. Additive fields may appear under schema version `1.x`; a
removal or semantic change requires a new major version.

## Frame fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Currently `1.0` |
| `frame_index` | integer | Original source-frame index, even with frame stride |
| `timestamp_ms` | number | Source index divided by reported FPS |
| `processing_ms` | number | Detector and tracker wall time |
| `detections` | array | Objects observed in this processed frame |
| `diagnostics` | object | Mask pixels, SIFT keypoints, and candidate counts |

## Detection fields

| Field | Type | Meaning |
|---|---|---|
| `track_id` | string | Semantic label plus monotonically increasing sequence |
| `first_observation` | boolean | Whether the tracker allocated the ID in this frame |
| `label` | string | Template stem, `red_barrel`, or `blue_barrel` |
| `kind` | enum | `hazmat` or `barrel` |
| `confidence` | number | Heuristic score in `[0, 1]`, not a probability |
| `bbox` | object | Source-pixel `x`, `y`, `width`, and `height` |
| `centroid_normalized` | object | Frame-relative `x` and `y` in `[0, 1]` |
| `polygon` | array | Four source-pixel points for placards, empty for barrels |
| `evidence` | object | Rule measurements, thresholds, and association result |

## Example

```json
{
  "schema_version": "1.0",
  "frame_index": 12,
  "timestamp_ms": 500.0,
  "processing_ms": 18.42,
  "detections": [
    {
      "track_id": "hazmat_flammable-solid_1",
      "first_observation": false,
      "label": "flammable-solid",
      "kind": "hazmat",
      "confidence": 0.86,
      "bbox": {"x": 392, "y": 102, "width": 181, "height": 179},
      "centroid_normalized": {"x": 0.5026, "y": 0.3546},
      "polygon": [
        {"x": 398.2, "y": 103.1},
        {"x": 571.9, "y": 108.4},
        {"x": 566.5, "y": 279.2},
        {"x": 392.8, "y": 274.4}
      ],
      "evidence": {
        "decision_rule": "sift_ratio_plus_ransac",
        "good_matches": 42,
        "minimum_matches": 12,
        "inliers": 33,
        "inlier_ratio": 0.7857,
        "polygon_area_ratio": 0.058,
        "association_rule": "nearest_same_label_centroid"
      }
    }
  ],
  "diagnostics": {
    "barrel_candidates": 2,
    "placard_candidates": 1,
    "scene_keypoints": 613,
    "templates_evaluated": 15,
    "placard_candidates_before_nms": 1
  }
}
```

The values above illustrate the contract and are not benchmark results.

## Summary

`summary.json` records source and processed frame counts, FPS, loaded template names,
unique tracks, per-track and per-kind observation counts, timing, exact configuration,
input metadata, and output filenames. It always carries both a metric caveat and a
safety caveat.

## Consumer rules

- Check `schema_version` before parsing.
- Ignore unknown additive fields.
- Use `kind` before interpreting kind-specific evidence.
- Treat `first_observation` as tracker state, not proof that an object just entered the
  physical scene.
- Do not turn a missing detection into a safety conclusion.
