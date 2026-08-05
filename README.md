# Crisis Vision Lab

[![CI](https://github.com/cagataykavas/detectHazardsAndBarrels/actions/workflows/ci.yml/badge.svg)](https://github.com/cagataykavas/detectHazardsAndBarrels/actions/workflows/ci.yml)

An explainable computer-vision baseline for detecting hazardous-material placards
and colored barrels in crisis-scene video. It combines scale-aware HSV segmentation,
SIFT feature matching, RANSAC homography validation, centroid tracking, and a
versioned JSONL output contract.

The repository includes a deterministic generated scene, so a new contributor can
exercise the whole system without downloading private footage. The original
university prototype remains in [`legacy/HW1_monolith.py`](legacy/HW1_monolith.py)
for provenance and is never imported by the repaired application.

> This is a portfolio and research baseline. It is not a certified safety system and
> must not be the sole basis for emergency response or hazardous-material handling.

## Demonstrated engineering

- resolution-independent red/blue barrel rules with inspectable measurements
- SIFT + Lowe ratio matching and RANSAC inlier validation for placards
- geometric rejection of implausible or non-convex homographies
- semantic track IDs with explicit first-seen and association evidence
- headless video processing with annotated MP4, preview, JSONL, and summary outputs
- generated integration demo, typed configuration, unit tests, linting, and CI

Confidence values are heuristic ranking scores, not calibrated probabilities.
Detection counts are observations, not precision/recall.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Creates a scene with two barrels and a moving placard.
crisis-vision demo --output artifacts/demo --frames 48

# Processes a real recording with the bundled template directory.
crisis-vision analyze \
  --input video.mp4 \
  --templates hazmats/hazmats \
  --output artifacts/incident

# Prints the event contract.
crisis-vision explain-schema
```

The historical filename remains a compatibility entry point:

```bash
python HW1.py demo --output artifacts/demo
```

## Artifact bundle

| File | Contents |
|---|---|
| `events.jsonl` | One explainable result per processed frame |
| `summary.json` | Configuration, input metadata, timings, and observed track counts |
| `annotated.mp4` | Bounding boxes, placard polygons, track IDs, and scores |
| `preview.jpg` | Final annotated frame for fast review |

Each detection records its geometry and decision evidence:

```json
{
  "track_id": "hazmat_flammable-solid_1",
  "label": "flammable-solid",
  "kind": "hazmat",
  "confidence": 0.86,
  "evidence": {
    "decision_rule": "sift_ratio_plus_ransac",
    "good_matches": 42,
    "inlier_ratio": 0.79,
    "polygon_area_ratio": 0.051
  }
}
```

See [Architecture](docs/architecture.md), [JSON contract](docs/json-contract.md),
and [Evaluation guide](docs/evaluation.md).

## Configuration

```bash
cp config.example.json my-config.json
crisis-vision analyze \
  --input video.mp4 \
  --templates hazmats/hazmats \
  --config my-config.json \
  --output artifacts/incident \
  --max-frames 500
```

Unknown configuration keys and invalid ranges fail before video processing.

## Template assets

The historical repository did not record where the bundled placard PNGs came from.
They remain for backwards compatibility and demonstration, but their provenance and
reuse rights must be verified before redistribution or commercial deployment. See
[`ASSET_NOTICE.md`](ASSET_NOTICE.md). The code and documentation are MIT licensed.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
crisis-vision demo --output artifacts/smoke --frames 24 --no-video
```

## Known limitations

- HSV barrels assume visible red or blue paint and relatively stable illumination.
- SIFT matching needs enough texture and can fail under blur, glare, or heavy
  occlusion.
- Similar placards can compete when artwork shares strong local features.
- Centroid association is deliberately simple and can swap IDs at crossings.
- A deployment needs licensed templates, representative labeled footage, calibrated
  thresholds, and human review.

## License

MIT for source code and documentation. Bundled image assets are excluded pending
provenance verification.
