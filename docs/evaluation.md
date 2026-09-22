# Evaluation guide

The generated demo proves integration behavior; it does not measure real-world
accuracy. A defensible evaluation needs representative labeled footage and a frozen
protocol.

## Dataset card minimum

Record:

- source, owner, license, and consent basis;
- camera model, resolution, frame rate, and compression;
- environments, lighting, weather, distance, and viewing angle;
- placard classes and barrel colors;
- class counts, occlusion levels, blur, glare, and negative scenes;
- split method and checks for adjacent-frame leakage;
- labeling instructions, reviewer process, and known ambiguity.

Do not place proprietary or identifiable footage in this public repository.

## Recommended splits

Split by recording session or physical scene—not random frames. Neighboring frames are
nearly duplicates and would inflate a random-frame test score. Keep a held-out stress
set for low light, blur, partial occlusion, compression, and look-alike artwork.

## Detection metrics

Report at minimum:

- precision, recall, and F1 by class;
- average precision at documented IoU thresholds;
- false positives per minute on negative footage;
- miss rate by distance and occlusion bucket;
- median and p95 processing latency on named hardware.

For homography quality, also report corner reprojection error. For tracking, report ID
switches and track fragmentation. Include confidence intervals or bootstrap intervals
when sample counts are small.

## Temporal consistency preflight

Before scoring against labels, the emitted evidence can be checked for internal track
stability:

```bash
crisis-vision audit-tracks \
  --events artifacts/incident/events.jsonl \
  --min-observations 3 \
  --max-frame-gap 8 \
  --min-mean-confidence 0.45 \
  --max-confidence-drop 0.35 \
  --max-normalized-speed-per-frame 0.12 \
  --require-pass > artifacts/incident/track-stability.json
```

Exit code `0` means the artifact passed (or no gate was requested), `3` means a valid
artifact violated at least one configured threshold, and `2` means the artifact or
policy was malformed. Store the report beside the run configuration so reviewers can
reproduce the decision.

This preflight is deliberately not an ID-switch metric: without labeled object
identities it cannot know whether a stable-looking track follows the correct object.
Normalized speed is also affected by camera motion, perspective, sampling stride, and
frame rate. Calibrate every threshold on representative validation recordings and use
labeled MOT-style evaluation for release claims.

## Threshold selection

Tune HSV, ratio-test, RANSAC, and tracking thresholds only on training/validation data.
Freeze them before running the held-out test. Save the exact JSON configuration beside
every evaluation result.

## Deployment gate

Before any operational experiment:

1. verify template rights and source provenance;
2. evaluate representative cameras and environments;
3. define a human-review workflow and safe failure state;
4. log model/config versions and input health;
5. perform privacy, security, and misuse review; and
6. state clearly that the baseline is advisory, not safety certified.
