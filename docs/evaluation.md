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
