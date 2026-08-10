# Hazard Sign & Barrel Detection / Tracking

A classical computer-vision pipeline for detecting and tracking hazardous-material signs and colored barrels in video.

The project combines **SIFT feature matching**, **homography estimation**, **HSV color segmentation**, **morphological filtering**, **contour analysis**, and lightweight **centroid-based tracking**. It also provides a live visualization panel for detected HAZMAT templates, confidence indicators, and cumulative detection counts.

> This repository is an academic computer-vision project. It is preserved as an example of a feature-engineering / classical-CV approach rather than a modern deep-learning detector.

## What the pipeline does

Given an input video, the program:

1. Loads reference HAZMAT sign images from the `hazmats/` directory.
2. Extracts SIFT keypoints and descriptors for each template.
3. Processes video frames at a configurable interval.
4. Searches the central region of each processed frame for matching HAZMAT signs.
5. Uses Lowe's ratio test and RANSAC homography estimation to validate template matches.
6. Detects red and blue barrels using HSV color thresholds and contour filtering.
7. Associates detections across frames using centroid-distance tracking.
8. Maintains counts for recognized HAZMAT signs.
9. Displays the processed video alongside a template/status panel.

## Techniques demonstrated

- OpenCV video processing
- SIFT feature extraction
- Brute-force descriptor matching
- Lowe ratio test
- Homography estimation with RANSAC
- Perspective transformation
- HSV color-space segmentation
- Morphological opening / closing
- Contour extraction and area filtering
- Otsu thresholding for ROI refinement
- Centroid-based object association
- Frame skipping for performance
- Real-time visualization with OpenCV

## Repository structure

```text
detectHazardsAndBarrels/
├── HW1.py              # Detection, tracking and visualization pipeline
├── hazmats/            # Reference HAZMAT sign images
├── requirements.txt    # Python dependencies
├── ReadMe.txt           # Original assignment notes
└── README.md            # Project documentation
```

The input video is expected as `video.mp4` in the project directory, as described by the original assignment notes.

## Installation

Python 3 is required.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Input data

Place the video to be processed in the repository root:

```text
video.mp4
```

Reference HAZMAT images should be PNG files inside the template directory used by the script:

```text
hazmats/
└── hazmats/
    ├── <template-1>.png
    ├── <template-2>.png
    └── ...
```

The filename (without `.png`) is used as the object/template name.

## Running

```bash
python HW1.py
```

The application opens an OpenCV window containing the processed video and a HAZMAT status panel.

## Detection architecture

### HAZMAT signs

Each reference sign is represented using SIFT descriptors. Scene descriptors are matched against the templates using a brute-force L2 matcher followed by Lowe's ratio test.

When enough good matches are available, the corresponding template points and scene points are used to estimate a homography with RANSAC. The transformed template boundary is then subjected to geometric checks before being accepted as a detection.

The implementation also performs segmentation within the detected region to refine the object's visible contour.

### Barrels

Barrels are detected independently using color information. Frames are converted from BGR to HSV and thresholded for red and blue regions. Morphological filtering suppresses small artifacts, after which sufficiently large contours become barrel detections.

### Tracking

The project uses a lightweight tracking strategy based primarily on centroid proximity and disappearance counters. This avoids requiring a dedicated tracking model while demonstrating basic multi-frame object association.

## Important configuration values

Several thresholds are intentionally exposed near the top of `HW1.py`, including:

- minimum feature-match count
- frame-skip interval
- object disappearance threshold
- centroid recognition distance
- visualization grid dimensions

The barrel detector also contains class-specific contour-area thresholds and HSV ranges.

These values were tuned for the assignment footage rather than designed as universal detection parameters.

## Limitations

This is a handcrafted CV pipeline, so performance is sensitive to the visual conditions of the input video.

In particular:

- SIFT matching can deteriorate under severe blur, occlusion, or weak texture.
- Fixed HSV thresholds depend on lighting and camera characteristics.
- Fixed contour-area thresholds depend on scale and camera distance.
- Centroid association can confuse objects that cross or move abruptly.
- Several geometric and matching thresholds are dataset-specific.
- The system is not intended as a safety-certified hazard-recognition system.

A modern extension could replace the handcrafted detection stages with a trained object detector while retaining this repository as a useful classical-CV baseline.

## Why this project is useful

The project demonstrates an end-to-end computer-vision workflow without relying on a pretrained neural detector: feature extraction, matching, geometric verification, segmentation, object association, counting, and video visualization are implemented explicitly with NumPy and OpenCV.
