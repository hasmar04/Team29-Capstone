# System Architecture

## Purpose

The system assists rugby offside review by detecting ruck and lineout events, estimating offside lines, identifying players between those lines, and producing outputs for a desktop GUI and future frontend layers.

## High-Level Flow

```text
Video input
  -> YOLO event inference
  -> event validation and cooldowns
  -> field-line detection
  -> offside-line geometry
  -> player detection
  -> team classification
  -> ByteTrack tracking
  -> offside player filtering
  -> annotated video, text report, JSON payload
```

## Module Boundaries

```text
src/detection/
  YOLO loading/inference and event-specific model helpers.

src/classification/
  Player box filtering, jersey colour extraction, team classification, and
  offside geometry filtering.

src/tracking/
  ByteTrack player ID persistence and team-label stabilisation.

src/processing/
  CLI, GUI orchestration, and batch processing.

src/logging/
  Structured event and session statistics dataclasses.

src/utils/
  Field lines, geometry, drawing, UI helpers, constants, and video utilities.
```

Legacy modules at `src/*.py` mostly wrap or duplicate older implementations. Prefer the modular folders for new work.

## Event Detection

Ruck and lineout detections are produced by separate YOLO models. Batch and auto modes use consecutive-frame thresholds to reduce false positives and cooldown windows to avoid duplicate events.

When ruck and lineout candidates overlap in the same frame, the pipeline keeps the candidate with the higher confidence score.

## Field Geometry

Field-line detection uses OpenCV image processing and line intersection helpers. Ruck offside lines are projected from the last feet of the ruck to the estimated field-line intersection. Lineout offside lines use estimated field points and homography-based offside points.

## Player Layer

Players are detected by the player model, converted into bottom-centre coordinates, assigned jersey colours, clustered into teams, and passed through ByteTrack. Bottom-centre points are used for offside checks because they best approximate the player contact point on the field.

## Integration Outputs

The main integration contract is the batch JSON payload:

```text
events[] + summary + logs + output_files
```

The GUI can read JSON directly and still supports older text report parsing as a fallback.

## Key Risks

- Camera angle and field-line visibility strongly affect geometry accuracy.
- Similar jersey colours can confuse team classification.
- Heavy occlusion can cause track ID switches.
- Model confidence is not a calibrated legal decision confidence.
