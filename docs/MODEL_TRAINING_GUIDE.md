# Model Training Guide

This repository stores trained model artifacts but does not contain the full training dataset. Treat this guide as the handover checklist for future model work.

## Current Models

```text
models/ruck.pt       ruck formation detector
models/lineout.pt    lineout and hooker detector
models/ball.pt       rugby ball detector
models/player-id.pt  player/person/referee detector
```

## Recommended Dataset Coverage

Future training data should include:

- broadcast and sideline camera angles
- close and wide shots
- day, night, rain, glare, and shadow conditions
- different jersey colour combinations
- rucks with heavy occlusion
- lineouts near both sidelines
- referees and assistant referees
- clips where no offside event is present

## Label Quality

Use consistent labels that match `src/utils/constants.py`. Any class remapping must be updated in that file and in tests.

Recommended player labels:

- `player` or `person`
- `ref`, `referee`, or `official`

Recommended event labels:

- `Ruck`
- `Lineout`
- `Hooker` when available in lineout data

## Training Workflow

- See [Team29-Capstone-Training](https://github.com/hasmar04/Team29-Capstone-Training) for additional instructions

1. Collect representative footage and still frames.
2. Split into train/validation/test sets by match or clip, not by adjacent frames only.
3. Annotate boxes consistently.
4. Train YOLO with Ultralytics.
5. Validate on clips not used for training.
6. Export `.pt` weights.
7. Replace files in `models/`.
8. Run backend tests.
9. Run at least one short manual visual regression clip.
10. Document the dataset version, class map, and known failure cases.

Example training command shape:

```bash
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

Adjust model size and epochs for available hardware.

## Evaluation Checklist

Track these metrics before replacing a model:

- precision and recall per class
- confusion between ruck and lineout formations
- ball detection under occlusion
- player detection at long range
- referee false positives
- inference speed on the target laptop/GPU

## Integration Checks

After adding a new model:

- Confirm file name matches the loader paths.
- Confirm class IDs match `constants.py`.
- Run `python -m pytest`.
- Process a known clip through GUI batch mode.
- Inspect `*_analysis.json` for expected event types and confidence values.
- Compare annotated MP4 output against the old model on the same clip.

## Known Model Limitations

Model outputs are candidate detections. Final offside interpretation still depends on geometry, camera angle, and human review.
