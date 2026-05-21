"""
Batch processing pipeline for rugby offside detection.

The functions in this module are intentionally GUI-safe: they process clips
sequentially, return structured dictionaries, and write both human-readable text
reports and JSON summaries for frontend integration.
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.classification import offside_functions as offside
from src.classification import player_detection as player
from src.detection import lineout_functions as lineout
from src.detection import ruck_functions as ruck
from src.detection import yolo_functions as YOLO
from src.tracking import player_tracking as tracker
from src.utils import drawing_functions as draw
from src.utils import field_functions as field
from src.utils import general_functions as general
from src.utils import line_functions as line
from src.utils import point_functions as points
from src.utils.constants import LINEOUT_MODEL_CLASS_NUMBERS, RUCK_MODEL_CLASS_NUMBERS


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".gif", ".MP4", ".AVI", ".MOV", ".GIF"}


def make_log_entry(stage, message, level="INFO", event_type=None, frame_number=None,
                   confidence=None, player_ids=None, error=None):
    """
    Build a frontend-safe processing log entry.

    Parameters:
        stage (str): Pipeline stage that produced the log.
        message (str): Human-readable log message.
        level (str, optional): Severity label. Defaults to ``"INFO"``.
        event_type (str, optional): Related event type, such as ``"ruck"``.
        frame_number (int, optional): Related frame number.
        confidence (float, optional): Confidence score in the 0.0-1.0 range.
        player_ids (list, optional): ByteTrack IDs related to the event.
        error (Exception or str, optional): Error details for failed stages.

    Returns:
        dict: JSON-serialisable log entry.

    Error handling:
        Optional values are normalised to safe JSON values. The function itself
        does not raise unless non-serialisable custom objects are passed in
        ``player_ids``.

    Example:
        ``make_log_entry("tracking", "Updated tracks", player_ids=[2, 3])``
    """
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "stage": stage,
        "message": message,
        "event_type": event_type,
        "frame_number": frame_number,
        "confidence": confidence,
        "player_ids": player_ids or [],
        "error": str(error) if error else None,
    }


def get_video_files(directory_path):
    """
    Retrieve supported video files from a directory.

    Parameters:
        directory_path (str): Path to a directory containing clips.

    Returns:
        list: Sorted absolute video file paths.

    Error handling:
        Raises ``FileNotFoundError`` when the directory does not exist.

    Example:
        ``clips = get_video_files("./inference/general")``
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    return sorted(
        str(file_path.absolute())
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix in VIDEO_EXTENSIONS
    )


def build_summary(detections_data):
    """
    Calculate summary statistics from a clip result.

    Parameters:
        detections_data (dict): Detection payload containing an ``events`` list.

    Returns:
        dict: Summary counts and average confidence.

    Error handling:
        Missing keys are treated as empty values.
    """
    events = detections_data.get("events", [])
    ruck_events = [e for e in events if e.get("type") == "ruck"]
    lineout_events = [e for e in events if e.get("type") == "lineout"]
    confidences = [float(e.get("detection_confidence", 0.0)) for e in events]

    return {
        "total_events": len(events),
        "ruck_count": len(ruck_events),
        "lineout_count": len(lineout_events),
        "total_offside_players": sum(int(e.get("offside_count", 0)) for e in events),
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def build_frontend_payload(video_path, detections_data, output_files=None):
    """
    Normalise clip results for GUI/frontend consumption.

    Parameters:
        video_path (str): Source video path.
        detections_data (dict): Raw detection result from ``auto_mode_batch``.
        output_files (dict, optional): Generated report/video/JSON paths.

    Returns:
        dict: Stable integration payload with events, logs, summary, and files.

    Error handling:
        Missing detection fields are filled with safe defaults.
    """
    payload = dict(detections_data)
    payload.setdefault("video_name", Path(video_path).stem)
    payload.setdefault("source_video", str(Path(video_path).absolute()))
    payload.setdefault("processing_date", datetime.now().isoformat(timespec="seconds"))
    payload.setdefault("total_frames", 0)
    payload.setdefault("fps", 0)
    payload.setdefault("events", [])
    payload.setdefault("logs", [])
    payload["summary"] = build_summary(payload)
    payload["output_files"] = output_files or payload.get("output_files", {})
    return payload


def write_json_summary(output_path, detections_data):
    """
    Write a structured JSON summary for frontend integration.

    Parameters:
        output_path (str or Path): Destination JSON file.
        detections_data (dict): JSON-serialisable result payload.

    Returns:
        str: Absolute path to the written JSON file.

    Error handling:
        Propagates file-system and serialisation errors to the caller.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2)

    return str(output_path.absolute())


def _format_report_confidence(value):
    """Return a report-friendly confidence percentage."""
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def generate_summary_report(video_path, output_path, detections_data):
    """
    Generate a human-readable text summary report for a processed video.

    Parameters:
        video_path (str): Source video path.
        output_path (str): Destination text report path.
        detections_data (dict): Detection data containing ``events``.

    Returns:
        None

    Error handling:
        Propagates ``OSError`` if the report cannot be written.
    """
    video_name = Path(video_path).name
    timestamp = detections_data.get(
        "processing_date",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    events = detections_data.get("events", [])
    summary = build_summary(detections_data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("QUT P541 RUGBY OFFSIDE REVIEW REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("CLIP SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Processed: {timestamp}\n")
        f.write(f"Frames analysed: {detections_data.get('total_frames', 0)}\n")
        f.write(f"Frame rate: {detections_data.get('fps', 'N/A')} fps\n")
        f.write(f"Events detected: {summary['total_events']}\n")
        f.write(f"Candidate offside players: {summary['total_offside_players']}\n")
        f.write(f"Average event confidence: {_format_report_confidence(summary['average_confidence'])}\n\n")

        f.write("EVENT BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Rucks: {summary['ruck_count']}\n")
        f.write(f"Lineouts: {summary['lineout_count']}\n\n")

        if not events:
            f.write("No candidate ruck or lineout offside events were recorded for this clip.\n\n")
        else:
            f.write("DETAILED EVENTS\n")
            f.write("-" * 80 + "\n")

            for i, event in enumerate(events, 1):
                event_type = event.get("type", "unknown").upper()
                confidence = _format_report_confidence(event.get("detection_confidence"))
                counts = event.get("team_counts", {})

                f.write(f"{i}. {event_type}\n")
                f.write(f"   Time: {event.get('timestamp', '0.00s')} | Frame: {event.get('frame_number', 0)}\n")
                f.write(f"   Confidence: {confidence}\n")
                f.write(f"   Offside players: {event.get('offside_count', 0)}\n")
                f.write(
                    "   Visible team counts: "
                    f"team_0={counts.get('team_0', 0)}, "
                    f"team_1={counts.get('team_1', 0)}, "
                    f"unknown={counts.get('unknown_team', 0)}, "
                    f"refs={counts.get('refs', 0)}\n"
                )

                if event.get("offside_players"):
                    f.write("   Offside player positions:\n")
                    for j, player_info in enumerate(event["offside_players"], 1):
                        f.write(
                            f"     - Player {j}: "
                            f"({player_info.get('x')}, {player_info.get('y')}), "
                            f"confidence {_format_report_confidence(player_info.get('confidence'))}\n"
                        )
                else:
                    f.write("   No players were classified as offside for this event.\n")

                f.write("\n")

        f.write("REVIEW NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("This report identifies candidate events for review. It is not a referee decision.\n")
        f.write("Use the annotated video to inspect the field-line guides, intersection point, offside lines, and player boxes.\n")
        f.write("Low-confidence detections should be treated as prompts for manual review.\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Summary report saved: {output_path}")


def save_annotated_video(frames_data, output_path, fps, frame_size):
    """
    Save annotated frames to a video file.

    Parameters:
        frames_data (list): ``(frame_number, frame)`` tuples.
        output_path (str): Destination MP4 path.
        fps (int): Output frame rate.
        frame_size (tuple): ``(width, height)`` output dimensions.

    Returns:
        None

    Error handling:
        Raises ``ValueError`` for empty frame data and ``RuntimeError`` when
        OpenCV cannot create the writer.
    """
    if not frames_data:
        raise ValueError("frames_data cannot be empty")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for: {output_path}")

    for _, frame in frames_data:
        if frame is None or getattr(frame, "size", 0) == 0:
            continue
        out.write(frame)

    out.release()
    print(f"Annotated video saved: {output_path}")


def _safe_next(generator):
    """Return the next generator item or ``None`` when exhausted."""
    try:
        return next(generator)
    except StopIteration:
        return None


def _serialise_players(players):
    """Convert player dictionaries into JSON-safe tracking summaries."""
    return [
        {
            "track_id": p.get("track_id"),
            "team": p.get("team"),
            "box": list(p.get("box", ())),
            "bottom_centre": list(p.get("bottom_centre", ())),
            "confidence": float(p.get("confidence", 0.0)),
        }
        for p in players
    ]


def _draw_event_header(frame, event_type, confidence, offside_count):
    """
    Draw a compact event label on an annotated frame.

    Parameters:
        frame (np.ndarray): Frame to annotate.
        event_type (str): Event type label.
        confidence (float): Detection confidence.
        offside_count (int): Number of offside players.

    Returns:
        np.ndarray: Frame with a top-left event label.

    Error handling:
        Assumes a valid OpenCV frame. Invalid frames will raise through OpenCV.
    """
    label = f"{event_type.upper()} | confidence {confidence:.0%} | offside players {offside_count}"
    cv2.rectangle(frame, (12, 12), (520, 46), (30, 30, 30), -1)
    cv2.putText(
        frame,
        label,
        (22, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _draw_geometry_overlay(frame, field_lines, intersection_point, anchor_points,
                           offside_lines, event_type):
    """
    Draw field-line guides, intersection, anchor points, and offside lines.

    Parameters:
        frame (np.ndarray): Frame to annotate.
        field_lines (list): Detected pitch guide lines.
        intersection_point (tuple): Point where guide lines converge.
        anchor_points (list): Event-specific points, such as lineout centre and
            projected offside points.
        offside_lines (list): Final offside boundary lines.
        event_type (str): Event label used to choose line colour.

    Returns:
        np.ndarray: Annotated frame showing the geometry construction.

    Error handling:
        Invalid optional geometry values are skipped so one bad guide line does
        not prevent the final offside overlay from being drawn.
    """
    overlay = frame.copy()

    for guide in field_lines or []:
        if guide is None or len(guide) != 4:
            continue
        x1, y1, x2, y2 = map(int, guide)
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 220, 220), 2, cv2.LINE_AA)

    if intersection_point is not None:
        ix, iy = map(int, intersection_point)
        cv2.circle(overlay, (ix, iy), 8, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            "field-line intersection",
            (max(8, ix + 10), max(20, iy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for label, point, colour in anchor_points:
        if point is None:
            continue
        px, py = map(int, point)
        cv2.circle(overlay, (px, py), 7, colour, -1)
        cv2.putText(
            overlay,
            label,
            (px + 8, max(18, py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    if event_type == "lineout" and intersection_point is not None:
        centre_point = next(
            (point for label, point, _ in anchor_points if label == "lineout centre"),
            None,
        )
        if centre_point is not None:
            centre_ray = _clip_line_to_frame(
                (intersection_point[0], intersection_point[1], centre_point[0], centre_point[1]),
                frame.shape[1],
                frame.shape[0],
            )
            x1, y1, x2, y2 = map(int, centre_ray)
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    offside_colour = (255, 0, 0) if event_type == "lineout" else (0, 0, 255)
    for boundary in offside_lines:
        if boundary is None or len(boundary) != 4:
            continue
        x1, y1, x2, y2 = map(int, boundary)
        cv2.line(overlay, (x1, y1), (x2, y2), offside_colour, 5, cv2.LINE_AA)

    return overlay


def _lineout_perspective_lines(intersection_point, lineout_centre, lineout_box,
                               field_outline, frame_size):
    """
    Build lineout offside lines that share the field-line vanishing point.

    Parameters:
        intersection_point (tuple): Detected field-line convergence point.
        lineout_centre (tuple): Centre point of the lineout on the image.
        lineout_box (tuple): Converted lineout detection box.
        field_outline (np.ndarray): Field outline used for clipping.
        frame_size (tuple): ``(width, height)`` of the annotation frame.

    Returns:
        tuple: ``(left_line, right_line, left_point, right_point)``.

    Error handling:
        Falls back to unclipped perspective rays when field-outline clipping
        cannot produce valid intersections.
    """
    ix, iy = map(float, intersection_point)
    cx, cy = map(float, lineout_centre)
    dx = cx - ix
    dy = cy - iy
    norm = float(np.hypot(dx, dy)) or 1.0

    # Normal to the perspective ray. This creates two guide points that still
    # project back to the same field-line intersection.
    nx = -dy / norm
    ny = dx / norm

    box_width = abs(float(lineout_box[2]) - float(lineout_box[0])) if lineout_box else 0.0
    frame_width, frame_height = frame_size
    offset = max(box_width * 0.65, frame_width * 0.055, 36.0)

    left_point = (int(cx - nx * offset), int(cy - ny * offset))
    right_point = (int(cx + nx * offset), int(cy + ny * offset))

    left_ray = (int(ix), int(iy), left_point[0], left_point[1])
    right_ray = (int(ix), int(iy), right_point[0], right_point[1])

    left_line = field.fit_line_to_field(left_ray, field_outline)
    right_line = field.fit_line_to_field(right_ray, field_outline)

    if left_line == left_ray:
        left_line = _clip_line_to_frame(left_ray, frame_width, frame_height)
    if right_line == right_ray:
        right_line = _clip_line_to_frame(right_ray, frame_width, frame_height)

    return left_line, right_line, left_point, right_point


def _clip_line_to_frame(line_coords, frame_width, frame_height):
    """Clip an infinite line to the frame rectangle."""
    x1, y1, x2, y2 = map(float, line_coords)
    candidates = []

    if x2 != x1:
        for x in (0.0, float(frame_width - 1)):
            t = (x - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            if 0 <= y <= frame_height - 1:
                candidates.append((int(x), int(y)))

    if y2 != y1:
        for y in (0.0, float(frame_height - 1)):
            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            if 0 <= x <= frame_width - 1:
                candidates.append((int(x), int(y)))

    unique = []
    for point in candidates:
        if point not in unique:
            unique.append(point)

    if len(unique) >= 2:
        return [unique[0][0], unique[0][1], unique[1][0], unique[1][1]]

    return list(map(int, line_coords))


def _build_full_length_annotated_frames(video_path, overlay_records):
    """
    Build a full-length annotated video frame list from static event overlays.

    Parameters:
        video_path (str): Source video path.
        overlay_records (list): Dictionaries containing ``start``, ``end``, and
            geometry keys. The latest active overlay is redrawn per frame.

    Returns:
        list: ``(frame_number, frame)`` tuples covering the full source clip.

    Error handling:
        Raises ``RuntimeError`` if the source video cannot be opened.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video for annotation: {video_path}")

    annotated_frames = []
    frame_number = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_number += 1
        output_frame = cv2.resize(frame, (800, 450))

        active_overlay = None
        for record in overlay_records:
            if record["start"] <= frame_number <= record["end"]:
                active_overlay = record

        if active_overlay is not None:
            output_frame = _draw_geometry_overlay(
                output_frame,
                active_overlay["field_lines"],
                active_overlay["intersection_point"],
                active_overlay["anchor_points"],
                active_overlay["offside_lines"],
                active_overlay["event_type"],
            )
            output_frame = draw.draw_boxes(
                output_frame,
                active_overlay["offside_boxes"],
                box_annotation="Offside",
                outline_colour=(0, 0, 255),
                show_image=False,
                font_scale=0.6,
                font_thickness=2,
            )
            output_frame = _draw_event_header(
                output_frame,
                active_overlay["event_type"],
                active_overlay["confidence"],
                len(active_overlay["offside_boxes"]),
            )

        annotated_frames.append((frame_number, output_frame))

    capture.release()
    return annotated_frames


def _build_player_pipeline(frame, player_model, player_tracker):
    """Run player detection, team classification, and ByteTrack tracking."""
    player_result = _safe_next(YOLO.perform_inference(frame, player_model, False))
    players = player.build_player_data(frame, player_result)
    players = player.assign_teams_by_colour(players)
    return player_tracker.update(players)


def auto_mode_batch(video_path, output_dir, ruck_model, lineout_model, ball_model, player_model):
    """
    Process one video in headless automatic mode.

    Parameters:
        video_path (str): Input video file.
        output_dir (str): Output directory, retained for compatibility.
        ruck_model (YOLO): Ruck detector.
        lineout_model (YOLO): Lineout detector.
        ball_model (YOLO): Ball detector. Currently consumed to keep inference
            streams aligned for future ball-release logic.
        player_model (YOLO): Player detector.

    Returns:
        tuple: ``(detections_data, annotated_frames, fps)``.

    Error handling:
        Recoverable frame, field-line, and geometry failures are logged and the
        clip continues. Unrecoverable model/video errors propagate to the batch
        orchestrator so the next clip can continue.
    """
    del output_dir
    video_name = Path(video_path).stem
    fps = int(general.get_video_fps(video_path) or 25)
    frame_threshold = max(1, fps // 10)
    frame_number = 0
    ruck_frame_count = 0
    lineout_frame_count = 0
    frame_buffer = deque(maxlen=max(1, fps * 3))
    player_tracker = tracker.PlayerTracker()

    detections_data = {
        "video_name": video_name,
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_frames": 0,
        "fps": fps,
        "events": [],
        "logs": [],
    }

    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
    success, _ = video_capture.read()
    video_capture.release()

    ui_thresh = 120
    if not success:
        warning = f"Could not read threshold frame from {video_path}; using default threshold"
        print(f"Warning: {warning}")
        detections_data["logs"].append(make_log_entry("threshold", warning, level="WARNING"))

    print(f"Processing: {video_name}")
    print(f"  FPS: {fps}, Frames: {total_frames}, Auto Threshold: {ui_thresh}")

    ruck_results = YOLO.perform_inference(video_path, ruck_model, False, 0.1)
    lineout_results = YOLO.perform_inference(video_path, lineout_model, False, 0.1)
    ball_results = YOLO.perform_inference(video_path, ball_model, False, 0.4)
    overlay_records = []

    print("  Video inference ready. Analysing...")

    for ruck_result, lineout_result, _ in zip(ruck_results, lineout_results, ball_results):
        frame_number += 1

        if not np.array_equal(ruck_result.orig_img, lineout_result.orig_img):
            warning = f"Generators out of sync at frame {frame_number}"
            print(f"Warning: {warning}")
            detections_data["logs"].append(
                make_log_entry("inference", warning, level="WARNING", frame_number=frame_number)
            )
            continue

        frame = ruck_result.orig_img
        if frame is None or getattr(frame, "size", 0) == 0:
            warning = f"Invalid frame at {frame_number}; skipping"
            print(f"Warning: {warning}")
            detections_data["logs"].append(
                make_log_entry("frame_read", warning, level="WARNING", frame_number=frame_number)
            )
            continue

        frame_buffer.append((frame_number, frame.copy()))

        ruck_boxes_raw, ruck_classes, ruck_conf_raw = general.get_class_detections(ruck_result)
        lineout_boxes_raw, lineout_classes, lineout_conf_raw = general.get_class_detections(lineout_result)

        ruck_boxes = [
            box for box, cls in zip(ruck_boxes_raw, ruck_classes)
            if cls == RUCK_MODEL_CLASS_NUMBERS["Ruck"]
        ]
        lineout_boxes = [
            box for box, cls in zip(lineout_boxes_raw, lineout_classes)
            if cls == LINEOUT_MODEL_CLASS_NUMBERS["Lineout"]
        ]
        ruck_confidences = [
            conf for conf, cls in zip(ruck_conf_raw, ruck_classes)
            if cls == RUCK_MODEL_CLASS_NUMBERS["Ruck"]
        ]
        lineout_confidences = [
            conf for conf, cls in zip(lineout_conf_raw, lineout_classes)
            if cls == LINEOUT_MODEL_CLASS_NUMBERS["Lineout"]
        ]

        if ruck_frame_count < 0:
            ruck_frame_count += 1
        if lineout_frame_count < 0:
            lineout_frame_count += 1

        if not ruck_boxes and not lineout_boxes:
            ruck_frame_count = 0 if ruck_frame_count > 0 else ruck_frame_count
            lineout_frame_count = 0 if lineout_frame_count > 0 else lineout_frame_count
            continue

        if ruck_boxes and lineout_boxes:
            if np.max(ruck_confidences) >= np.max(lineout_confidences):
                lineout_frame_count = 0 if lineout_frame_count > 0 else lineout_frame_count
                ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
            else:
                ruck_frame_count = 0 if ruck_frame_count > 0 else ruck_frame_count
                lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count
            continue

        if ruck_boxes:
            ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
        if lineout_boxes:
            lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count

        if ruck_frame_count >= frame_threshold:
            ruck_frame_count = -(fps * 3) + frame_threshold
            confidence = float(np.max(ruck_confidences))
            print(f"  Frame {frame_number}: Ruck detected (confidence: {confidence:.2%})")

            try:
                ruck_box = ruck_boxes[int(np.argmax(ruck_confidences))]
                imsize = (frame.shape[1], frame.shape[0])
                ruck_box_converted = general.convert_coordinates(tuple(ruck_box), imsize)
                resized_frame = cv2.resize(frame, (800, 450))
                ruck_left, ruck_right = ruck.get_last_feet(ruck_box_converted)

                field_lines, field_outline = field.get_field_lines(
                    resized_frame, thresh=ui_thresh, is_path=False, visualise_steps=False
                )
                if not field_lines:
                    raise ValueError("No field lines detected")

                intersection_point = line.find_average_intersection_point(field_lines)
                if intersection_point is None or not all(coord is not None for coord in intersection_point):
                    raise ValueError("No valid field-line intersection")

                left_line = field.fit_line_to_field(
                    (intersection_point[0], intersection_point[1], ruck_left[0], ruck_left[1]),
                    field_outline,
                )
                right_line = field.fit_line_to_field(
                    (intersection_point[0], intersection_point[1], ruck_right[0], ruck_right[1]),
                    field_outline,
                )

                players = _build_player_pipeline(resized_frame, player_model, player_tracker)
                player_dict = player.build_player_coord_dict(players)
                player_dict = offside.filter_for_offside_detection(
                    player_dict,
                    ruck_box_converted,
                    overlap_threshold=0.4,
                    width_expansion_factor=0,
                    height_expansion_factor=0,
                )
                offside_boxes = offside.get_players_between_lines(player_dict, left_line, right_line)

                event_frame = _draw_geometry_overlay(
                    resized_frame,
                    field_lines,
                    intersection_point,
                    [
                        ("ruck left foot", ruck_left, (0, 255, 0)),
                        ("ruck right foot", ruck_right, (0, 180, 0)),
                    ],
                    [left_line, right_line],
                    "ruck",
                )
                event_frame = draw.draw_boxes(
                    event_frame,
                    offside_boxes,
                    box_annotation="Offside",
                    outline_colour=(0, 0, 255),
                    show_image=False,
                    font_scale=0.5,
                    font_thickness=1,
                )
                event_frame = _draw_event_header(event_frame, "ruck", confidence, len(offside_boxes))

                overlay_records.append({
                    "start": 1 if not overlay_records else frame_number,
                    "end": max(total_frames, frame_number),
                    "event_type": "ruck",
                    "confidence": confidence,
                    "field_lines": field_lines,
                    "intersection_point": intersection_point,
                    "anchor_points": [
                        ("ruck left foot", ruck_left, (0, 255, 0)),
                        ("ruck right foot", ruck_right, (0, 180, 0)),
                    ],
                    "offside_lines": [left_line, right_line],
                    "offside_boxes": offside_boxes,
                })

                event_data = _build_event_data(
                    "ruck", frame_number, fps, confidence, players, player_dict, offside_boxes,
                    {"ruck_confidence": confidence}
                )
                detections_data["events"].append(event_data)
                detections_data["logs"].append(
                    make_log_entry(
                        "offside_detection",
                        f"Ruck event recorded with {len(offside_boxes)} offside players",
                        event_type="ruck",
                        frame_number=frame_number,
                        confidence=confidence,
                        player_ids=[p.get("track_id") for p in players if p.get("track_id") is not None],
                    )
                )
                print(f"    Offside players detected: {len(offside_boxes)}")
            except Exception as exc:
                warning = f"Ruck processing failed at frame {frame_number}"
                print(f"    Warning: {warning}: {exc}")
                detections_data["logs"].append(
                    make_log_entry(
                        "ruck_processing", warning, level="WARNING",
                        event_type="ruck", frame_number=frame_number,
                        confidence=confidence, error=exc,
                    )
                )

        if lineout_frame_count >= frame_threshold:
            lineout_frame_count = -(fps * 10) + frame_threshold
            confidence = float(np.max(lineout_confidences))
            print(f"  Frame {frame_number}: Lineout detected (confidence: {confidence:.2%})")

            try:
                lineout_box = lineout_boxes[int(np.argmax(lineout_confidences))]
                imsize = (frame.shape[1], frame.shape[0])
                resized_frame = cv2.resize(frame, (800, 450))
                lineout_box_converted = general.convert_coordinates(tuple(lineout_box), imsize)
                lineout_centre_converted = general.box_bottom_centre(lineout_box_converted)

                field_lines, field_outline = field.get_field_lines(
                    resized_frame,
                    lineout_centre=lineout_centre_converted,
                    exclusion_box=lineout_box_converted,
                    thresh=ui_thresh,
                    is_path=False,
                    visualise_steps=False,
                )
                if not field_lines:
                    raise ValueError("No field lines detected")

                intersection_point = line.find_average_intersection_point(field_lines)
                if intersection_point is None or not all(coord is not None for coord in intersection_point):
                    raise ValueError("No valid field-line intersection")

                frame_height, frame_width = resized_frame.shape[:2]
                left_line, right_line, left_point, right_point = _lineout_perspective_lines(
                    intersection_point,
                    lineout_centre_converted,
                    lineout_box_converted,
                    field_outline,
                    (frame_width, frame_height),
                )

                players = _build_player_pipeline(resized_frame, player_model, player_tracker)
                player_dict = player.build_player_coord_dict(players)
                player_dict = offside.filter_for_offside_detection(player_dict, lineout_box_converted, 0)
                player_dict = offside.filter_detections_off_the_field(player_dict, lineout_box_converted, (800, 450))
                offside_boxes = offside.get_players_between_lines(player_dict, left_line, right_line)

                event_frame = _draw_geometry_overlay(
                    resized_frame,
                    field_lines,
                    intersection_point,
                    [
                        ("lineout centre", lineout_centre_converted, (0, 255, 0)),
                        ("left offside point", left_point, (0, 0, 255)),
                        ("right offside point", right_point, (0, 0, 255)),
                    ],
                    [left_line, right_line],
                    "lineout",
                )
                event_frame = draw.draw_boxes(
                    event_frame,
                    offside_boxes,
                    box_annotation="Offside",
                    outline_colour=(0, 0, 255),
                    show_image=False,
                )
                event_frame = _draw_event_header(event_frame, "lineout", confidence, len(offside_boxes))

                overlay_records.append({
                    "start": 1 if not overlay_records else frame_number,
                    "end": max(total_frames, frame_number),
                    "event_type": "lineout",
                    "confidence": confidence,
                    "field_lines": field_lines,
                    "intersection_point": intersection_point,
                    "anchor_points": [
                        ("lineout centre", lineout_centre_converted, (0, 255, 0)),
                        ("left offside point", left_point, (0, 0, 255)),
                        ("right offside point", right_point, (0, 0, 255)),
                    ],
                    "offside_lines": [left_line, right_line],
                    "offside_boxes": offside_boxes,
                })

                event_data = _build_event_data(
                    "lineout", frame_number, fps, confidence, players, player_dict, offside_boxes,
                    {"lineout_confidence": confidence}
                )
                detections_data["events"].append(event_data)
                detections_data["logs"].append(
                    make_log_entry(
                        "offside_detection",
                        f"Lineout event recorded with {len(offside_boxes)} offside players",
                        event_type="lineout",
                        frame_number=frame_number,
                        confidence=confidence,
                        player_ids=[p.get("track_id") for p in players if p.get("track_id") is not None],
                    )
                )
                print(f"    Offside players detected: {len(offside_boxes)}")
            except Exception as exc:
                warning = f"Lineout processing failed at frame {frame_number}"
                print(f"    Warning: {warning}: {exc}")
                detections_data["logs"].append(
                    make_log_entry(
                        "lineout_processing", warning, level="WARNING",
                        event_type="lineout", frame_number=frame_number,
                        confidence=confidence, error=exc,
                    )
                )

    detections_data["total_frames"] = frame_number
    detections_data["summary"] = build_summary(detections_data)
    annotated_frames = _build_full_length_annotated_frames(video_path, overlay_records) if overlay_records else []
    return detections_data, annotated_frames, fps


def _build_event_data(event_type, frame_number, fps, confidence, players, player_dict,
                      offside_boxes, extra_confidences):
    """Create a standard event dictionary for reports and frontend payloads."""
    offside_box_lists = [list(box) for box in offside_boxes]
    event = {
        "type": event_type,
        "frame_number": frame_number,
        "timestamp": f"{frame_number / fps:.2f}s",
        "detection_confidence": float(confidence),
        "processing_stage": "offside_detection",
        "offside_count": len(offside_boxes),
        "offside_players": [],
        "team_counts": player.count_teams_and_refs(players),
        "tracked_players": _serialise_players(players),
        "errors": [],
    }
    event.update(extra_confidences)

    for player_pos, player_box in player_dict.items():
        if list(player_box) in offside_box_lists:
            event["offside_players"].append({
                "x": int(player_pos[0]),
                "y": int(player_pos[1]),
                "confidence": 0.85,
            })

    return event


def process_video_batch(input_path, output_dir, ruck_model, lineout_model, ball_model, player_model):
    """
    Process one video or a directory of videos sequentially.

    Parameters:
        input_path (str): Source video file or directory.
        output_dir (str): Destination directory for reports and annotated clips.
        ruck_model (YOLO): Ruck detector.
        lineout_model (YOLO): Lineout detector.
        ball_model (YOLO): Ball detector.
        player_model (YOLO): Player detector.

    Returns:
        list: Structured per-video payloads, including failed clip records.

    Error handling:
        Invalid input paths raise ``ValueError``. Per-clip processing failures
        are logged in the returned list and do not stop the remaining batch.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path_obj = Path(input_path)
    if input_path_obj.is_file():
        video_files = [str(input_path_obj.absolute())]
    elif input_path_obj.is_dir():
        video_files = get_video_files(input_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if not video_files:
        print("No video files found to process.")
        return []

    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING: {len(video_files)} video(s) found")
    print(f"{'=' * 80}\n")

    batch_results = []

    for index, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        print(f"\n[{index}/{len(video_files)}] Processing: {Path(video_path).name}")
        print("-" * 80)

        try:
            detections_data, annotated_frames, fps = auto_mode_batch(
                video_path, output_dir, ruck_model, lineout_model, ball_model, player_model
            )

            report_path = output_path / f"{video_name}_analysis_report.txt"
            generate_summary_report(video_path, str(report_path), detections_data)

            output_files = {
                "report": str(report_path.absolute()),
                "json": None,
                "annotated_video": None,
            }

            if annotated_frames:
                video_output_path = output_path / f"{video_name}_annotated.mp4"
                frame_size = (annotated_frames[0][1].shape[1], annotated_frames[0][1].shape[0])
                save_annotated_video(annotated_frames, str(video_output_path), fps, frame_size)
                output_files["annotated_video"] = str(video_output_path.absolute())
                print(f"  Annotated video saved with {len(annotated_frames)} detection frames")
            else:
                print("  No detections found - no annotated video generated")

            json_path = output_path / f"{video_name}_analysis.json"
            payload = build_frontend_payload(video_path, detections_data, output_files)
            output_files["json"] = str(json_path.absolute())
            payload["output_files"] = output_files
            write_json_summary(json_path, payload)
            batch_results.append(payload)
            print(f"  Processing complete: {len(detections_data['events'])} events detected")
        except Exception as exc:
            print(f"  Error processing {Path(video_path).name}: {exc}")
            failed_payload = {
                "video_name": video_name,
                "source_video": str(Path(video_path).absolute()),
                "processing_date": datetime.now().isoformat(timespec="seconds"),
                "status": "failed",
                "total_frames": 0,
                "fps": 0,
                "events": [],
                "summary": build_summary({"events": []}),
                "logs": [
                    make_log_entry("batch", f"Error processing {Path(video_path).name}", level="ERROR", error=exc)
                ],
                "output_files": {},
            }
            batch_results.append(failed_payload)
            continue

    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    return batch_results
