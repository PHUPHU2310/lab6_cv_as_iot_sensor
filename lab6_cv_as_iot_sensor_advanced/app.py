"""
Lab 6 Advanced - Computer Vision as IoT Sensor: Parameter Experiment

One-file backend with ROI, adjustable threshold/edge, blur score,
quality-based events, advanced motion capture, and parameter experiment logging.

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
Open:
    http://127.0.0.1:8000/
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_images"
PROCESSED_DIR = DATA_DIR / "processed_images"
VIDEO_DIR = DATA_DIR / "videos"
OUTPUT_DIR = ROOT / "outputs"
METADATA_CSV = OUTPUT_DIR / "image_metadata.csv"
EVENT_CSV = OUTPUT_DIR / "image_event_log.csv"
PARAM_LOG_CSV = OUTPUT_DIR / "parameter_experiment_log.csv"
INDEX_HTML = ROOT / "index.html"

for _folder in [RAW_DIR, PROCESSED_DIR, VIDEO_DIR, OUTPUT_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CSV schemas
# ---------------------------------------------------------------------------
METADATA_FIELDS = [
    "image_id", "device_id", "timestamp", "source_type", "image_path", "processed_path",
    "width", "height", "brightness", "blur_score", "roi", "threshold_val",
    "canny_low", "canny_high", "processing_status", "processing_time_ms", "note",
]

EVENT_FIELDS = [
    "event_id", "image_id", "timestamp", "event_type", "score",
    "severity", "explanation", "action_hint", "rule_used",
]

PARAM_FIELDS = [
    "experiment_id", "timestamp", "image_id", "source_type",
    "roi_x1", "roi_y1", "roi_x2", "roi_y2",
    "threshold_val", "canny_low", "canny_high",
    "diff_threshold", "min_area", "cooldown",
    "brightness", "blur_score",
    "event_type", "severity", "rule_used", "note",
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def relative_url(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


def validate_image_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def frame_to_jpeg_bytes(frame_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG")
    return buffer.tobytes()

# ---------------------------------------------------------------------------
# Core image analysis functions
# ---------------------------------------------------------------------------

def compute_brightness(frame_bgr: np.ndarray, roi: Optional[Tuple[int,int,int,int]] = None) -> float:
    """Mean brightness of a frame (or ROI sub-region) in grayscale.

    A high value (>200) may indicate over-exposure.
    A low value (<60) indicates poor lighting.
    """
    region = crop_roi(frame_bgr, roi) if roi else frame_bgr
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_blur_score(frame_bgr: np.ndarray, roi: Optional[Tuple[int,int,int,int]] = None) -> float:
    """Variance of Laplacian as a sharpness / blur metric.

    Higher value → sharper image.
    Values below ~80 typically indicate a blurry image.
    Camera shake or out-of-focus lens reduces this score significantly.
    """
    region = crop_roi(frame_bgr, roi) if roi else frame_bgr
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_roi(frame_bgr: np.ndarray, roi: Tuple[int,int,int,int]) -> np.ndarray:
    """Crop frame to the ROI rectangle (x1, y1, x2, y2).

    Clamps coordinates to valid range so out-of-bounds values are safe.
    """
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(int(roi[0]), w - 1))
    y1 = max(0, min(int(roi[1]), h - 1))
    x2 = max(x1 + 1, min(int(roi[2]), w))
    y2 = max(y1 + 1, min(int(roi[3]), h))
    return frame_bgr[y1:y2, x1:x2]


def parse_roi(roi_str: str, frame_bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """Parse 'x1,y1,x2,y2' string. Returns None for empty / 'full'."""
    roi_str = roi_str.strip()
    if not roi_str or roi_str.lower() in ("full", "0", "none", ""):
        return None
    try:
        parts = [int(v.strip()) for v in roi_str.split(",")]
        if len(parts) != 4:
            return None
        return (parts[0], parts[1], parts[2], parts[3])
    except ValueError:
        return None


def event_from_quality(brightness: float, blur_score: float) -> Dict[str, str]:
    """Derive a quality event from brightness and blur metrics.

    Rules (applied top-down, first match wins):
    1. brightness < 60  → LOW_LIGHT / WARNING
    2. brightness > 210 → OVER_EXPOSED_IMAGE / WARNING
    3. blur_score < 80  → BLURRY_IMAGE / WARNING
    4. else             → IMAGE_QUALITY_OK / NORMAL
    """
    if brightness < 60:
        return {
            "event_type": "LOW_LIGHT",
            "severity": "WARNING",
            "explanation": f"Brightness={brightness:.1f} < 60. Image is too dark for reliable AI inference.",
            "action_hint": "Improve ambient lighting or increase camera exposure.",
            "rule_used": "brightness < 60",
        }
    if brightness > 210:
        return {
            "event_type": "OVER_EXPOSED_IMAGE",
            "severity": "WARNING",
            "explanation": f"Brightness={brightness:.1f} > 210. Image is over-exposed; details may be washed out.",
            "action_hint": "Reduce camera exposure or add shading to the scene.",
            "rule_used": "brightness > 210",
        }
    if blur_score < 80:
        return {
            "event_type": "BLURRY_IMAGE",
            "severity": "WARNING",
            "explanation": f"Blur score (Laplacian variance)={blur_score:.1f} < 80. Image is not sharp enough.",
            "action_hint": "Stabilise the camera, clean the lens, or increase focus distance.",
            "rule_used": "blur_score < 80",
        }
    return {
        "event_type": "IMAGE_QUALITY_OK",
        "severity": "NORMAL",
        "explanation": f"Brightness={brightness:.1f}, blur_score={blur_score:.1f}. Image quality is acceptable.",
        "action_hint": "Image is ready to be passed to Lab 7 object detection.",
        "rule_used": "brightness 60-210 AND blur_score >= 80",
    }


# ---------------------------------------------------------------------------
# Advanced contact sheet: 6 processing steps
# ---------------------------------------------------------------------------

def _label_tile(tile: np.ndarray, text: str) -> np.ndarray:
    canvas = tile.copy()
    cv2.rectangle(canvas, (0, 0), (tile.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(canvas, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return canvas


def create_processed_contact_sheet_advanced(
    frame_bgr: np.ndarray,
    image_id: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    threshold_val: int = 120,
    canny_low: int = 80,
    canny_high: int = 160,
) -> Tuple[Path, float, Dict[str, Any]]:
    """Create a 3×2 contact sheet showing 6 processing steps.

    Panels (left→right, top→bottom):
    1. Original + ROI overlay
    2. Grayscale (ROI if set, else full)
    3. Threshold binary (adjustable threshold_val)
    4. Canny edge (adjustable canny_low/canny_high)
    5. Morphological mask (dilated threshold)
    6. Quality info overlay
    """
    start = time.perf_counter()
    W, H = 320, 240

    # --- Panel 1: Original + ROI rectangle ---
    panel1 = cv2.resize(frame_bgr, (W, H))
    if roi:
        scale_x = W / frame_bgr.shape[1]
        scale_y = H / frame_bgr.shape[0]
        rx1 = int(roi[0] * scale_x)
        ry1 = int(roi[1] * scale_y)
        rx2 = int(roi[2] * scale_x)
        ry2 = int(roi[3] * scale_y)
        cv2.rectangle(panel1, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

    # --- Working region (ROI or full) ---
    work = crop_roi(frame_bgr, roi) if roi else frame_bgr
    work_resized = cv2.resize(work, (W, H))

    # --- Panel 2: Grayscale ---
    gray = cv2.cvtColor(work_resized, cv2.COLOR_BGR2GRAY)
    panel2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- Panel 3: Threshold ---
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    panel3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # --- Panel 4: Canny edge ---
    edges = cv2.Canny(gray, canny_low, canny_high)
    panel4 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # --- Panel 5: Morphological mask (dilated threshold) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(thresh, kernel, iterations=2)
    mask_overlay = work_resized.copy()
    mask_overlay[mask == 0] = (mask_overlay[mask == 0] * 0.3).astype(np.uint8)
    panel5 = mask_overlay

    # --- Panel 6: Quality info ---
    brightness = compute_brightness(frame_bgr, roi)
    blur_score = compute_blur_score(frame_bgr, roi)
    panel6 = np.full((H, W, 3), 30, dtype=np.uint8)
    quality_event = event_from_quality(brightness, blur_score)
    evt_color = (40, 200, 80) if quality_event["severity"] == "NORMAL" else (40, 100, 220)
    lines = [
        "QUALITY ANALYSIS",
        f"Brightness: {brightness:.1f}",
        f"Blur score: {blur_score:.1f}",
        f"ROI: {roi if roi else 'full frame'}",
        f"Thresh: {threshold_val}",
        f"Canny: {canny_low}/{canny_high}",
        "",
        quality_event["event_type"],
        quality_event["severity"],
    ]
    for i, line in enumerate(lines):
        color = evt_color if i >= 7 else (220, 220, 220)
        scale = 0.7 if i >= 7 else 0.52
        cv2.putText(panel6, line, (12, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if i < 7 else 2)

    # --- Assemble 3×2 grid ---
    row1 = np.hstack([
        _label_tile(panel1, "1. ORIGINAL + ROI"),
        _label_tile(panel2, "2. GRAYSCALE"),
        _label_tile(panel3, f"3. THRESHOLD ({threshold_val})"),
    ])
    row2 = np.hstack([
        _label_tile(panel4, f"4. CANNY EDGE ({canny_low}/{canny_high})"),
        _label_tile(panel5, "5. MASK COMBINED"),
        _label_tile(panel6, "6. QUALITY INFO"),
    ])
    sheet = np.vstack([row1, row2])

    out_path = PROCESSED_DIR / f"{image_id}_processed_advanced.jpg"
    cv2.imwrite(str(out_path), sheet)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    stats: Dict[str, Any] = {
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "width": int(frame_bgr.shape[1]),
        "height": int(frame_bgr.shape[0]),
    }
    return out_path, elapsed_ms, stats


# ---------------------------------------------------------------------------
# Advanced pipeline: log image + metadata + event + param experiment
# ---------------------------------------------------------------------------

def log_image_pipeline_advanced(
    frame_bgr: np.ndarray,
    source_type: str,
    device_id: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    threshold_val: int = 120,
    canny_low: int = 80,
    canny_high: int = 160,
    diff_threshold: int = 25,
    min_area: int = 800,
    cooldown: float = 1.0,
    note: str = "",
) -> Dict[str, Any]:
    """Full advanced pipeline: save raw, create contact sheet, write metadata+event+param log."""
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    timestamp = now_iso()
    raw_path = RAW_DIR / f"{image_id}.jpg"
    cv2.imwrite(str(raw_path), frame_bgr)

    roi_str = f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}" if roi else "full"

    processed_path, proc_time_ms, stats = create_processed_contact_sheet_advanced(
        frame_bgr, image_id,
        roi=roi,
        threshold_val=threshold_val,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    brightness = stats["brightness"]
    blur_score = stats["blur_score"]

    # --- Metadata ---
    metadata_row = {
        "image_id": image_id,
        "device_id": device_id,
        "timestamp": timestamp,
        "source_type": source_type,
        "image_path": str(raw_path.relative_to(ROOT)),
        "processed_path": str(processed_path.relative_to(ROOT)),
        "width": stats["width"],
        "height": stats["height"],
        "brightness": brightness,
        "blur_score": blur_score,
        "roi": roi_str,
        "threshold_val": threshold_val,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "processing_status": "processed",
        "processing_time_ms": proc_time_ms,
        "note": note,
    }
    append_csv(METADATA_CSV, METADATA_FIELDS, metadata_row)

    # --- Quality event ---
    q = event_from_quality(brightness, blur_score)
    event_row = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "image_id": image_id,
        "timestamp": timestamp,
        "event_type": q["event_type"],
        "score": round(brightness, 2),
        "severity": q["severity"],
        "explanation": q["explanation"],
        "action_hint": q["action_hint"],
        "rule_used": q["rule_used"],
    }
    append_csv(EVENT_CSV, EVENT_FIELDS, event_row)

    # --- Parameter experiment log ---
    x1, y1, x2, y2 = roi if roi else ("", "", "", "")
    param_row = {
        "experiment_id": f"exp_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "image_id": image_id,
        "source_type": source_type,
        "roi_x1": x1, "roi_y1": y1, "roi_x2": x2, "roi_y2": y2,
        "threshold_val": threshold_val,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "cooldown": cooldown,
        "brightness": brightness,
        "blur_score": blur_score,
        "event_type": q["event_type"],
        "severity": q["severity"],
        "rule_used": q["rule_used"],
        "note": note,
    }
    append_csv(PARAM_LOG_CSV, PARAM_FIELDS, param_row)

    return {
        "image_id": image_id,
        "metadata": metadata_row,
        "event": event_row,
        "param_experiment": param_row,
        "raw_image_url": relative_url(raw_path),
        "processed_image_url": relative_url(processed_path),
    }


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def parse_camera_source(source: str) -> Any:
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def simulated_frame(counter: int = 0, width: int = 640, height: int = 360) -> np.ndarray:
    """Fallback stream when no laptop/IP camera is available."""
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 30 + (counter * 12) % max(1, width - 180)
    y = 80 + (counter * 7) % max(1, height - 170)
    # Moving rectangle and circle to simulate motion
    cv2.rectangle(frame, (x, 120), (x + 130, 240), (40, 140, 240), -1)
    cv2.circle(frame, (width - 110, y), 38, (80, 200, 120), -1)
    cv2.putText(frame, "SIMULATED CAMERA STREAM", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, "Use source=0 for laptop camera or IP camera URL", (25, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 80), 1)
    return frame


def open_capture(source: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(parse_camera_source(source))
    if not cap.isOpened():
        return None
    # Verify we can actually read a frame (MSMF may open but fail to grab)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def read_one_frame(source: str = "0") -> Tuple[np.ndarray, str]:
    cap = open_capture(source)
    if cap is None:
        return simulated_frame(0), "simulated"
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return simulated_frame(0), "simulated"
    return frame, "camera"


def stream_frames(source: str = "0", roi_str: str = "") -> Iterable[bytes]:
    cap = open_capture(source)
    counter = 0
    consecutive_failures = 0
    MAX_FAILURES = 5
    while True:
        if cap is None:
            frame = simulated_frame(counter)
            label = "SIMULATED"
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    cap.release()
                    cap = None
                frame = simulated_frame(counter)
                label = "SIMULATED_FALLBACK"
            else:
                consecutive_failures = 0
                label = "LIVE_CAMERA"

        # Draw ROI overlay on stream
        if roi_str and roi_str.lower() not in ("full", "none", ""):
            roi = parse_roi(roi_str, frame)
            if roi:
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.putText(frame, "ROI", (roi[0] + 4, roi[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (255, 255, 255), -1)
        cv2.putText(frame, f"{label} | source={source} | frame={counter}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        jpg = frame_to_jpeg_bytes(frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        counter += 1
        time.sleep(0.08)


# ---------------------------------------------------------------------------
# Advanced motion capture
# ---------------------------------------------------------------------------

# Simple in-memory cooldown state (per source key)
_last_motion_event_time: Dict[str, float] = {}


def motion_capture_advanced(
    source: str,
    seconds: int = 8,
    diff_threshold: int = 25,
    min_area: int = 800,
    cooldown: float = 1.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
    threshold_val: int = 120,
    canny_low: int = 80,
    canny_high: int = 160,
) -> Dict[str, Any]:
    """Collect frames for `seconds`, compute frame-difference motion score,
    optionally respect cooldown, and log the best frame through the pipeline."""
    seconds = max(1, min(int(seconds), 30))
    cap = open_capture(source)
    prev_gray = None
    best_frame = None
    best_score = 0.0
    frames_seen = 0
    motion_events_skipped = 0
    start = time.perf_counter()

    while time.perf_counter() - start < seconds:
        if cap is None:
            frame = simulated_frame(frames_seen)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = simulated_frame(frames_seen)
        frames_seen += 1

        # Use ROI for motion analysis if specified
        work = crop_roi(frame, roi) if roi else frame
        gray = cv2.cvtColor(cv2.resize(work, (320, 240)), cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            score = float(sum(cv2.contourArea(c) for c in contours))
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

        prev_gray = gray
        time.sleep(0.08)

    if cap is not None:
        cap.release()
    if best_frame is None:
        best_frame = simulated_frame(frames_seen)

    # --- Cooldown check ---
    now_ts = time.time()
    last_evt = _last_motion_event_time.get(source, 0.0)
    cooldown_active = (now_ts - last_evt) < cooldown and best_score >= float(min_area)
    if cooldown_active:
        motion_events_skipped += 1

    # --- Log pipeline ---
    result = log_image_pipeline_advanced(
        best_frame,
        source_type="motion_capture_advanced",
        device_id=f"camera:{source}",
        roi=roi,
        threshold_val=threshold_val,
        canny_low=canny_low,
        canny_high=canny_high,
        diff_threshold=diff_threshold,
        min_area=min_area,
        cooldown=cooldown,
        note=f"motion_score={round(best_score,2)}, diff_threshold={diff_threshold}, min_area={min_area}, cooldown={cooldown}",
    )

    # --- Motion-specific event ---
    motion_detected = best_score >= float(min_area) and not cooldown_active
    if motion_detected:
        _last_motion_event_time[source] = now_ts

    motion_event = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "image_id": result["image_id"],
        "timestamp": now_iso(),
        "event_type": "MOTION_DETECTED" if motion_detected else (
            "COOLDOWN_SKIP" if cooldown_active else "NO_SIGNIFICANT_MOTION"
        ),
        "score": round(best_score, 2),
        "severity": "WARNING" if motion_detected else "NORMAL",
        "explanation": (
            f"Motion score {best_score:.0f} >= min_area {min_area}." if motion_detected else (
                f"Cooldown active ({cooldown}s): event suppressed." if cooldown_active
                else f"Motion score {best_score:.0f} < min_area {min_area}."
            )
        ),
        "action_hint": (
            "Review captured image." if motion_detected else
            "Reduce diff_threshold or min_area if motion is being missed." if not cooldown_active
            else "Reduce cooldown if events are being lost."
        ),
        "rule_used": f"diff_threshold={diff_threshold}, min_area={min_area}, cooldown={cooldown}s",
    }
    append_csv(EVENT_CSV, EVENT_FIELDS, motion_event)

    result["motion_event"] = motion_event
    result["motion_detected"] = motion_detected
    result["motion_score"] = round(best_score, 2)
    result["frames_seen"] = frames_seen
    result["cooldown_active"] = cooldown_active
    result["motion_events_skipped"] = motion_events_skipped
    return result


# ---------------------------------------------------------------------------
# Video recording (reused from basic, unchanged)
# ---------------------------------------------------------------------------

def record_short_video(source: str, seconds: int = 5) -> Dict[str, Any]:
    seconds = max(1, min(int(seconds), 30))
    cap = open_capture(source)
    fps, width, height = 10, 640, 360
    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    out_path = VIDEO_DIR / f"{video_id}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        frame = simulated_frame(frame_count, width, height) if cap is None else _safe_read(cap, frame_count, width, height)
        frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        frame_count += 1
        time.sleep(1.0 / fps)
    if cap is not None:
        cap.release()
    writer.release()

    event_row = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "image_id": video_id,
        "timestamp": now_iso(),
        "event_type": "VIDEO_RECORDED",
        "score": frame_count,
        "severity": "NORMAL",
        "explanation": f"Recorded a short video clip with {frame_count} frames.",
        "action_hint": "Use the video clip for review or future frame analysis.",
        "rule_used": f"seconds={seconds}",
    }
    append_csv(EVENT_CSV, EVENT_FIELDS, event_row)
    return {
        "video_id": video_id,
        "video_url": relative_url(out_path),
        "video_path": str(out_path.relative_to(ROOT)),
        "seconds": seconds,
        "frames": frame_count,
        "event": event_row,
    }


def _safe_read(cap: cv2.VideoCapture, counter: int, w: int, h: int) -> np.ndarray:
    ok, frame = cap.read()
    return frame if ok and frame is not None else simulated_frame(counter, w, h)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Lab 6 Advanced - CV as IoT Sensor: Parameter Experiment",
    description="ROI, adjustable threshold/edge, blur score, quality events, motion analysis, parameter experiment log.",
)
app.mount("/files", StaticFiles(directory=str(ROOT)), name="files")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "lab": "Lab 6 Advanced - Computer Vision as IoT Sensor",
        "outputs": {
            "metadata_csv": str(METADATA_CSV.relative_to(ROOT)),
            "event_csv": str(EVENT_CSV.relative_to(ROOT)),
            "param_log_csv": str(PARAM_LOG_CSV.relative_to(ROOT)),
        },
    }


@app.get("/video_feed")
def video_feed(
    source: str = Query("0"),
    roi: str = Query("", description="ROI as x1,y1,x2,y2 or empty for full frame"),
) -> StreamingResponse:
    return StreamingResponse(stream_frames(source, roi), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot-advanced")
def snapshot_advanced(
    source: str = Query("0"),
    roi: str = Query("", description="ROI as x1,y1,x2,y2"),
    threshold_val: int = Query(120, ge=0, le=255),
    canny_low: int = Query(80, ge=0, le=500),
    canny_high: int = Query(160, ge=0, le=500),
) -> Dict[str, Any]:
    frame, source_type = read_one_frame(source)
    parsed_roi = parse_roi(roi, frame)
    return log_image_pipeline_advanced(
        frame,
        source_type=source_type,
        device_id=f"camera:{source}",
        roi=parsed_roi,
        threshold_val=threshold_val,
        canny_low=canny_low,
        canny_high=canny_high,
        note="snapshot_advanced",
    )


@app.post("/upload-image-advanced")
async def upload_image_advanced(
    file: UploadFile = File(...),
    device_id: str = "upload_client",
    roi: str = Query(""),
    threshold_val: int = Query(120, ge=0, le=255),
    canny_low: int = Query(80, ge=0, le=500),
    canny_high: int = Query(160, ge=0, le=500),
) -> Dict[str, Any]:
    data = await file.read()
    img = validate_image_bytes(data)
    frame = pil_to_bgr(img)
    parsed_roi = parse_roi(roi, frame)
    return log_image_pipeline_advanced(
        frame,
        source_type="upload",
        device_id=device_id,
        roi=parsed_roi,
        threshold_val=threshold_val,
        canny_low=canny_low,
        canny_high=canny_high,
        note=f"filename={file.filename}",
    )


@app.get("/motion-capture-advanced")
def motion_capture_advanced_endpoint(
    source: str = Query("0"),
    seconds: int = Query(8, ge=1, le=30),
    diff_threshold: int = Query(25, ge=1, le=255),
    min_area: int = Query(800, ge=10, le=50000),
    cooldown: float = Query(1.0, ge=0.0, le=60.0),
    roi: str = Query(""),
    threshold_val: int = Query(120, ge=0, le=255),
    canny_low: int = Query(80, ge=0, le=500),
    canny_high: int = Query(160, ge=0, le=500),
) -> Dict[str, Any]:
    frame_dummy, _ = read_one_frame(source)
    parsed_roi = parse_roi(roi, frame_dummy)
    return motion_capture_advanced(
        source=source,
        seconds=seconds,
        diff_threshold=diff_threshold,
        min_area=min_area,
        cooldown=cooldown,
        roi=parsed_roi,
        threshold_val=threshold_val,
        canny_low=canny_low,
        canny_high=canny_high,
    )


@app.get("/record-video")
def record_video(
    source: str = Query("0"),
    seconds: int = Query(5, ge=1, le=30),
) -> Dict[str, Any]:
    return record_short_video(source, seconds=seconds)


@app.get("/metadata")
def metadata(limit: int = 20) -> Dict[str, Any]:
    rows = read_csv(METADATA_CSV)
    return {"count": len(rows), "items": rows[-limit:]}


@app.get("/events")
def events(limit: int = 20) -> Dict[str, Any]:
    rows = read_csv(EVENT_CSV)
    return {"count": len(rows), "items": rows[-limit:]}


@app.get("/parameter-experiments")
def parameter_experiments(limit: int = 30) -> Dict[str, Any]:
    rows = read_csv(PARAM_LOG_CSV)
    return {"count": len(rows), "items": rows[-limit:]}


@app.get("/latest")
def latest() -> Dict[str, Any]:
    meta_rows = read_csv(METADATA_CSV)
    event_rows = read_csv(EVENT_CSV)
    param_rows = read_csv(PARAM_LOG_CSV)
    latest_meta = meta_rows[-1] if meta_rows else None
    raw_url = processed_url = None
    if latest_meta:
        raw_url = relative_url(ROOT / latest_meta.get("image_path", ""))
        processed_url = relative_url(ROOT / latest_meta.get("processed_path", ""))
    return {
        "latest_metadata": latest_meta,
        "latest_event": event_rows[-1] if event_rows else None,
        "latest_param_experiment": param_rows[-1] if param_rows else None,
        "raw_image_url": raw_url,
        "processed_image_url": processed_url,
        "metadata_count": len(meta_rows),
        "event_count": len(event_rows),
        "param_experiment_count": len(param_rows),
    }


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    frame = simulated_frame(1)
    result = log_image_pipeline_advanced(frame, source_type="script", device_id="smoke_test",
                                          note="python app.py smoke test")
    print(json.dumps(result, indent=2, ensure_ascii=False))
