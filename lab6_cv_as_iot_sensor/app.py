
"""
Lab 6 - Computer Vision as IoT Sensor

One-file backend for the student lab:
- camera stream from laptop camera or IP camera URL
- snapshot capture
- short video recording
- motion capture
- image upload
- image preprocessing contact sheet
- metadata and event logging
- browser dashboard served from index.html

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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_images"
PROCESSED_DIR = DATA_DIR / "processed_images"
VIDEO_DIR = DATA_DIR / "videos"
OUTPUT_DIR = ROOT / "outputs"
METADATA_CSV = OUTPUT_DIR / "image_metadata.csv"
EVENT_CSV = OUTPUT_DIR / "image_event_log.csv"
INDEX_HTML = ROOT / "index.html"

for folder in [RAW_DIR, PROCESSED_DIR, VIDEO_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

METADATA_FIELDS = [
    "image_id", "device_id", "timestamp", "source_type", "image_path", "processed_path",
    "width", "height", "brightness", "processing_status", "processing_time_ms", "note"
]

EVENT_FIELDS = [
    "event_id", "image_id", "timestamp", "event_type", "score", "severity", "explanation", "action_hint"
]


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


def compute_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def create_processed_contact_sheet(frame_bgr: np.ndarray, image_id: str) -> Tuple[Path, float, Dict[str, Any]]:
    """Create one observable image containing four processing steps."""
    start = time.perf_counter()
    resized = cv2.resize(frame_bgr, (320, 240))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 80, 160)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    threshold_bgr = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def label(tile: np.ndarray, text: str) -> np.ndarray:
        canvas = tile.copy()
        cv2.rectangle(canvas, (0, 0), (320, 30), (255, 255, 255), -1)
        cv2.putText(canvas, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
        return canvas

    top = np.hstack([label(resized, "1. RESIZE"), label(gray_bgr, "2. GRAYSCALE")])
    bottom = np.hstack([label(threshold_bgr, "3. THRESHOLD"), label(edge_bgr, "4. EDGE")])
    sheet = np.vstack([top, bottom])

    out_path = PROCESSED_DIR / f"{image_id}_processed_steps.jpg"
    cv2.imwrite(str(out_path), sheet)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    stats = {"brightness": round(compute_brightness(frame_bgr), 2), "width": int(frame_bgr.shape[1]), "height": int(frame_bgr.shape[0])}
    return out_path, elapsed_ms, stats


def log_image_pipeline(frame_bgr: np.ndarray, source_type: str, device_id: str, note: str = "") -> Dict[str, Any]:
    """Save raw image, preprocess image, write metadata, and create a visual event."""
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    timestamp = now_iso()
    raw_path = RAW_DIR / f"{image_id}.jpg"
    cv2.imwrite(str(raw_path), frame_bgr)

    processed_path, processing_time_ms, stats = create_processed_contact_sheet(frame_bgr, image_id)
    brightness = stats["brightness"]

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
        "processing_status": "processed",
        "processing_time_ms": processing_time_ms,
        "note": note,
    }
    append_csv(METADATA_CSV, METADATA_FIELDS, metadata_row)

    if brightness < 70:
        event_type = "LOW_LIGHT"
        severity = "WARNING"
        explanation = "Image brightness is low; later AI inference may be less reliable."
        action_hint = "Improve lighting or review image quality before using the image for object detection."
    else:
        event_type = "IMAGE_PROCESSED"
        severity = "NORMAL"
        explanation = "Image was received, saved, preprocessed, and registered as visual data."
        action_hint = "Continue monitoring or pass the image to Lab 7 object detection."

    event_row = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "image_id": image_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "score": brightness,
        "severity": severity,
        "explanation": explanation,
        "action_hint": action_hint,
    }
    append_csv(EVENT_CSV, EVENT_FIELDS, event_row)

    return {
        "image_id": image_id,
        "metadata": metadata_row,
        "event": event_row,
        "raw_image_url": relative_url(raw_path),
        "processed_image_url": relative_url(processed_path),
    }


def parse_camera_source(source: str) -> Any:
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def simulated_frame(counter: int = 0, width: int = 640, height: int = 360) -> np.ndarray:
    """Fallback stream when no laptop/IP camera is available."""
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 30 + (counter * 12) % max(1, width - 180)
    y = 80 + (counter * 7) % max(1, height - 170)
    cv2.rectangle(frame, (x, 120), (x + 130, 240), (40, 140, 240), -1)
    cv2.circle(frame, (width - 110, y), 38, (80, 200, 120), -1)
    cv2.putText(frame, "SIMULATED CAMERA STREAM", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, "Use source=0 for laptop camera or an IP camera URL", (25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
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


def stream_frames(source: str = "0") -> Iterable[bytes]:
    cap = open_capture(source)
    counter = 0
    consecutive_failures = 0
    MAX_FAILURES = 5
    while True:
        if cap is None:
            frame = simulated_frame(counter)
            source_label = "SIMULATED"
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    cap.release()
                    cap = None
                frame = simulated_frame(counter)
                source_label = "SIMULATED_AFTER_CAMERA_ERROR"
            else:
                consecutive_failures = 0
                source_label = "LIVE_CAMERA"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (255, 255, 255), -1)
        cv2.putText(frame, f"{source_label} | source={source} | frame={counter}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
        jpg = frame_to_jpeg_bytes(frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        counter += 1
        time.sleep(0.08)


def record_short_video(source: str, seconds: int = 5) -> Dict[str, Any]:
    seconds = max(1, min(int(seconds), 30))
    cap = open_capture(source)
    fps = 10
    width, height = 640, 360
    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    out_path = VIDEO_DIR / f"{video_id}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_count = 0
    start = time.perf_counter()

    while time.perf_counter() - start < seconds:
        if cap is None:
            frame = simulated_frame(frame_count, width, height)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = simulated_frame(frame_count, width, height)
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
        "action_hint": "Use the video clip for later review or image analysis.",
    }
    append_csv(EVENT_CSV, EVENT_FIELDS, event_row)
    return {"video_id": video_id, "video_path": str(out_path.relative_to(ROOT)), "video_url": relative_url(out_path), "seconds": seconds, "frames": frame_count, "event": event_row}


def motion_capture(source: str, seconds: int = 8, threshold: int = 25, min_area: int = 800) -> Dict[str, Any]:
    """Capture the most changed frame and create a motion event."""
    seconds = max(1, min(int(seconds), 30))
    cap = open_capture(source)
    prev_gray = None
    best_frame = None
    best_score = 0.0
    frames_seen = 0
    start = time.perf_counter()

    while time.perf_counter() - start < seconds:
        if cap is None:
            frame = simulated_frame(frames_seen)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = simulated_frame(frames_seen)
        frames_seen += 1
        gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
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

    result = log_image_pipeline(best_frame, source_type="motion_capture", device_id=f"camera:{source}", note=f"motion_score={round(best_score, 2)}, threshold={threshold}, min_area={min_area}")
    motion_detected = best_score >= float(min_area)
    motion_event = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "image_id": result["image_id"],
        "timestamp": now_iso(),
        "event_type": "MOTION_DETECTED" if motion_detected else "NO_SIGNIFICANT_MOTION",
        "score": round(best_score, 2),
        "severity": "WARNING" if motion_detected else "NORMAL",
        "explanation": "Frame difference exceeded threshold." if motion_detected else "No significant frame difference was detected.",
        "action_hint": "Review captured image and keep the event in the dashboard." if motion_detected else "Continue visual monitoring.",
    }
    append_csv(EVENT_CSV, EVENT_FIELDS, motion_event)
    result["motion_event"] = motion_event
    result["motion_detected"] = motion_detected
    result["frames_seen"] = frames_seen
    return result


app = FastAPI(title="Lab 6 - Computer Vision as IoT Sensor", description="Camera stream, snapshot, video, motion, metadata, image event and dashboard.")
app.mount("/files", StaticFiles(directory=str(ROOT)), name="files")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/camera-demo")
def camera_demo() -> RedirectResponse:
    return RedirectResponse("/")


@app.get("/image-demo")
def image_demo() -> RedirectResponse:
    return RedirectResponse("/")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "lab": "Lab 6 - Computer Vision as IoT Sensor", "outputs": {"metadata_csv": str(METADATA_CSV.relative_to(ROOT)), "event_csv": str(EVENT_CSV.relative_to(ROOT))}}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), device_id: str = "upload_client") -> Dict[str, Any]:
    data = await file.read()
    img = validate_image_bytes(data)
    return log_image_pipeline(pil_to_bgr(img), source_type="upload", device_id=device_id, note=f"filename={file.filename}")


@app.get("/snapshot")
def snapshot(source: str = Query("0", description="0 for laptop camera, or IP camera/video URL")) -> Dict[str, Any]:
    frame, source_type = read_one_frame(source)
    return log_image_pipeline(frame, source_type=source_type, device_id=f"camera:{source}", note="snapshot button")


@app.get("/record-video")
def record_video(source: str = Query("0"), seconds: int = Query(5, ge=1, le=30)) -> Dict[str, Any]:
    return record_short_video(source, seconds=seconds)


@app.get("/motion-capture")
def motion_capture_endpoint(source: str = Query("0"), seconds: int = Query(8, ge=1, le=30), threshold: int = Query(25, ge=1, le=255), min_area: int = Query(800, ge=10, le=50000)) -> Dict[str, Any]:
    return motion_capture(source, seconds=seconds, threshold=threshold, min_area=min_area)


@app.get("/video_feed")
def video_feed(source: str = Query("0")) -> StreamingResponse:
    return StreamingResponse(stream_frames(source), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/metadata")
def metadata(limit: int = 20) -> Dict[str, Any]:
    rows = read_csv(METADATA_CSV)
    return {"count": len(rows), "items": rows[-limit:]}


@app.get("/events")
def events(limit: int = 20) -> Dict[str, Any]:
    rows = read_csv(EVENT_CSV)
    return {"count": len(rows), "items": rows[-limit:]}


@app.get("/latest")
def latest() -> Dict[str, Any]:
    meta_rows = read_csv(METADATA_CSV)
    event_rows = read_csv(EVENT_CSV)
    latest_meta = meta_rows[-1] if meta_rows else None
    raw_url = processed_url = None
    if latest_meta:
        raw_url = relative_url(ROOT / latest_meta.get("image_path", ""))
        processed_url = relative_url(ROOT / latest_meta.get("processed_path", ""))
    return {"latest_metadata": latest_meta, "latest_event": event_rows[-1] if event_rows else None, "raw_image_url": raw_url, "processed_image_url": processed_url, "metadata_count": len(meta_rows), "event_count": len(event_rows)}


if __name__ == "__main__":
    frame = simulated_frame(1)
    result = log_image_pipeline(frame, source_type="script", device_id="local_smoke", note="python app.py smoke test")
    print(json.dumps(result, indent=2, ensure_ascii=False))
