import cv2
import time
import os
import tempfile
import subprocess
import shutil
import threading
import base64

from ultralytics import YOLO
from datetime import datetime

from dashboard_state import latest_data

import sys

SRC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from pipeline.sign_pipeline import detect_and_classify

MODEL_PATH = r"runs/detect/runs/detect/indian_yolo11s_production_v1/weights/best.pt"

VIOLATION_SIGNS = {
    "no entry":      "HIGH",
    "stop":          "HIGH",
    "do not enter":  "HIGH",
    "wrong way":     "HIGH",
    "school zone":   "MEDIUM",
    "no overtaking": "MEDIUM",
    "no parking":    "MEDIUM",
    "speed limit":   "LOW",
}

model = YOLO(MODEL_PATH)

_camera = None

# ── Thread-safety ──────────────────────────────────────────────────────────────
# One lock guards all writes to the video-progress keys in latest_data.
# Reads in the Flask polling route are lock-free (GIL gives us atomic reads
# of individual Python objects, and we only need eventual consistency there).
_video_lock = threading.Lock()

# Each job gets a unique ID.  The running thread checks this every frame;
# if the ID has changed (new upload arrived) it exits immediately.
_current_job_id = None


def _new_job_id():
    import uuid
    return uuid.uuid4().hex


def _set_video_progress(job_id, frames_done, frames_total, done=False, output_file=""):
    """Write video progress atomically — but only if this job is still current."""
    with _video_lock:
        if _current_job_id != job_id:
            return False          # stale job, discard the write
        latest_data["video_frames_done"]  = frames_done
        latest_data["video_frames_total"] = frames_total
        latest_data["video_done"]         = done
        latest_data["video_output_file"]  = output_file
        return True


def _is_current_job(job_id):
    with _video_lock:
        return _current_job_id == job_id


# ─────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────

def _get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
        latest_data["camera_active"] = _camera.isOpened()
    return _camera


def release_camera():
    global _camera
    if _camera and _camera.isOpened():
        _camera.release()
        _camera = None
    latest_data["camera_active"] = False


# ─────────────────────────────────────────────
#  DETECTION HELPERS
# ─────────────────────────────────────────────

def _update_avg_confidence(conf_value):
    if conf_value > 0:
        latest_data["confidence_sum"]   += conf_value
        latest_data["confidence_count"] += 1
        latest_data["avg_confidence"]    = round(
            latest_data["confidence_sum"] / latest_data["confidence_count"], 2
        )


def _maybe_add_violation(sign_name):
    if not latest_data.get("alert_engine_active"):
        return   # engine is in standby — detection still runs, alerts don't

    key = sign_name.lower()
    for pattern, severity in VIOLATION_SIGNS.items():
        if pattern in key:
            timestamp = datetime.now().strftime("%H:%M:%S")
            if (
                len(latest_data["violations"]) == 0
                or latest_data["violations"][0]["sign"] != sign_name
            ):
                latest_data["violations"].insert(0, {
                    "sign":     sign_name,
                    "time":     timestamp,
                    "severity": severity
                })
                latest_data["alerts"] += 1
            latest_data["violations"] = latest_data["violations"][:50]
            break
def _process_detections(frame, results):
    """
    Single source of truth for turning a YOLO pass into everything the
    dashboard needs: history, violations, sign counts, AND the Sign
    Inspector snapshot (crop + OCR + both confidence scores).

    YOLO already ran once in the caller (generate_frames / detect_image /
    _run_video_pipeline) — this just runs EfficientNet + OCR on the boxes
    it found, via sign_pipeline.detect_and_classify.
    """
    detections = detect_and_classify(frame, results)

    detected_signs = []
    highest_conf   = 0.0
    best_detection = None

    for det in detections:
        sign_name  = det["sign"].replace("_", " ")
        confidence = det["confidence"]   # classifier %, 0-100

        if sign_name not in detected_signs:
            detected_signs.append(sign_name)
            timestamp = datetime.now().strftime("%H:%M:%S")

            if (
                len(latest_data["history"]) == 0
                or latest_data["history"][0]["sign"] != sign_name
            ):
                latest_data["history"].insert(0, {
                    "sign": sign_name,
                    "time": timestamp
                })

            _maybe_add_violation(sign_name)
            latest_data["sign_counts"][sign_name] = (
                latest_data["sign_counts"].get(sign_name, 0) + 1
            )

        if confidence > highest_conf:
            highest_conf   = confidence
            best_detection = det

    latest_data["history"]       = latest_data["history"][:20]
    latest_data["current_signs"] = detected_signs
    latest_data["confidence"]    = round(highest_conf, 2)
    latest_data["total_signs"]   = sum(latest_data["sign_counts"].values())

    _update_avg_confidence(round(highest_conf, 2))

    if latest_data["sign_counts"]:
        latest_data["top_sign"] = max(
            latest_data["sign_counts"],
            key=latest_data["sign_counts"].get
        )

    # ── Sign Inspector snapshot — the single best detection this frame ──
    if best_detection is not None:
        ok, buf = cv2.imencode(".jpg", best_detection["crop"])
        crop_b64 = base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None

        latest_data["inspector"] = {
            "sign":                  best_detection["sign"].replace("_", " "),
            "ocr":                   best_detection["ocr"],
            "yolo_confidence":       best_detection["yolo_confidence"],
            "classifier_confidence": best_detection["confidence"],
            "time":                  datetime.now().strftime("%H:%M:%S"),
            "crop_image":            crop_b64,
        }

    return detections


# ─────────────────────────────────────────────
#  LIVE FEED
# ─────────────────────────────────────────────

def generate_frames():
    camera = _get_camera()
    try:
        while True:
            start = time.time()
            success, frame = camera.read()
            if not success:
                break

            results = model(frame, conf=0.35, verbose=False)
            _process_detections(frame, results)

            elapsed = time.time() - start
            latest_data["fps"] = round(1 / elapsed, 1) if elapsed > 0 else 0

            annotated = results[0].plot()
            ret, buffer = cv2.imencode(".jpg", annotated)

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buffer.tobytes()
                + b'\r\n'
            )
    finally:
        # Runs whether the loop breaks naturally OR the browser disconnects
        # (tab switch, navigate away, page close all raise GeneratorExit on
        # the next `yield`) — so the webcam is never left running unattended.
        release_camera()


# ─────────────────────────────────────────────
#  IMAGE DETECTION
# ─────────────────────────────────────────────

def detect_image(file_storage):
    suffix = os.path.splitext(file_storage.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    frame = cv2.imread(tmp_path)
    os.unlink(tmp_path)

    if frame is None:
        return None, []

    results = model(frame, conf=0.35, verbose=False)
    detections = _process_detections(frame, results)

    annotated = results[0].plot()
    ret, buffer = cv2.imencode(".jpg", annotated)

    # Build a clean, JSON-safe payload per sign — including its own crop,
    # OCR reading, and both confidence scores — for the Sign Inspector cards.
    signs_payload = []
    for det in detections:
        ok, crop_buf = cv2.imencode(".jpg", det["crop"])
        signs_payload.append({
            "sign":                  det["sign"].replace("_", " "),
            "ocr":                   det["ocr"],
            "yolo_confidence":       det["yolo_confidence"],
            "classifier_confidence": det["confidence"],
            "crop_image":            base64.b64encode(crop_buf.tobytes()).decode("utf-8") if ok else None,
        })

    return buffer.tobytes(), signs_payload


# ─────────────────────────────────────────────
#  VIDEO — re-encode for browser
# ─────────────────────────────────────────────

def _reencode_for_browser(raw_path, final_path):
    """Re-encode mp4v → H.264 so every browser can play it."""
    if shutil.which("ffmpeg") is None:
        # No ffmpeg: just rename and hope the browser can handle mp4v
        shutil.move(raw_path, final_path)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        final_path
    ]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        shutil.move(raw_path, final_path)
    finally:
        if os.path.exists(raw_path):
            try:
                os.unlink(raw_path)
            except OSError:
                pass


# ─────────────────────────────────────────────
#  VIDEO PIPELINE  (job-ID aware, lock-protected)
# ─────────────────────────────────────────────

def _count_frames(path):
    """Return reliable frame count for a video file."""
    cap = cv2.VideoCapture(path)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if reported > 1:
        return reported, fps, w, h

    # Unreliable container — count manually (slow but accurate)
    cap2 = cv2.VideoCapture(path)
    n = 0
    while cap2.read()[0]:
        n += 1
    cap2.release()
    return n, fps, w, h


def _run_video_pipeline(job_id, tmp_path, out_path):
    """
    Process tmp_path frame-by-frame, write annotated H.264 MP4 to out_path.
    Checks job_id every frame; exits early if a newer job has started.
    """
    total_frames, fps_in, width, height = _count_frames(tmp_path)
    total_frames = max(total_frames, 1)

    # Initial state — locked write so the poller never sees a partial update
    with _video_lock:
        if _current_job_id != job_id:
            return          # already superseded before we even started
        latest_data["video_frames_done"]  = 0
        latest_data["video_frames_total"] = total_frames
        latest_data["video_done"]         = False
        latest_data["video_output_file"]  = ""

    raw_path = out_path + ".raw.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    cap      = cv2.VideoCapture(tmp_path)
    writer   = cv2.VideoWriter(raw_path, fourcc, fps_in, (width, height))

    frames_done = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Bail out if a newer job has taken over
            if not _is_current_job(job_id):
                return

            results = model(frame, conf=0.35, verbose=False)
            _process_detections(frame, results)

            annotated = results[0].plot()
            writer.write(annotated)

            frames_done += 1
            # Write progress — use the FIXED total, never change it mid-run
            _set_video_progress(job_id, frames_done, total_frames)

    finally:
        cap.release()
        writer.release()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # If we were superseded, don't touch output_file or done flag
    if not _is_current_job(job_id):
        try:
            os.unlink(raw_path)
        except OSError:
            pass
        return

    # Re-encode to H.264
    _reencode_for_browser(raw_path, out_path)

    # Mark complete
    _set_video_progress(job_id, frames_done, total_frames,
                        done=True,
                        output_file=os.path.basename(out_path))


# ─────────────────────────────────────────────
#  PUBLIC API — called by app.py
# ─────────────────────────────────────────────

def detect_video_from_path(tmp_path, out_path):
    """
    Called by app.py in a background thread.
    app.py passes the job_id it registered; we claim it here.
    """
    # The job_id was set by app.py via start_video_job() before the thread
    # started, so _current_job_id is already correct — just run the pipeline.
    global _current_job_id
    with _video_lock:
        job_id = _current_job_id      # read the ID that app.py registered

    _run_video_pipeline(job_id, tmp_path, out_path)


def start_video_job():
    """
    Called by app.py BEFORE starting the background thread.
    Registers a new job ID, cancels any previous job, resets state.
    Returns the new job ID (app.py can ignore it; it's stored globally).
    """
    global _current_job_id
    job_id = _new_job_id()
    with _video_lock:
        _current_job_id = job_id
        latest_data["video_frames_done"]  = 0
        latest_data["video_frames_total"] = 0
        latest_data["video_done"]         = False
        latest_data["video_output_file"]  = ""
    return job_id