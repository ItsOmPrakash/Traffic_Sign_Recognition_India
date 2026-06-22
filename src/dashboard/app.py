import os
import base64
import uuid
import tempfile
import threading

from flask import Flask, render_template, Response, jsonify, request, send_file

import detector
from dashboard_state import latest_data

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ── Pages ──────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

# ── Live feed ─────────────────────────────────────────────────────────────────

@app.route("/video")
def video():
    return Response(
        detector.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video/stop", methods=["POST"])
def video_stop():
    detector.release_camera()
    return jsonify({"ok": True})

# ── Dashboard data ─────────────────────────────────────────────────────────────

@app.route("/detections")
def detections():
    payload = {k: v for k, v in latest_data.items()
               if k not in ("confidence_sum", "confidence_count")}
    return jsonify(payload)

@app.route("/violations")
def violations():
    return jsonify(latest_data["violations"])

# ── Image detection ────────────────────────────────────────────────────────────

@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    jpeg_bytes, signs = detector.detect_image(f)

    if jpeg_bytes is None:
        return jsonify({"error": "Could not read image"}), 400

    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return jsonify({"image": b64, "signs": signs, "count": len(signs)})

# ── Video detection ────────────────────────────────────────────────────────────

@app.route("/detect/video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    out_name = f"output_{uuid.uuid4().hex[:8]}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    # Save upload to disk NOW — FileStorage is not thread-safe
    suffix = os.path.splitext(f.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.save(tmp.name)
    tmp_path = tmp.name
    tmp.close()

    # Reset progress immediately
    latest_data["video_frames_done"]  = 0
    latest_data["video_frames_total"] = 1
    latest_data["video_done"]         = False
    latest_data["video_output_file"]  = ""

    # Process in background so Flask can serve /progress polls
    def run():
        detector.detect_video_from_path(tmp_path, out_path)

    threading.Thread(target=run, daemon=True).start()

    # Return immediately — frontend polls /detect/video/progress
    return jsonify({"ok": True, "output_file": out_name})


@app.route("/detect/video/progress")
def video_progress():
    return jsonify({
        "frames_done":  latest_data.get("video_frames_done", 0),
        "frames_total": latest_data.get("video_frames_total", 0),
        "done":         latest_data.get("video_done", False),
        "output_file":  latest_data.get("video_output_file", "")
    })

# ── Video serve ────────────────────────────────────────────────────────────────

@app.route("/preview/video/<filename>")
def preview_video(filename):
    """Inline video playback (range-request capable)."""
    if not filename.startswith("output_") or not filename.endswith(".mp4"):
        return jsonify({"error": "Invalid filename"}), 400
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)

@app.route("/download/video/<filename>")
def download_video(filename):
    """Force-download the processed video."""
    if not filename.startswith("output_") or not filename.endswith(".mp4"):
        return jsonify({"error": "Invalid filename"}), 400
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=filename)

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)