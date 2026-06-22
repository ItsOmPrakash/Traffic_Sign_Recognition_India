import os
import base64
import uuid

from flask import Flask, render_template, Response, jsonify, request, send_file

import detector
from dashboard_state import latest_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# ── Live feed (only streamed when requested) ───────────────────────────────────

@app.route("/video")
def video():
    return Response(
        detector.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

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
    return jsonify({
        "image": b64,
        "signs": signs,
        "count": len(signs)
    })

# ── Video detection (stream + save output) ────────────────────────────────────

@app.route("/detect/video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    # generate a unique output filename
    out_name = f"output_{uuid.uuid4().hex[:8]}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    return Response(
        detector.detect_video(f, out_path),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Output-File": out_name}
    )

@app.route("/detect/video/progress")
def video_progress():
    """Returns current processing progress (frames done / total)."""
    return jsonify({
        "frames_done":  latest_data.get("video_frames_done", 0),
        "frames_total": latest_data.get("video_frames_total", 0),
        "done":         latest_data.get("video_done", False),
        "output_file":  latest_data.get("video_output_file", "")
    })

@app.route("/preview/video/<filename>")
def preview_video(filename):
    """Stream the processed video inline for in-page playback."""
    if not filename.startswith("output_") or not filename.endswith(".mp4"):
        return jsonify({"error": "Invalid filename"}), 400
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)

@app.route("/download/video/<filename>")
def download_video(filename):
    """Serve a processed output video for download."""
    # sanitise – only allow the hex filenames we generate
    if not filename.startswith("output_") or not filename.endswith(".mp4"):
        return jsonify({"error": "Invalid filename"}), 400
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=filename)

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)