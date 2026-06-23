latest_data = {
    "fps": 0,
    "total_signs": 0,

    # hardware / engine status — driven by detector.py, read by the frontend
    "camera_active": False,
    "alert_engine_active": False,

    # Sign Inspector — snapshot of the single best detection on the current
    # frame: crop image, OCR reading, YOLO confidence, classifier confidence.
    # None until the first sign is detected.
    "inspector": None,

    # live confidence (current frame highest)
    "confidence": 0,

    # rolling average confidence
    "avg_confidence": 0,
    "confidence_sum": 0.0,
    "confidence_count": 0,

    "current_signs": [],
    "history": [],
    "top_sign": "NONE",
    "alerts": 0,
    "sign_counts": {},

    # list of violation dicts: {sign, time, severity}
    "violations": [],

    # video processing progress
    "video_frames_done":  0,
    "video_frames_total": 0,
    "video_done":         False,
    "video_output_file":  "",
}