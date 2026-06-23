import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import cv2

from PIL import Image

from classification.classifier import classify_image

# ─────────────────────────────────────────────
#  OCR — lazy-loaded so the app doesn't crash on
#  import if easyocr isn't installed yet, and so
#  startup isn't slowed down when OCR is unused.
# ─────────────────────────────────────────────

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _maybe_read_ocr(crop_bgr, class_name):
    """
    Only worth running OCR on signs that actually carry a printed number —
    running it on every crop would tank real-time FPS for no benefit.
    """
    if not class_name or "speed" not in class_name.lower():
        return None

    try:
        reader = _get_ocr_reader()
        results = reader.readtext(crop_bgr)
        text = "".join(r[1] for r in results)
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or None
    except Exception as e:
        print("OCR error:", e)
        return None


def detect_and_classify(frame, results):
    """
    Runs the EfficientNet classifier + OCR stage on boxes YOLO already found.

    frame:   raw BGR frame (numpy array)
    results: ultralytics YOLO Results — computed ONCE by the caller (detector.py)
             and passed in here, so YOLO never runs twice on the same frame.
    """

    detections = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            yolo_conf = float(
                box.conf[0]
            )

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            pil_crop = Image.fromarray(
                crop_rgb
            )

            try:

                class_name, cls_conf = (
                    classify_image(
                        pil_crop
                    )
                )

            except Exception as e:

                print(e)

                class_name = "Unknown"
                cls_conf = 0

            ocr_text = _maybe_read_ocr(crop, class_name)

            detections.append({

                "sign":
                class_name,

                "confidence":
                round(
                    cls_conf * 100,
                    2
                ),

                "yolo_confidence":
                round(
                    yolo_conf * 100,
                    2
                ),

                "box":
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "ocr":
                ocr_text,

                # raw BGR crop — caller (detector.py) encodes it to base64
                # only when it actually needs to send it to the frontend
                "crop":
                crop,

            })

    return detections
