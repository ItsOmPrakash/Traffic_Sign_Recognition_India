import cv2
import os
import sys
import time
from ultralytics import YOLO

# ----------------------------------
# Import OCR
# ----------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "ocr"
        )
    )
)

from speed_reader import read_speed_limit

# ----------------------------------
# Load Model
# ----------------------------------

model = YOLO(
    r"runs/detect/runs/detect/indian_yolo11s_augmented_v2/weights/best.pt"
)

# ----------------------------------
# Webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

# ----------------------------------
# OCR Cache
# ----------------------------------

ocr_cache = {}
OCR_CACHE_TIME = 2.0

# ----------------------------------
# FPS
# ----------------------------------

prev_time = time.time()

print("=" * 50)
print("LIVE DETECTION V2")
print("Press Q to quit")
print("=" * 50)

# ----------------------------------
# Main Loop
# ----------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    results = model(
        frame,
        conf=0.40,
        verbose=False
    )

    detected_signs = set()

    # --------------------------
    # Process Detections
    # --------------------------

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            label = class_name.replace(
                "_",
                " "
            )

            # ----------------------
            # Speed Signs
            # ----------------------

            if "speed_limit" in class_name:

                key = (
                    x1 // 50,
                    y1 // 50
                )

                use_cache = False

                if key in ocr_cache:

                    value, timestamp = ocr_cache[key]

                    if current_time - timestamp < OCR_CACHE_TIME:

                        label = f"Speed Limit {value}"
                        use_cache = True

                if not use_cache:

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if crop.size > 0:

                        temp_file = "temp_speed.jpg"

                        cv2.imwrite(
                            temp_file,
                            crop
                        )

                        try:

                            speed = read_speed_limit(
                                temp_file
                            )

                            if speed:

                                label = (
                                    f"Speed Limit {speed}"
                                )

                                ocr_cache[key] = (
                                    speed,
                                    current_time
                                )

                        except Exception:
                            pass

            detected_signs.add(label)

            # ----------------------
            # Draw Box
            # ----------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ----------------------------------
    # FPS
    # ----------------------------------

    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ----------------------------------
    # Sign Panel
    # ----------------------------------

    y = 60

    cv2.putText(
        frame,
        "Detected Signs",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    y += 30

    for sign in sorted(detected_signs):

        cv2.putText(
            frame,
            sign,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y += 25

    # ----------------------------------
    # Show
    # ----------------------------------

    cv2.imshow(
        "Traffic Sign Recognition V2",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------------
# Cleanup
# ----------------------------------

cap.release()
cv2.destroyAllWindows()

if os.path.exists("temp_speed.jpg"):
    os.remove("temp_speed.jpg")