import cv2
import os
import sys
from ultralytics import YOLO

# ----------------------------
# OCR Import
# ----------------------------

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

# ----------------------------
# Load Model
# ----------------------------

model = YOLO(
    r"runs/detect/runs/detect/indian_yolo11s_production_v1/weights/best.pt"
)
# ----------------------------
# Video Path
# ----------------------------

video_path = r"test_videos\benchmark_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

# ----------------------------
# Output Video
# ----------------------------

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

os.makedirs("results", exist_ok=True)

output_path = "results/output_video.mp4"

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0

print("=" * 50)
print("VIDEO DETECTION STARTED")
print("=" * 50)

# ----------------------------
# Process Video
# ----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model(
        frame,
        conf=0.30,
        verbose=False
    )

    detected_signs = []

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

            # --------------------
            # OCR Speed Signs
            # --------------------

            if "speed_limit" in class_name:

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:

                    temp_file = "temp_crop.jpg"

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

                    except:
                        pass

            detected_signs.append(label)

            # --------------------
            # Draw Box
            # --------------------

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

    # ----------------------------
    # Detected Sign Count
    # ----------------------------

    cv2.putText(
        frame,
        f"Signs: {len(detected_signs)}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    writer.write(frame)

    cv2.imshow(
        "Traffic Sign Video Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------
# Cleanup
# ----------------------------

cap.release()
writer.release()

cv2.destroyAllWindows()

if os.path.exists("temp_crop.jpg"):
    os.remove("temp_crop.jpg")

print()
print("=" * 50)
print("Processing Complete")
print("Saved:", output_path)
print("=" * 50)