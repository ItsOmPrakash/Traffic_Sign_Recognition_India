from ultralytics import YOLO
from PIL import Image

import cv2

from src.classification.classifier import classify_image

# ==========================================
# CONFIG
# ==========================================

YOLO_MODEL = r"runs/detect/reports/gtsrb_yolo11s_finetune/weights/best.pt"

IMAGE_PATH = "test.jpg"

# ==========================================
# LOAD MODEL
# ==========================================

detector = YOLO(YOLO_MODEL)

print("YOLO Loaded")

# ==========================================
# DETECTION
# ==========================================

results = detector.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=False
)

image = cv2.imread(IMAGE_PATH)

for result in results:

    boxes = result.boxes

    for box in boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        crop = image[
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

        class_name, confidence = classify_image(
            pil_crop
        )

        label = (
            f"{class_name} "
            f"{confidence*100:.1f}%"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        print(
            f"Detected: {class_name}"
        )

# ==========================================
# SHOW RESULT
# ==========================================

cv2.imshow(
    "Traffic Sign Recognition",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()