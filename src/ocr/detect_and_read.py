import cv2
from ultralytics import YOLO
from speed_reader import read_speed_limit

# -----------------------------
# Load model
# -----------------------------

model = YOLO(
    r"runs/detect/runs/detect/indian_yolo11s_finetune-5/weights/best.pt"
)

# -----------------------------
# Image
# -----------------------------

image_path = r"test_images/image12.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Could not load image")
    exit()

# -----------------------------
# Detection
# -----------------------------

results = model(
    image_path,
    conf=0.01
)

crop_id = 0

print("\nDETECTIONS")
print("=" * 50)

for result in results:

    print(f"Boxes Found: {len(result.boxes)}")

    for box in result.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        print(
            f"Class: {cls_id} | "
            f"Confidence: {conf:.3f}"
        )

        # speed_limit class
        if cls_id != 87:
            continue

        print("\nSpeed sign detected")

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Add padding
        h, w = img.shape[:2]

        pad = 20

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = img[y1:y2, x1:x2]

        # Skip invalid crops
        if crop.size == 0:
            print("Empty crop. Skipping.")
            continue

        print("Crop Shape:", crop.shape)

        crop_path = (
            f"test_images/crop_{crop_id}.jpg"
        )

        cv2.imwrite(
            crop_path,
            crop
        )

        print(
            f"Saved crop -> {crop_path}"
        )

        try:

            speed = read_speed_limit(
                crop_path
            )

            if speed:

                print(
                    f"Detected Speed Limit: {speed}"
                )

            else:

                print(
                    "OCR could not read speed value"
                )

        except Exception as e:

            print(
                f"OCR Error: {e}"
            )

        crop_id += 1

print("\nFinished.")