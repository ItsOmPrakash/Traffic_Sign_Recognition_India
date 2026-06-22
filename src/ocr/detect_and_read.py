from ultralytics import YOLO
from speed_reader import read_speed_limit
import cv2
import os

# Load trained model
model = YOLO(
    r"runs/detect/runs/detect/indian_yolo11s_production_v1/weights/best.pt"
)

# Test image
image_path = r"test_images/image12.jpg"

# Run detection
results = model(
    image_path,
    conf=0.25,
    verbose=True
)

print("\nDETECTIONS")
print("=" * 50)

# Load original image
image = cv2.imread(image_path)

crop_count = 0

for result in results:

    print(f"Boxes Found: {len(result.boxes)}")

    for box in result.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls_id]

        print(
            f"Class: {class_name} | Confidence: {conf:.3f}"
        )

        # Check if this is any speed-limit sign
        if "speed_limit" in class_name:

            print("\nSpeed sign detected")

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = image[y1:y2, x1:x2]

            print("Crop Shape:", crop.shape)

            crop_path = (
                f"test_images/crop_{crop_count}.jpg"
            )

            cv2.imwrite(
                crop_path,
                crop
            )

            print(
                f"Saved crop -> {crop_path}"
            )

            crop_count += 1

            try:

                speed = read_speed_limit(
                    crop_path
                )

                if speed:

                    print(
                        f"Final Output: Speed Limit {speed}"
                    )

                else:

                    print(
                        "OCR could not read number"
                    )

            except Exception as e:

                print(
                    f"OCR Error: {e}"
                )

print("\nFinished.")