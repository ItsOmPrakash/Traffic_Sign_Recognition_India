from ultralytics import YOLO
from speed_reader import read_speed_limit
import cv2

# Load model
model = YOLO(
    r"runs/detect/runs/detect/indian_yolo11s_augmented_v2/weights/best.pt"
)

image_path = r"test_images/image12.jpg"

results = model(
    image_path,
    conf=0.25
)

image = cv2.imread(image_path)

print("\nDETECTED SIGNS")
print("=" * 50)

signs_found = set()

crop_count = 0

for result in results:

    for box in result.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls_id]

        # SPEED LIMIT SIGN
        if "speed_limit" in class_name:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = image[y1:y2, x1:x2]

            crop_path = (
                f"test_images/crop_{crop_count}.jpg"
            )

            cv2.imwrite(
                crop_path,
                crop
            )

            crop_count += 1

            speed = read_speed_limit(
                crop_path
            )

            if speed:

                signs_found.add(
                    f"Speed Limit {speed}"
                )

            else:

                signs_found.append(
                    "Speed Limit Sign"
                )

        # OTHER SIGNS
        else:

            readable_name = (
                class_name
                .replace("_", " ")
                .title()
            )

            signs_found.add(
                readable_name
            )

print()

for i, sign in enumerate(sorted(signs_found), 1):
    print(f"{i}. {sign}")

print("\nTotal Signs:", len(signs_found))