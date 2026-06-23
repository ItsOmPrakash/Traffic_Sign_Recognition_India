import sys
import os
import random
import cv2

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from sign_pipeline import detect_and_classify

TEST_FOLDER = r"Test_images"

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

all_images = [

    os.path.join(
        TEST_FOLDER,
        file
    )

    for file in os.listdir(
        TEST_FOLDER
    )

    if file.lower().endswith(
        valid_extensions
    )
]

if len(all_images) == 0:

    print(
        "No images found"
    )

    exit()

sample_size = min(
    15,
    len(all_images)
)

random_images = random.sample(
    all_images,
    sample_size
)

print()
print("=" * 60)
print("YOLO + EfficientNet Random Test")
print("=" * 60)

total_images = 0
detected_images = 0

for image_path in random_images:

    print()
    print("-" * 60)

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not read:",
            image_path
        )

        continue

    detections = detect_and_classify(
        image
    )

    total_images += 1

    print(
        "Image:",
        os.path.basename(
            image_path
        )
    )

    if len(detections) == 0:

        print(
            "Detection: NONE"
        )

        continue

    detected_images += 1

    for detection in detections:

        print(
            "Sign:",
            detection["sign"]
        )

        print(
            "Classifier:",
            detection["confidence"],
            "%"
        )

        print(
            "YOLO:",
            detection["yolo_confidence"],
            "%"
        )

print()
print("=" * 60)
print(
    f"Images Tested: {total_images}"
)
print(
    f"Images Detected: {detected_images}"
)
print(
    f"Detection Rate: {(detected_images/total_images)*100:.2f}%"
)
print("=" * 60)