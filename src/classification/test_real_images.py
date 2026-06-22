import os
from PIL import Image

from classifier import classify_image

TEST_DIR = r"Test_images"

print("\nTesting Real Images\n")

for file_name in os.listdir(TEST_DIR):

    if not file_name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):
        continue

    image_path = os.path.join(
        TEST_DIR,
        file_name
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        class_name, confidence = classify_image(
            image
        )

        print("=" * 50)
        print(f"Image      : {file_name}")
        print(f"Predicted  : {class_name}")
        print(
            f"Confidence : {confidence * 100:.2f}%"
        )

    except Exception as e:

        print(
            f"\nFailed: {file_name}"
        )

        print(e)