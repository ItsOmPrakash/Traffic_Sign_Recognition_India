from ultralytics import YOLO
from pathlib import Path
import random

# Load model
model = YOLO(
    r"runs\detect\reports\gtsrb_yolo11s_finetune\weights\best.pt"
)

# Dataset path
dataset_path = Path(
    r"data\raw\indian\indian_85_class\test"
)

# Find all images recursively
image_extensions = ["*.jpg", "*.jpeg", "*.png"]

all_images = []
for ext in image_extensions:
    all_images.extend(dataset_path.rglob(ext))

print(f"Found {len(all_images)} images")

detected = 0
total = 20

for i in range(total):
    random_image = random.choice(all_images)

    results = model.predict(
        source=str(random_image),
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) > 0:
        detected += 1
        print(f"[{i+1}/{total}] DETECTED -> {random_image.parent.name}")
    else:
        print(f"[{i+1}/{total}] NO DETECTION -> {random_image.parent.name}")

print("\n====================")
print(f"Detection Rate: {detected}/{total}")
print(f"Success Rate: {(detected/total)*100:.2f}%")