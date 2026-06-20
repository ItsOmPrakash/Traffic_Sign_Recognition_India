from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO(
    "runs/detect/runs/detect/indian_yolo11s_finetune-5/weights/best.pt"
)
# Folder containing test images
test_folder = Path("test_images")

# Find all images
images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    images.extend(test_folder.glob(ext))

print(f"\nFound {len(images)} images\n")

detected = 0

for image in sorted(images):

    results = model.predict(
        source=str(image),
        conf=0.05,
        imgsz=1280,
        save=True,       # Save prediction image
        save_txt=True,   # Save detection labels
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) > 0:
        detected += 1

        cls = int(boxes[0].cls[0])
        conf = float(boxes[0].conf[0])

        print(
            f"✅ {image.name} -> Class {cls} ({conf:.2f})"
        )
    else:
        print(
            f"❌ {image.name} -> No Detection"
        )

print("\n======================")
print(f"Detected: {detected}/{len(images)}")
print(f"Success Rate: {(detected/len(images))*100:.2f}%")
print("======================")

print("\nPredictions saved in:")
print("runs/detect/predict*/")