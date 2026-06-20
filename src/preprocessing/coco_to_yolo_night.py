import json
from pathlib import Path

# Paths
COCO_JSON = r"data/raw/ITS_Night_dataset1/annotations.json"

OUTPUT_DIR = Path("data/ITS_Night_YOLO")
LABELS_DIR = OUTPUT_DIR / "labels"

LABELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading annotations...")

with open(COCO_JSON, "r") as f:
    coco = json.load(f)

# Image info lookup
images = {}

for img in coco["images"]:
    images[img["id"]] = {
        "filename": img["file_name"],
        "width": img["width"],
        "height": img["height"]
    }

# Category mapping
cat_map = {}

for idx, cat in enumerate(coco["categories"]):
    cat_map[cat["id"]] = idx

print(f"Categories: {len(cat_map)}")

# Create labels
for ann in coco["annotations"]:

    image_id = ann["image_id"]

    img = images[image_id]

    width = img["width"]
    height = img["height"]

    x, y, w, h = ann["bbox"]

    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height

    w /= width
    h /= height

    cls = cat_map[ann["category_id"]]

    label_file = LABELS_DIR / f"{Path(img['filename']).stem}.txt"

    with open(label_file, "a") as f:
        f.write(
            f"{cls} {x_center} {y_center} {w} {h}\n"
        )

print("Done!")
print(f"Labels saved to: {LABELS_DIR}")