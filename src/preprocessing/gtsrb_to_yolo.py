import pandas as pd
from pathlib import Path
import shutil

# Paths
ROOT = Path("data/raw/gtsrb")
CSV_FILE = ROOT / "Train.csv"

OUTPUT_IMAGES = Path("data/processed/gtsrb/images")
OUTPUT_LABELS = Path("data/processed/gtsrb/labels")

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_FILE)

print(f"Found {len(df)} annotations")

for _, row in df.iterrows():

    width = row["Width"]
    height = row["Height"]

    x1 = row["Roi.X1"]
    y1 = row["Roi.Y1"]
    x2 = row["Roi.X2"]
    y2 = row["Roi.Y2"]

    class_id = int(row["ClassId"])

    image_path = ROOT / row["Path"]

    if not image_path.exists():
        continue

    # YOLO format
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height

    bbox_width = (x2 - x1) / width
    bbox_height = (y2 - y1) / height

    label_text = (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{bbox_width:.6f} "
        f"{bbox_height:.6f}"
    )

    image_name = image_path.name

    shutil.copy(
        image_path,
        OUTPUT_IMAGES / image_name
    )

    label_file = (
        OUTPUT_LABELS /
        image_name.replace(".png", ".txt")
    )

    with open(label_file, "w") as f:
        f.write(label_text)

print("Conversion completed.")