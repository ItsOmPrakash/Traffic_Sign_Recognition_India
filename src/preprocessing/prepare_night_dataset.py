import pandas as pd
import shutil
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

FOLDS_CSV = r"data/raw/ITS_Night_dataset1/folds.csv"

SOURCE_ROOT = Path(
    "data/raw/ITS_Night_dataset1/INTSD-subset"
)

LABELS_DIR = Path(
    "data/ITS_Night_YOLO/labels"
)

OUTPUT_ROOT = Path(
    "data/ITS_Night_YOLO"
)

# -----------------------------
# Create folders
# -----------------------------

for split in ["train", "valid", "test"]:

    (OUTPUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )

# -----------------------------
# Read folds
# -----------------------------

df = pd.read_csv(FOLDS_CSV)

# -----------------------------
# Build image lookup
# -----------------------------

image_lookup = {}

for img in SOURCE_ROOT.rglob("*.jpeg"):
    image_lookup[img.name] = img

for img in SOURCE_ROOT.rglob("*.jpg"):
    image_lookup[img.name] = img

print(f"Images Found: {len(image_lookup)}")

# -----------------------------
# Copy files
# -----------------------------

train_count = 0
valid_count = 0
test_count = 0

for _, row in df.iterrows():

    filename = row["file_name"]
    fold = int(row["fold"])

    if filename not in image_lookup:
        continue

    image_path = image_lookup[filename]

    label_path = LABELS_DIR / f"{Path(filename).stem}.txt"

    if not label_path.exists():
        continue

    if fold in [0, 1, 2]:
        split = "train"
        train_count += 1

    elif fold == 3:
        split = "valid"
        valid_count += 1

    else:
        split = "test"
        test_count += 1

    shutil.copy(
        image_path,
        OUTPUT_ROOT / split / "images" / image_path.name
    )

    shutil.copy(
        label_path,
        OUTPUT_ROOT / split / "labels" / label_path.name
    )

print("\nDataset Created")
print("-" * 30)
print("Train:", train_count)
print("Valid:", valid_count)
print("Test :", test_count)