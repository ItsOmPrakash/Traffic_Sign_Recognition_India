from pathlib import Path
import shutil
import random

random.seed(42)

SOURCE_IMAGES = Path("data/processed/gtsrb/images")
SOURCE_LABELS = Path("data/processed/gtsrb/labels")

TRAIN_IMAGES = Path("data/train/images")
TRAIN_LABELS = Path("data/train/labels")

VAL_IMAGES = Path("data/val/images")
VAL_LABELS = Path("data/val/labels")

for folder in [
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VAL_IMAGES,
    VAL_LABELS,
]:
    folder.mkdir(parents=True, exist_ok=True)

images = list(SOURCE_IMAGES.glob("*.png"))

random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

print(f"Train Images: {len(train_images)}")
print(f"Val Images: {len(val_images)}")

for img_path in train_images:

    label_path = SOURCE_LABELS / img_path.with_suffix(".txt").name

    shutil.copy(img_path, TRAIN_IMAGES / img_path.name)
    shutil.copy(label_path, TRAIN_LABELS / label_path.name)

for img_path in val_images:

    label_path = SOURCE_LABELS / img_path.with_suffix(".txt").name

    shutil.copy(img_path, VAL_IMAGES / img_path.name)
    shutil.copy(label_path, VAL_LABELS / label_path.name)

print("Dataset split complete.")