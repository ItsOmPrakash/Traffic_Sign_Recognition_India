import json
import shutil
import re
from pathlib import Path
import yaml

# ==================================================
# CONFIG
# ==================================================

DATASETS = [
    "data/raw/ITS_dataset1",
    "data/raw/ITS_dataset2",
    "data/raw/ITS_dataset3"
]

OUTPUT = Path("data/Indian_Traffic_Signs_Unified")

# ==================================================
# HELPERS
# ==================================================

def normalize(name):
    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

# ==================================================
# LOAD CONFIGS
# ==================================================

with open("configs/class_aliases.json", "r", encoding="utf-8") as f:
    aliases = json.load(f)

with open("configs/global_class_mapping.json", "r", encoding="utf-8") as f:
    global_mapping = json.load(f)

# ==================================================
# CREATE OUTPUT STRUCTURE
# ==================================================

for split in ["train", "valid", "test"]:

    (OUTPUT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )

# ==================================================
# PROCESS DATASETS
# ==================================================

total_images = 0
total_labels = 0

for dataset_path in DATASETS:

    dataset = Path(dataset_path)
    dataset_name = dataset.name

    print(f"\nProcessing: {dataset_name}")

    # --------------------------
    # Load dataset.yaml
    # --------------------------

    yaml_file = dataset / "data.yaml"

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_classes = data["names"]

    # --------------------------
    # Process splits
    # --------------------------

    for split in ["train", "valid", "test"]:

        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"

        images = list(image_dir.glob("*.*"))

        for image_file in images:

            label_file = label_dir / f"{image_file.stem}.txt"

            if not label_file.exists():
                continue

            # ---------------------------------
            # Copy image
            # ---------------------------------

            new_image_name = (
                f"{dataset_name}_{image_file.name}"
            )

            shutil.copy2(
                image_file,
                OUTPUT / split / "images" / new_image_name
            )

            # ---------------------------------
            # Rewrite labels
            # ---------------------------------

            rewritten_lines = []

            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                old_class_id = int(parts[0])

                class_name = dataset_classes[old_class_id]

                normalized_name = normalize(class_name)

                final_name = aliases.get(
                    normalized_name,
                    normalized_name
                )

                global_id = global_mapping[final_name]

                new_line = (
                    f"{global_id} "
                    f"{parts[1]} "
                    f"{parts[2]} "
                    f"{parts[3]} "
                    f"{parts[4]}"
                )

                rewritten_lines.append(new_line)

            new_label_name = (
                f"{dataset_name}_{label_file.name}"
            )

            with open(
                OUTPUT / split / "labels" / new_label_name,
                "w"
            ) as f:

                f.write(
                    "\n".join(rewritten_lines)
                )

            total_images += 1
            total_labels += 1

# ==================================================
# GENERATE FINAL DATA.YAML
# ==================================================

sorted_classes = sorted(
    global_mapping,
    key=lambda x: global_mapping[x]
)

yaml_data = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(sorted_classes),
    "names": sorted_classes
}

with open(
    OUTPUT / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    yaml.dump(
        yaml_data,
        f,
        sort_keys=False,
        allow_unicode=True
    )

# ==================================================
# DONE
# ==================================================

print("\n" + "=" * 50)
print("UNIFIED DATASET CREATED")
print("=" * 50)

print(f"\nImages : {total_images}")
print(f"Labels : {total_labels}")
print(f"Classes: {len(sorted_classes)}")

print("\nSaved to:")
print(OUTPUT)