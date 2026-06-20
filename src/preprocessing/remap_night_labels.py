import json
from pathlib import Path

LABELS_DIR = Path("data/ITS_Night_YOLO/labels")

# Load files
with open("configs/global_class_mapping.json", "r") as f:
    global_mapping = json.load(f)

with open("configs/night_class_mapping.json", "r") as f:
    night_mapping = json.load(f)

with open("configs/night_category_ids.json", "r") as f:
    night_ids = json.load(f)

# Build final mapping:
# night class id -> unified class id

remap = {}

for night_id_str, night_class_name in night_ids.items():

    unified_class_name = night_mapping.get(night_class_name)

    # Skip ignored classes
    if unified_class_name is None:
        continue

    unified_class_id = global_mapping[unified_class_name]

    remap[int(night_id_str)] = int(unified_class_id)

print("=" * 50)
print("CLASS REMAPPING")
print("=" * 50)

for k, v in sorted(remap.items()):
    print(f"{k} -> {v}")

print("\nRemapping label files...")

processed = 0

for txt_file in LABELS_DIR.glob("*.txt"):

    new_lines = []

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        old_class = int(parts[0])

        if old_class not in remap:
            continue

        parts[0] = str(remap[old_class])

        new_lines.append(" ".join(parts))

    with open(txt_file, "w") as f:
        f.write("\n".join(new_lines))

    processed += 1

print("\nDone.")
print(f"Processed label files: {processed}")