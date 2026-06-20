import yaml
import json
import re

yaml_files = [
    r"data\raw\ITS_dataset1\data.yaml",
    r"data\raw\ITS_dataset2\data.yaml",
    r"data\raw\ITS_dataset3\data.yaml",
]

all_classes = {}

def normalize(name):
    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

for file in yaml_files:

    with open(file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for cls in data["names"]:

        normalized = normalize(cls)

        if normalized not in all_classes:
            all_classes[normalized] = []

        if cls not in all_classes[normalized]:
            all_classes[normalized].append(cls)

print(f"\nUnique Classes: {len(all_classes)}")

with open("configs/master_classes.json", "w", encoding="utf-8") as f:
    json.dump(all_classes, f, indent=4)

print("\nSaved -> configs/master_classes.json")