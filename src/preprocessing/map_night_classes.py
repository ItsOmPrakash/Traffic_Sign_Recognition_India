import json
from pathlib import Path

# Load unified classes
with open("configs/global_class_mapping.json", "r") as f:
    unified_classes = json.load(f)

# Load night mapping
with open("configs/night_class_mapping.json", "r") as f:
    night_mapping = json.load(f)

name_to_id = unified_classes

print("=" * 50)
print("NIGHT → UNIFIED CLASS MAPPING")
print("=" * 50)

mapped = 0
ignored = 0

for night_class, unified_class in night_mapping.items():

    if unified_class is None:
        print(f"{night_class} -> IGNORED")
        ignored += 1
        continue

    if unified_class not in name_to_id:
        print(f"ERROR: {unified_class} not found")
        continue

    print(
        f"{night_class} -> "
        f"{unified_class} "
        f"(ID {name_to_id[unified_class]})"
    )

    mapped += 1

print("\n")
print(f"Mapped Classes : {mapped}")
print(f"Ignored Classes: {ignored}")