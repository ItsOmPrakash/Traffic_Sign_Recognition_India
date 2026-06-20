import json

# Load master classes
with open("configs/master_classes.json", "r", encoding="utf-8") as f:
    master_classes = json.load(f)

# Load aliases
with open("configs/class_aliases.json", "r", encoding="utf-8") as f:
    aliases = json.load(f)

# Build final class list
final_classes = set()

for cls in master_classes.keys():
    final_classes.add(
        aliases.get(cls, cls)
    )

# Sort alphabetically
final_classes = sorted(list(final_classes))

# Create global mapping
global_mapping = {}

for idx, cls in enumerate(final_classes):
    global_mapping[cls] = idx

# Save mapping
with open(
    "configs/global_class_mapping.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        global_mapping,
        f,
        indent=4
    )

print("=" * 50)
print("GLOBAL CLASS MAPPING CREATED")
print("=" * 50)

print(f"\nTotal Classes: {len(final_classes)}")

print("\nFirst 20 Classes:\n")

for i, cls in enumerate(final_classes[:20]):
    print(f"{i:03d} -> {cls}")

print("\nSaved:")
print("configs/global_class_mapping.json")