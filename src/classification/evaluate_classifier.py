from torchvision import datasets
import json

dataset = datasets.ImageFolder(
    "data/raw/indian/indian_85_class/Train"
)

with open(
    "models/classification/class_mapping.json",
    "w"
) as f:
    json.dump(
        dataset.class_to_idx,
        f,
        indent=4
    )

print("Saved class mapping")