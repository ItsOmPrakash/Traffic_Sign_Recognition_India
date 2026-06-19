import os
import json
import random
import torch
import timm

from PIL import Image
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================

TEST_DIR = "data/raw/indian/indian_85_class/Test"

MODEL_PATH = "models/classification/efficientnet_b0_best.pth"

CLASS_MAP_PATH = "configs/indian_85_class_mapping.json"

NUM_SAMPLES = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# LOAD CLASS MAP
# ==========================================

with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)

idx_to_class = {
    v: k for k, v in class_map.items()
}

# ==========================================
# MODEL
# ==========================================

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(class_map)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# COLLECT IMAGES
# ==========================================

all_images = []

for class_name in os.listdir(TEST_DIR):

    class_path = os.path.join(
        TEST_DIR,
        class_name
    )

    if os.path.isdir(class_path):

        for image_name in os.listdir(class_path):

            image_path = os.path.join(
                class_path,
                image_name
            )

            all_images.append(
                (image_path, class_name)
            )

# ==========================================
# RANDOM SAMPLE
# ==========================================

samples = random.sample(
    all_images,
    NUM_SAMPLES
)

correct = 0

print("\nTesting Random Images\n")

for image_path, true_class in samples:

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    predicted_class = idx_to_class[
        pred.item()
    ]

    is_correct = (
        predicted_class == true_class
    )

    if is_correct:
        correct += 1

    print(
        f"\nTrue      : {true_class}"
    )

    print(
        f"Predicted : {predicted_class}"
    )

    print(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    print(
        f"Correct   : {is_correct}"
    )

# ==========================================
# SUMMARY
# ==========================================

accuracy = (
    correct / NUM_SAMPLES
) * 100

print("\n" + "="*50)

print(
    f"Random Sample Accuracy: {accuracy:.2f}%"
)

print("="*50)