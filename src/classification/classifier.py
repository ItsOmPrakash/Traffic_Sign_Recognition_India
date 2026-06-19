import json
import torch
import timm

from PIL import Image
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "models/classification/efficientnet_b0_best.pth"

CLASS_MAP_PATH = "configs/indian_85_class_mapping.json"

IMAGE_SIZE = 224

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
# TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD MODEL
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

print("Classifier Loaded Successfully")


# ==========================================
# CLASSIFICATION FUNCTION
# ==========================================

def classify_image(image):

    """
    image -> PIL Image

    Returns:
        class_name
        confidence
    """

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    class_name = idx_to_class[
        prediction.item()
    ]

    confidence = confidence.item()

    return class_name, confidence