import os
import json
import torch
import timm

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"D:\WorkSpace\Traffic Sign Recognition System\data\processed\indian_85_classes"

OUTPUT_DIR = r"D:\WorkSpace\Traffic Sign Recognition System\models\classification"

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_masked_best.pth"
)

CLASS_MAP_PATH = r"D:\WorkSpace\Traffic Sign Recognition System\configs\indian_85_class_mapping.json"

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

VAL_SIZE = 0.20

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MASKED INDIAN TRAFFIC SIGN CLASSIFICATION")
print("EfficientNet-B0 Transfer Learning")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.RandomHorizontalFlip(
        p=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading masked dataset...")

train_full = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transform
)

val_full = datasets.ImageFolder(
    DATA_DIR,
    transform=val_transform
)

num_classes = len(train_full.classes)

print(f"Classes found: {num_classes}")
print(f"Total images: {len(train_full)}")


# ============================================================
# CHECK 85 CLASSES
# ============================================================

if num_classes != 85:

    raise ValueError(
        f"Expected 85 classes but found {num_classes}."
    )


# ============================================================
# CLASS MAPPING
# ============================================================

print("\nClasses:")

for index, class_name in enumerate(
    train_full.classes
):

    print(
        f"{index:02d} -> {class_name}"
    )


# ============================================================
# SAVE IMAGEFOLDER CLASS MAPPING
# ============================================================

imagefolder_mapping = {
    class_name: index
    for index, class_name
    in enumerate(train_full.classes)
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "masked_class_mapping.json"
    ),
    "w"
) as f:

    json.dump(
        imagefolder_mapping,
        f,
        indent=4
    )


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

indices = list(
    range(len(train_full))
)

labels = train_full.targets


train_indices, val_indices = train_test_split(

    indices,

    test_size=VAL_SIZE,

    random_state=SEED,

    stratify=labels
)


print("\nDataset split:")

print(
    f"Training images:   {len(train_indices)}"
)

print(
    f"Validation images: {len(val_indices)}"
)


# ============================================================
# CREATE SUBSETS
# ============================================================

train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    val_full,
    val_indices
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=4,

    pin_memory=True
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    pin_memory=True
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading ImageNet pretrained EfficientNet-B0...")

model = timm.create_model(

    "efficientnet_b0",

    pretrained=True,

    num_classes=num_classes
)


model = model.to(DEVICE)


print("EfficientNet-B0 loaded successfully.")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    correct = 0

    total = 0


    progress = tqdm(

        train_loader,

        desc="Training"
    )


    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        accuracy = (
            100 * correct / total
        )


        progress.set_postfix(

            loss=f"{loss.item():.4f}",

            acc=f"{accuracy:.2f}%"
        )


    return (

        total_loss / len(train_loader),

        100 * correct / total

    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            total_loss += loss.item()


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    return (

        total_loss / len(val_loader),

        100 * correct / total

    )


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        validate()
    )


    scheduler.step(
        val_accuracy
    )


    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Learning Rate: {current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(

            model.state_dict(),

            MODEL_PATH

        )


        print(
            f"✓ Best model saved: {MODEL_PATH}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"\nModel saved at:"
)

print(MODEL_PATH)

print("\nTraining finished successfully.")