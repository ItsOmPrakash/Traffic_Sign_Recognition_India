"""
Indian Traffic Sign Dataset - Shape Masking

Automatically masks traffic sign images according to their shape.

INPUT:
D:/WorkSpace/Traffic Sign Recognition System/data/raw/indian/indian_85_class/train

OUTPUT:
D:/WorkSpace/Traffic Sign Recognition System/data/processed/indian_85_classes
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path(
    r"D:\WorkSpace\Traffic Sign Recognition System\data\raw\indian\indian_85_class\train"
)

OUTPUT_DIR = Path(
    r"D:\WorkSpace\Traffic Sign Recognition System\data\processed\indian_85_classes"
)


# ============================================================
# CLASS -> SHAPE MAPPING
# ============================================================

CLASS_SHAPE_MAP = {

    # Circular signs
    "ALL_MOTOR_VEHICLE_PROHIBITED": "circle",
    "AXLE_LOAD_LIMIT": "circle",
    "BULLOCK_AND_HANDCART_PROHIBITED": "circle",
    "BULLOCK_PROHIBITED": "circle",

    "COMPULSARY_AHEAD": "circle",
    "COMPULSARY_AHEAD_OR_TURN_LEFT": "circle",
    "COMPULSARY_AHEAD_OR_TURN_RIGHT": "circle",
    "COMPULSARY_CYCLE_TRACK": "circle",
    "COMPULSARY_KEEP_LEFT": "circle",
    "COMPULSARY_KEEP_RIGHT": "circle",
    "COMPULSARY_MINIMUM_SPEED": "circle",
    "COMPULSARY_SOUND_HORN": "circle",
    "COMPULSARY_TURN_LEFT": "circle",
    "COMPULSARY_TURN_LEFT_AHEAD": "circle",
    "COMPULSARY_TURN_RIGHT": "circle",
    "COMPULSARY_TURN_RIGHT_AHEAD": "circle",

    "CYCLE_PROHIBITED": "circle",
    "HANDCART_PROHIBITED": "circle",
    "HEIGHT_LIMIT": "circle",
    "HORN_PROHIBITED": "circle",
    "LEFT_TURN_PROHIBITED": "circle",
    "LENGTH_LIMIT": "circle",
    "LOAD_LIMIT": "circle",
    "NO_ENTRY": "circle",
    "NO_PARKING": "circle",
    "NO_STOPPING_OR_STANDING": "circle",
    "OVERTAKING_PROHIBITED": "circle",
    "PASS_EITHER_SIDE": "circle",
    "RESTRICTION_ENDS": "circle",
    "RIGHT_TURN_PROHIBITED": "circle",
    "ROUNDABOUT": "circle",

    "SPEED_LIMIT_5": "circle",
    "SPEED_LIMIT_15": "circle",
    "SPEED_LIMIT_20": "circle",
    "SPEED_LIMIT_30": "circle",
    "SPEED_LIMIT_40": "circle",
    "SPEED_LIMIT_50": "circle",
    "SPEED_LIMIT_60": "circle",
    "SPEED_LIMIT_70": "circle",
    "SPEED_LIMIT_80": "circle",

    "STRAIGHT_PROHIBITED": "circle",
    "TONGA_PROHIBITED": "circle",
    "TRUCK_PROHIBITED": "circle",
    "U_TURN_PROHIBITED": "circle",
    "WIDTH_LIMIT": "circle",

    # Triangular signs
    "BARRIER_AHEAD": "triangle",
    "CATTLE": "triangle",
    "CROSS_ROAD": "triangle",
    "CYCLE_CROSSING": "triangle",
    "DANGEROUS_DIP": "triangle",
    "FALLING_ROCKS": "triangle",
    "FERRY": "triangle",
    "GAP_IN_MEDIAN": "triangle",
    "GUARDED_LEVEL_CROSSING": "triangle",
    "HUMP_OR_ROUGH_ROAD": "triangle",
    "LEFT_HAIR_PIN_BEND": "triangle",
    "LEFT_HAND_CURVE": "triangle",
    "LEFT_REVERSE_BEND": "triangle",
    "LOOSE_GRAVEL": "triangle",
    "MEN_AT_WORK": "triangle",
    "NARROW_BRIDGE": "triangle",
    "NARROW_ROAD_AHEAD": "triangle",
    "PEDESTRIAN_CROSSING": "triangle",
    "PRIORITY_FOR_ONCOMING_VEHICLES": "triangle",
    "QUAY_SIDE_OR_RIVER_BANK": "triangle",
    "RIGHT_HAIR_PIN_BEND": "triangle",
    "RIGHT_HAND_CURVE": "triangle",
    "RIGHT_REVERSE_BEND": "triangle",
    "ROAD_WIDENS_AHEAD": "triangle",
    "SCHOOL_AHEAD": "triangle",
    "SIDE_ROAD_LEFT": "triangle",
    "SIDE_ROAD_RIGHT": "triangle",
    "SLIPPERY_ROAD": "triangle",
    "STAGGERED_INTERSECTION": "triangle",
    "STEEP_ASCENT": "triangle",
    "STEEP_DESCENT": "triangle",
    "TRAFFIC_SIGNAL": "triangle",
    "TURN_RIGHT": "triangle",
    "T_INTERSECTION": "triangle",
    "UNGUARDED_LEVEL_CROSSING": "triangle",
    "Y_INTERSECTION": "triangle",

    # Special signs
    "GIVE_WAY": "inverted_triangle",
    "STOP": "octagon",

    # Rectangular / informatory
    "DIRECTION": "rectangle",
}


# ============================================================
# POLYGON GENERATOR
# ============================================================

def polygon_points(cx, cy, radius, sides, rotation=0):

    points = []

    for i in range(sides):

        angle = np.deg2rad(
            rotation + i * (360 / sides)
        )

        x = cx + radius * np.sin(angle)
        y = cy - radius * np.cos(angle)

        points.append((x, y))

    return points


# ============================================================
# CREATE MASK
# ============================================================

def create_mask(size, shape):

    width, height = size

    mask = Image.new(
        "L",
        size,
        0
    )

    draw = ImageDraw.Draw(mask)

    cx = width / 2
    cy = height / 2

    margin = min(width, height) * 0.05

    radius = (
        min(width, height) / 2
        - margin
    )

    # ---------------- CIRCLE ----------------

    if shape == "circle":

        draw.ellipse(
            [
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius
            ],
            fill=255
        )

    # ---------------- TRIANGLE ----------------

    elif shape == "triangle":

        points = polygon_points(
            cx,
            cy,
            radius,
            3,
            rotation=0
        )

        draw.polygon(
            points,
            fill=255
        )

    # ---------------- INVERTED TRIANGLE ----------------

    elif shape == "inverted_triangle":

        points = polygon_points(
            cx,
            cy,
            radius,
            3,
            rotation=180
        )

        draw.polygon(
            points,
            fill=255
        )

    # ---------------- OCTAGON ----------------

    elif shape == "octagon":

        points = polygon_points(
            cx,
            cy,
            radius,
            8,
            rotation=22.5
        )

        draw.polygon(
            points,
            fill=255
        )

    # ---------------- RECTANGLE ----------------

    elif shape == "rectangle":

        draw.rectangle(
            [0, 0, width, height],
            fill=255
        )

    else:

        raise ValueError(
            f"Unknown shape: {shape}"
        )

    return mask


# ============================================================
# APPLY MASK
# ============================================================

def apply_mask(image, shape):

    image = image.convert("RGB")

    mask = create_mask(
        image.size,
        shape
    )

    # Black background
    background = Image.new(
        "RGB",
        image.size,
        (0, 0, 0)
    )

    result = Image.composite(
        image,
        background,
        mask
    )

    return result


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    print()
    print("=" * 70)
    print("INDIAN TRAFFIC SIGN SHAPE MASKING")
    print("=" * 70)

    print()
    print("INPUT:")
    print(INPUT_DIR)

    print()
    print("OUTPUT:")
    print(OUTPUT_DIR)

    print()

    # Check input
    if not INPUT_DIR.exists():

        print("ERROR:")
        print("Input directory does not exist!")
        print(INPUT_DIR)

        return

    # Create output
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Image extensions
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    total = 0
    skipped = 0
    classes_processed = 0

    # Get class folders
    class_folders = sorted(
        [
            folder
            for folder in INPUT_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    print(
        f"Found {len(class_folders)} class folders."
    )

    print()

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_folder in class_folders:

        class_name = class_folder.name

        shape = CLASS_SHAPE_MAP.get(
            class_name
        )

        # ----------------------------------------------------
        # Missing mapping
        # ----------------------------------------------------

        if shape is None:

            print(
                f"[WARNING] No shape mapping: {class_name}"
            )

            skipped += 1

            continue

        # Output class folder
        output_class = (
            OUTPUT_DIR / class_name
        )

        output_class.mkdir(
            parents=True,
            exist_ok=True
        )

        class_count = 0

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        for image_path in sorted(
            class_folder.iterdir()
        ):

            if image_path.suffix.lower() not in extensions:

                continue

            try:

                with Image.open(image_path) as image:

                    masked = apply_mask(
                        image,
                        shape
                    )

                    output_path = (
                        output_class /
                        image_path.name
                    )

                    masked.save(
                        output_path
                    )

                total += 1
                class_count += 1

            except Exception as error:

                print(
                    f"[ERROR] {image_path.name}"
                )

                print(
                    f"        {error}"
                )

                skipped += 1

        classes_processed += 1

        print(
            f"[OK] {class_name:<45}"
            f" {shape:<20}"
            f" {class_count} images"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MASKING COMPLETED")
    print("=" * 70)

    print(
        f"Classes processed : {classes_processed}"
    )

    print(
        f"Images processed  : {total}"
    )

    print(
        f"Errors/skipped    : {skipped}"
    )

    print()
    print("OUTPUT DATASET:")
    print(OUTPUT_DIR)

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    process_dataset()