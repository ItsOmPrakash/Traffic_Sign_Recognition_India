from ultralytics import YOLO

model = YOLO(
    r"runs/detect/runs/detect/indian_yolo11s_augmented_v2/weights/best.pt"
)

model.train(
    data="data/Indian_Traffic_Signs_Unified/data.yaml",

    epochs=20,
    imgsz=800,
    batch=8,

    patience=10,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.5,

    degrees=8,
    translate=0.15,
    scale=0.6,
    shear=3,
    perspective=0.001,

    fliplr=0.0,
    flipud=0.0,

    mosaic=1.0,
    mixup=0.2,

    workers=0,

    project="runs/detect",
    name="indian_yolo11s_production_v1"
)