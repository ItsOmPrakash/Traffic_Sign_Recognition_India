from ultralytics import YOLO

model = YOLO(
    r"runs/detect/runs/detect/indian_yolo11s_augmented_v2/weights/best.pt"
)

results = model(
    "test_images/image13.jpg",
    conf=0.01
)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        print(
            model.names[cls],
            conf
        )