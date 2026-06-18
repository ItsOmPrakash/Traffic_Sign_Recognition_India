from ultralytics import YOLO

def main():

    model = YOLO(
        "runs/detect/reports/gtsrb_yolo11s-4/weights/best.pt"
    )

    model.train(
        data="configs/gtsrb.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        cache=False,
        amp=True,
        optimizer="AdamW",
        project="reports",
        name="gtsrb_yolo11s_finetune"
    )

if __name__ == "__main__":
    main()