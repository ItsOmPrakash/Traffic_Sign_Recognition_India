from ultralytics import YOLO

def main():

    model = YOLO("yolo11s.pt")

    model.train(
    data="configs/gtsrb.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    cache=False,
    amp=True,
    optimizer="AdamW",
    project="reports",
    name="gtsrb_yolo11s"
)

if __name__ == "__main__":
    main()