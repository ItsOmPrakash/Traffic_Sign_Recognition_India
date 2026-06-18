from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11n.pt")

    model.train(
        data="configs/gtsrb.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )