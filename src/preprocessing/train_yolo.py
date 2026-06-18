from ultralytics import YOLO

def main():

    model = YOLO("yolo11s.pt")

    model.train(
        data="configs/gtsrb.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        cache= False,
        amp=True,
        project="reports",
        name="gtsrb_yolo11s"
    )

if __name__ == "__main__":
    main()