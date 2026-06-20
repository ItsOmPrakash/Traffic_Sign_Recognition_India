from ultralytics import YOLO


def main():
    model = YOLO(
        r"runs/detect/runs/detect/indian_yolo11s_finetune-5/weights/last.pt"
    )

    model.train(
        resume=True
    )


if __name__ == "__main__":
    main()