from ultralytics import YOLO
import cv2

# Load trained Indian model
model = YOLO(
     r"runs/detect/runs/detect/indian_yolo11s_production_v1/weights/best.pt"
)

image_path = "test_images/image12.jpg"

results = model(
    image_path,
    conf=0.05
)

image = cv2.imread(image_path)

for i, box in enumerate(results[0].boxes.xyxy):

    x1, y1, x2, y2 = map(int, box)

    crop = image[y1:y2, x1:x2]

    save_path = f"test_images/crop_{i}.jpg"

    cv2.imwrite(save_path, crop)

    print(f"Saved: {save_path}")