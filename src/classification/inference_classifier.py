from PIL import Image

from classifier import classify_image

IMAGE_PATH = r"D:\WorkSpace\Traffic Sign Recognition System\data\raw\indian\indian_85_class\Test\NO_ENTRY\42000.jpg"

image = Image.open(
    IMAGE_PATH
).convert("RGB")

class_name, confidence = classify_image(
    image
)

print("\nPrediction")
print("-------------------")
print(f"Class      : {class_name}")
print(f"Confidence : {confidence*100:.2f}%")