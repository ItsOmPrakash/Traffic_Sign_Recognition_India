import cv2
import easyocr
import re

reader = easyocr.Reader(['en'], gpu=True)


def read_speed_limit(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Failed to load image")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    results = reader.readtext(gray)

    for _, text, confidence in results:

        print(
            f"Detected Text: {text} "
            f"(Conf: {confidence:.2f})"
        )

        match = re.search(r"\d+", text)

        if match:
            return int(match.group())

    return None


if __name__ == "__main__":

    speed = read_speed_limit(
        "test_images/crop_0.jpg"
    )

    if speed:
        print(
            f"Speed Limit: {speed}"
        )
    else:
        print(
            "No speed value detected"
        )