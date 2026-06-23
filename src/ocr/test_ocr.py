import easyocr

reader = easyocr.Reader(['en'])

images = [
    "Test_images/crop_0.jpg",
    "Test_images/crop_1.jpg",
    "Test_images/crop_2.jpg"
]

for image_path in images:

    results = reader.readtext(
        image_path
    )

    print()
    print("=" * 50)
    print(image_path)

    for result in results:

        bbox, text, confidence = result

        print(
            f"Text: {text}"
        )

        print(
            f"Confidence: {confidence:.2f}"
        )