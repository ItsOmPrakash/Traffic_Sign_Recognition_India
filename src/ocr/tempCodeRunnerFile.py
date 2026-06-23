import easyocr

reader = easyocr.Reader(['en'])

image_path = "test_images/crop_0.jpg"

results = reader.readtext(image_path)

print("\nOCR RESULTS")
print("=" * 40)

for result in results:
    bbox, text, confidence = result

    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 20)