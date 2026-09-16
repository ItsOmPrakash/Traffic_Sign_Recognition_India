from pathlib import Path
import pandas as pd

OCR_TEST_DIR = Path("test_images")

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = sorted(
    p for p in OCR_TEST_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
)

ocr_df = pd.DataFrame({
    "file_path": [str(p).replace("\\", "/") for p in images],
    "ground_truth": ["" for _ in images],
    "condition": ["" for _ in images]
})

ocr_df.to_csv("results/ocr_test_labels.csv", index=False)

print(f"Created CSV with {len(ocr_df)} images")
display(ocr_df)