
import cv2
import os
from glob import glob

image_folder = r"test_videos/images"
output_video = r"test_videos/benchmark_video.mp4"

fps = 30
duration_per_image = 1  # seconds

image_paths = []

for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(
        glob(os.path.join(image_folder, ext))
    )

image_paths.sort()

if len(image_paths) == 0:
    print("No images found")
    exit()

first = cv2.imread(image_paths[0])

height, width = first.shape[:2]

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frames_per_image = fps * duration_per_image

print(f"Images Found: {len(image_paths)}")

for img_path in image_paths:

    img = cv2.imread(img_path)

    img = cv2.resize(
        img,
        (width, height)
    )

    filename = os.path.basename(img_path)

    cv2.putText(
        img,
        filename,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    for _ in range(frames_per_image):
        writer.write(img)

writer.release()

print()
print("Benchmark video created:")
print(output_video)