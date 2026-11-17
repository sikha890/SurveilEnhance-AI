
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# PROJECT: SURVEILLANCE SUPER-RESOLUTION
# PHASE 2: COMPUTER VISION FUNDAMENTALS
# EXPERIMENT 01: IMAGE BASICS
# ==========================================

print("=" * 55)
print("EXPERIMENT 01 - UNDERSTANDING DIGITAL IMAGES")
print("=" * 55)

# 1. Create a synthetic image
height = 256
width = 256

image = np.zeros((height, width, 3), dtype=np.uint8)

# Add colored shapes (OpenCV uses BGR)
image[20:100, 20:100] = [0, 0, 255]      # Red
image[20:100, 120:200] = [0, 255, 0]    # Green
image[120:200, 20:100] = [255, 0, 0]    # Blue

# Add a white circle
cv2.circle(image, (160, 160), 35, (255, 255, 255), -1)

# 2. Understand image dimensions
print("\n1. IMAGE DIMENSIONS")

print("Image shape:", image.shape)
print("Height:", image.shape[0])
print("Width:", image.shape[1])
print("Channels:", image.shape[2])

# 3. Understand data type
print("\n2. DATA TYPE")

print("Data type:", image.dtype)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())

# 4. Inspect individual pixels
print("\n3. PIXEL VALUES")

print("Red square pixel (BGR):", image[50, 50])
print("Green square pixel (BGR):", image[50, 150])
print("Blue square pixel (BGR):", image[150, 50])

# 5. Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("\n4. COLOR CONVERSION")
print("RGB red pixel:", rgb_image[50, 50])

# 6. Convert RGB to grayscale
gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

print("\n5. GRAYSCALE")
print("Grayscale shape:", gray_image.shape)

# 7. Save generated images
output_dir = Path("outputs/images")
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(output_dir / "synthetic_bgr.png"),
    image
)

cv2.imwrite(
    str(output_dir / "synthetic_gray.png"),
    gray_image
)

# 8. Visualize images
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(rgb_image)
plt.title("Original RGB Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rgb_image[20:100, 20:100])
plt.title("Red Region")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nExperiment 01 completed successfully!")
