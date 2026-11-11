
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
image = cv2.imread("image.png")

if image is None:
    raise FileNotFoundError("Image not found!")

# 2. Convert color formats
image_rgb = cv2.cvtColor(
    image, cv2.COLOR_BGR2RGB
)

gray = cv2.cvtColor(
    image, cv2.COLOR_BGR2GRAY
)

# 3. Gaussian smoothing
smoothed = cv2.GaussianBlur(
    gray, (5, 5), 0
)

# 4. Median filtering (extra comparison)
median = cv2.medianBlur(gray, 5)

# 5. Sharpen original image
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharpened = cv2.filter2D(
    image_rgb, -1, kernel
)

# 6. Canny edge detection
edges = cv2.Canny(
    smoothed, 100, 200
)

# 7. Contour detection
contours, hierarchy = cv2.findContours(
    edges.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 8. Draw detected contours
contour_image = image_rgb.copy()

cv2.drawContours(
    contour_image,
    contours,
    -1,
    (0, 255, 0),
    2
)

# 9. Display outputs
images = [
    image_rgb,
    gray,
    smoothed,
    median,
    sharpened,
    edges,
    contour_image
]

titles = [
    "Original",
    "Grayscale",
    "Gaussian Smoothing",
    "Median Filtering",
    "Sharpened",
    "Canny Edges",
    "Contours"
]

plt.figure(figsize=(14, 8))

for i, img in enumerate(images):
    plt.subplot(2, 4, i + 1)

    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("experiment4_result.png")
plt.show()

print("Number of contours:", len(contours))
