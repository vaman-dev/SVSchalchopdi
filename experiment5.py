
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Create and save a sample input image
canvas = np.full((500, 500, 3), 40, dtype=np.uint8)

triangle = np.array([
    [100, 100],
    [50, 200],
    [150, 200]
], dtype=np.int32)

cv2.fillPoly(canvas, [triangle], (0, 0, 255))
cv2.rectangle(canvas, (300, 100), (420, 220), (0, 255, 0), -1)
cv2.circle(canvas, (250, 380), 60, (255, 0, 0), -1)

cv2.imwrite("shapes.png", canvas)


# Load the input image
image = cv2.imread("shapes.png")

if image is None:
    raise FileNotFoundError("Image not found")

show_image(image, "Original Image")

result = image.copy()

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Define HSV ranges for each color
color_ranges = {
    "Red": [
        ([0, 100, 80], [10, 255, 255]),
        ([170, 100, 80], [179, 255, 255])
    ],
    "Green": [
        ([35, 80, 80], [85, 255, 255])
    ],
    "Blue": [
        ([90, 80, 80], [130, 255, 255])
    ]
}

kernel = np.ones((3, 3), np.uint8)

detected_objects = 0


for color, ranges in color_ranges.items():

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Create color masks
    for lower, upper in ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, color_mask)

    # Remove noise using morphological operations
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    # Detect contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        # Approximate the shape
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour, 0.04 * perimeter, True
        )

        corners = len(approx)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(contour)

        # Identify the geometric shape
        if corners == 3:
            shape = "Triangle"

        elif corners == 4:
            ratio = w / h

            if 0.85 <= ratio <= 1.15:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif circularity >= 0.80:
            shape = "Circle"

        else:
            shape = "Polygon"

        # Combine detected color and shape
        label = f"{color} {shape}"

        # Draw boundary and label
        cv2.drawContours(
            result, [contour], -1, (255, 255, 255), 2
        )

        cv2.putText(
            result,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        detected_objects += 1
        print("Detected:", label)


# Save and display the final result
cv2.imwrite("recognized_objects.png", result)

show_image(result, "Recognized Objects")

print("Total objects detected:", detected_objects)
print("Output saved as recognized_objects.png")
