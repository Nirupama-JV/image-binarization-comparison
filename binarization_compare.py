
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_niblack, threshold_sauvola

# Load image (grayscale)
image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Global Thresholding
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 2. Otsu's Thresholding
_, otsu_thresh = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 3. Adaptive Mean Thresholding
adaptive_mean = cv2.adaptiveThreshold(
    image, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# 4. Adaptive Gaussian Thresholding
adaptive_gaussian = cv2.adaptiveThreshold(
    image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# 5. Niblack Thresholding
thresh_niblack = threshold_niblack(image, window_size=25, k=0.8)
niblack = image > thresh_niblack

# 6. Sauvola Thresholding
thresh_sauvola = threshold_sauvola(image, window_size=25)
sauvola = image > thresh_sauvola

# Plot results
titles = [
    "Original",
    "Global",
    "Otsu",
    "Adaptive Mean",
    "Adaptive Gaussian",
    "Niblack",
    "Sauvola"
]

images = [
    image,
    global_thresh,
    otsu_thresh,
    adaptive_mean,
    adaptive_gaussian,
    niblack,
    sauvola
]

plt.figure(figsize=(14, 8))
for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
