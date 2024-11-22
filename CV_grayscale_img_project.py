import numpy as np

# Sobel and Canny

import cv2 as cv
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv.imread('grayscale-image.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Apply Canny edge detection
edges = cv.Canny(img, 100, 200)

# Apply Sobel edge detection
# Sobel detects edges in both x (horizontal) and y (vertical) directions
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)  # Sobel on x-axis
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)  # Sobel on y-axis
sobel_combined = cv.magnitude(sobelx, sobely)      # Combine both directions

# Plot the results
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# Canny edge detection result
plt.subplot(132), plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

# Sobel edge detection result
plt.subplot(133), plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection'), plt.xticks([]), plt.yticks([])

plt.show()