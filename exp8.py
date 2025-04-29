import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/gray_img.png', 0)

# Calculate the histogram
histogram = cv2.calcHist([img], [0], None, [256], [0,256])

# Plot the histogram
plt.figure(figsize=(10,6))
plt.title('Histogram of Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()

# Apply Global Thresholding (Simple thresholding)
T = 127  # Choose a threshold value (this can be adjusted based on histogram)
_, segmented_img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

# Display results
plt.figure(figsize=(12,8))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img, cmap='gray')
plt.title('Segmented Image using Global Thresholding')
plt.axis('off')

plt.show()
