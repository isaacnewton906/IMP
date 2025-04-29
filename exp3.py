import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/color_img.jpeg', cv2.IMREAD_GRAYSCALE)

# Plot original histogram
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()

# Perform Histogram Equalization
equalized_img = cv2.equalizeHist(img)

# Plot equalized histogram
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.hist(equalized_img.ravel(), 256, [0, 256])
plt.title('Histogram after Equalization')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()
