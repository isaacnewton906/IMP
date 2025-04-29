import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/binary_img.png', 0)  # Read as grayscale
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Make sure image is binary

# Define a simple structuring element (3x3 square)
kernel = np.ones((3,3), np.uint8)

# 1. Dilation
dilated = cv2.dilate(binary, kernel, iterations=1)

# 2. Erosion
eroded = cv2.erode(binary, kernel, iterations=1)

# 3. Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 4. Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 5. Boundary Extraction (Original - Eroded)
boundary = cv2.subtract(binary, eroded)

# 6. Hit-or-Miss Transformation
# Need a binary image with background = 0 and foreground = 1
binary_hitmiss = binary // 255  # Convert 255->1
kernel_hitmiss = np.array([[0,1,0],
                           [1,1,1],
                           [0,1,0]], np.uint8)
hitormiss = cv2.morphologyEx(binary_hitmiss, cv2.MORPH_HITMISS, kernel_hitmiss)

# Plotting results
titles = ['Original', 'Dilation', 'Erosion', 'Opening', 'Closing', 'Boundary', 'Hit-or-Miss']
images = [binary, dilated, eroded, opening, closing, boundary, hitormiss]

plt.figure(figsize=(14,8))
for i in range(7):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
