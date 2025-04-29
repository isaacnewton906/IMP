import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/color_img.jpeg', cv2.IMREAD_GRAYSCALE)

# 1. Image Negative
negative_img = 255 - img

# 2. Log Transformation
c = 255 / np.log(1 + np.max(img))  # Constant for scaling
log_img = c * np.log(1 + img)

# 3. Power Law Transformation (Gamma Correction)
gamma = 2.2  # Example gamma value
power_img = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

# 4. Contrast Stretching
min_pixel = np.min(img)
max_pixel = np.max(img)
contrast_stretch_img = np.uint8(((img - min_pixel) / (max_pixel - min_pixel)) * 255)

# 5. Bit Plane Extraction (extracting the 7th bit plane)
bit_plane_img = (img // 128) * 128  # Extract the most significant bit

# 6. Thresholding (Binarization)
_, threshold_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Plotting all images
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(negative_img, cmap='gray')
plt.title("Image Negative")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(log_img, cmap='gray')
plt.title("Log Transformation")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(power_img, cmap='gray')
plt.title("Power Law Transformation")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(contrast_stretch_img, cmap='gray')
plt.title("Contrast Stretching")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(bit_plane_img, cmap='gray')
plt.title("Bit Plane Extraction")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(threshold_img, cmap='gray')
plt.title("Thresholding")
plt.axis('off')

plt.show()
