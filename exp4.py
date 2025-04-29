import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/gray_img.png', 0)

# 1. Averaging (Smoothing)
kernel_avg = np.ones((3,3), np.float32) / 9
img_avg = cv2.filter2D(img, -1, kernel_avg)

# 2. Sharpening
kernel_sharp = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
img_sharp = cv2.filter2D(img, -1, kernel_sharp)

# 3. Unsharp Masking
blurred = cv2.GaussianBlur(img, (5, 5), 0)
img_unsharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

# 4. Highboost Filtering
k = 2  # Highboost factor
img_highboost = cv2.addWeighted(img, 1.5, blurred, -0.5, 0) + k * (img - blurred)

# 5. Median Filtering
img_median = cv2.medianBlur(img, 3)

# Plot results
plt.figure(figsize=(15,10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_avg, cmap='gray')
plt.title('Averaging (Smoothing)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_sharp, cmap='gray')
plt.title('Sharpening')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_unsharp, cmap='gray')
plt.title('Unsharp Masking')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_highboost, cmap='gray')
plt.title('Highboost Filtering')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_median, cmap='gray')
plt.title('Median Filtering')
plt.axis('off')

plt.tight_layout()
plt.show()
