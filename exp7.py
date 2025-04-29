import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/gray_img.png', 0)

# ---------- Prewitt Operator ----------
# Define Prewitt kernels
prewitt_kernel_x = np.array([[ -1, 0, 1],
                             [ -1, 0, 1],
                             [ -1, 0, 1]])
prewitt_kernel_y = np.array([[ 1,  1,  1],
                             [ 0,  0,  0],
                             [-1, -1, -1]])

# Apply Prewitt kernels
Gx_prewitt = cv2.filter2D(img, -1, prewitt_kernel_x)
Gy_prewitt = cv2.filter2D(img, -1, prewitt_kernel_y)
edge_prewitt = cv2.addWeighted(Gx_prewitt, 0.5, Gy_prewitt, 0.5, 0)

# ---------- Sobel Operator ----------
# Apply Sobel directly
Gx_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Gy_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edge_sobel = cv2.magnitude(Gx_sobel, Gy_sobel)

# ---------- Plot Results ----------
plt.figure(figsize=(14,10))

# Prewitt Results
plt.subplot(3,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3,3,2)
plt.imshow(Gx_prewitt, cmap='gray')
plt.title('Prewitt Gx')
plt.axis('off')

plt.subplot(3,3,3)
plt.imshow(Gy_prewitt, cmap='gray')
plt.title('Prewitt Gy')
plt.axis('off')

plt.subplot(3,3,4)
plt.imshow(edge_prewitt, cmap='gray')
plt.title('Prewitt Edge Magnitude')
plt.axis('off')

# Sobel Results
plt.subplot(3,3,6)
plt.imshow(Gx_sobel, cmap='gray')
plt.title('Sobel Gx')
plt.axis('off')

plt.subplot(3,3,7)
plt.imshow(Gy_sobel, cmap='gray')
plt.title('Sobel Gy')
plt.axis('off')

plt.subplot(3,3,8)
plt.imshow(edge_sobel, cmap='gray')
plt.title('Sobel Edge Magnitude')
plt.axis('off')

plt.tight_layout()
plt.show()
