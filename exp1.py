import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Read color image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/color_img.jpeg', 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display properties
print("Color Image")
print("Size:", img.size)
print("Shape:", img.shape)
print("Class:", img.__class__)
print("Datatype:", img.dtype)
print("---------------------------------")
print("Gray Image")
print("Size:", gray_img.size)
print("Shape:", gray_img.shape)
print("Class:", gray_img.__class__)
print("Datatype:", gray_img.dtype)
print("---------------------------------")

# Resize image to half
height, width = img.shape[:2]
resized_img = cv2.resize(img, (width//2, height//2))

# Rotate images
right_img = cv2.flip(img, 1)
bottom_img = cv2.flip(img, 0)

# Crop part of the image
crop_img = img[50:150, 100:200]

# Modify some image pixels
mod_img = img.copy()
mod_img[50:100, 0:50] = [255, 255, 255]  # Make white
mod_img[200:150, 0:50] = [0, 0, 0]        # Make black

# Plotting images
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(gray_img, cmap='gray')
plt.title("Gray Image")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
plt.title("Right Image")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(bottom_img, cv2.COLOR_BGR2RGB))
plt.title("Bottom Image")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
plt.title("Cropped Image")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(mod_img, cv2.COLOR_BGR2RGB))
plt.title("Modified Image")
plt.axis('off')

plt.show()
