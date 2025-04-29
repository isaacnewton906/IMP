import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread('C:/Users/Madhav/Desktop/IPMV_Python/gray_img.png', 0)

# Perform Fourier Transform
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center
m, n = img.shape
crow, ccol = m//2 , n//2   # center

# Define Cutoff
D0 = 50  # You can change this value

# Create filters
u = np.arange(m)
v = np.arange(n)
u, v = np.meshgrid(u, v, indexing='ij')
D = np.sqrt((u - crow)**2 + (v - ccol)**2)

# Ideal Low Pass Filter
H_ideal = np.zeros((m,n))
H_ideal[D <= D0] = 1

# Butterworth Low Pass Filter
n_order = 2  # Filter order
H_butter = 1 / (1 + (D/D0)**(2*n_order))

# Gaussian Low Pass Filter
H_gaussian = np.exp(-(D**2) / (2*(D0**2)))

# Apply each filter
G_ideal = dft_shift * H_ideal
G_butter = dft_shift * H_butter
G_gaussian = dft_shift * H_gaussian

# Inverse Fourier Transform
img_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(G_ideal)))
img_butter = np.abs(np.fft.ifft2(np.fft.ifftshift(G_butter)))
img_gaussian = np.abs(np.fft.ifft2(np.fft.ifftshift(G_gaussian)))

# Plot results
titles = ['Original', 'Ideal LPF', 'Butterworth LPF', 'Gaussian LPF']
images = [img, img_ideal, img_butter, img_gaussian]

plt.figure(figsize=(12,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
