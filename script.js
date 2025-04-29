document.addEventListener('DOMContentLoaded', function() {
    // Experiment files data
    const codeFiles = [
        {
            filename: 'exp1.py',
            codeId: 'code1'
        },
        {
            filename: 'exp2.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Apply different edge detection methods

# Canny edge detection
canny_edges = cv2.Canny(blur_img, 100, 200)

# Sobel edge detection
sobelx = cv2.Sobel(blur_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur_img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))
sobel_combined = cv2.bitwise_or(sobelx, sobely)

# Laplacian edge detection
laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(blur_img, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edges')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edges')
plt.axis('off')

plt.tight_layout()
plt.show()`
        },
        {
            filename: 'exp3.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert to RGB (for matplotlib display)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Apply Gaussian filter
gaussian = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Apply median filter
median = cv2.medianBlur(gray_img, 5)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(rgb_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Filter')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gaussian, cmap='gray')
plt.title('Gaussian Filter')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(median, cmap='gray')
plt.title('Median Filter')
plt.axis('off')

plt.tight_layout()
plt.show()`
        },
        {
            filename: 'exp4.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding
ret, thresh1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO_INV)

# Apply adaptive thresholding
adaptive_thresh1 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_thresh2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(thresh1, cmap='gray')
plt.title('Binary Threshold')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(thresh2, cmap='gray')
plt.title('Binary Inv Threshold')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(thresh3, cmap='gray')
plt.title('Truncate Threshold')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(thresh4, cmap='gray')
plt.title('To Zero Threshold')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(adaptive_thresh1, cmap='gray')
plt.title('Adaptive Mean')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(adaptive_thresh2, cmap='gray')
plt.title('Adaptive Gaussian')
plt.axis('off')

plt.tight_layout()
plt.show()`
        },
        {
            filename: 'exp5.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert BGR to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split the HSV image into its channels
h, s, v = cv2.split(hsv_img)

# Define range of some color in HSV (example: blue)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(img, img, mask=mask)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
plt.title('HSV Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(h, cmap='gray')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(s, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(v, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Blue Color Detection')
plt.axis('off')

plt.tight_layout()
plt.show()`
        },
        {
            filename: 'exp6.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate histogram
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

# Apply histogram equalization
eq_img = cv2.equalizeHist(gray_img)
eq_hist = cv2.calcHist([eq_img], [0], None, [256], [0, 256])

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_img)
clahe_hist = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.plot(hist)
plt.title('Histogram')
plt.xlim([0, 256])

plt.subplot(2, 3, 4)
plt.imshow(eq_img, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(clahe_img, cmap='gray')
plt.title('CLAHE')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.plot(hist, 'b', eq_hist, 'r', clahe_hist, 'g')
plt.title('Histograms Comparison')
plt.xlim([0, 256])
plt.legend(['Original', 'Equalized', 'CLAHE'])

plt.tight_layout()
plt.show()`
        },
        {
            filename: 'exp7.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Apply various morphological operations

# Define kernel
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(blurred, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(blurred, kernel, iterations=1)

# Opening - Erosion followed by dilation
opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

# Closing - Dilation followed by erosion
closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

# Morphological gradient - Difference between dilation and erosion
gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

# Top hat - Difference between input image and opening
tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

# Black hat - Difference between closing and input image
blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(gradient, cmap='gray')
plt.title('Gradient')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(tophat, cmap='gray')
plt.title('Top Hat')
plt.axis('off')

plt.tight_layout()
plt.show()`
        },
        {
            filename: 'exp8.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray_img, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Find lines using HoughLines
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
hough_img = img.copy()

# Draw the lines
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Contours')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB))
plt.title('Hough Lines')
plt.axis('off')

plt.tight_layout()
plt.show()`
        },
        {
            filename: 'exp9.py',
            content: `import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('your_img', 1)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply blur to reduce noise
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Apply Harris corner detection
harris = cv2.cornerHarris(np.float32(blur_img), 2, 3, 0.04)
harris = cv2.dilate(harris, None)

# Threshold for harris corners
threshold = 0.01 * harris.max()
corner_img = img.copy()
corner_img[harris > threshold] = [0, 0, 255]

# Apply Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(blur_img, 50, 0.01, 10)
shi_tomasi_img = img.copy()

if corners is not None:
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(shi_tomasi_img, (x, y), 5, (0, 255, 0), -1)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(blur_img, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(shi_tomasi_img, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corners')
plt.axis('off')

plt.tight_layout()
plt.show()`
        }
    ];

    // Populate the experiment list
    const expList = document.getElementById('exp-list');
    const hiddenCodeContainer = document.getElementById('hidden-code-containers');
    
    // Add the first experiment
    addExperimentItem('exp1.py', 'code1');
    
    // Add remaining experiments
    codeFiles.forEach((file, index) => {
        if (index > 0) { // Skip the first one as it's already in the HTML
            const codeId = `code${index + 2}`;
            
            // Create hidden code element
            const codeElement = document.createElement('pre');
            const code = document.createElement('code');
            code.id = codeId;
            code.textContent = file.content;
            codeElement.appendChild(code);
            hiddenCodeContainer.appendChild(codeElement);
            
            // Add experiment item to the list
            addExperimentItem(file.filename, codeId);
        }
    });
    
    function addExperimentItem(filename, codeId) {
        // Create experiment item
        const expItem = document.createElement('div');
        expItem.className = 'exp-item';
        expItem.textContent = filename;
        expItem.setAttribute('data-target', codeId);
        
        // Add double-click handler for experiment item
        expItem.addEventListener('dblclick', function() {
            const targetId = this.getAttribute('data-target');
            copyCode(targetId);
        });
        
        // Add to the list
        expList.appendChild(expItem);
    }
    
    // Function to copy code
    function copyCode(codeId) {
        const codeElement = document.getElementById(codeId);
        const codeText = codeElement.textContent;
        
        navigator.clipboard.writeText(codeText).then(() => {
            // Visual feedback - add a temporary "copied!" message
            const tempMessage = document.createElement('div');
            tempMessage.textContent = 'Copied!';
            tempMessage.style.position = 'fixed';
            tempMessage.style.top = '50%';
            tempMessage.style.left = '50%';
            tempMessage.style.transform = 'translate(-50%, -50%)';
            tempMessage.style.padding = '10px 20px';
            tempMessage.style.background = 'rgba(0, 0, 0, 0.7)';
            tempMessage.style.color = 'white';
            tempMessage.style.borderRadius = '4px';
            tempMessage.style.zIndex = '9999';
            
            document.body.appendChild(tempMessage);
            
            // Remove the message after 1.5 seconds
            setTimeout(() => {
                document.body.removeChild(tempMessage);
            }, 1500);
        }).catch(err => {
            console.error('Could not copy text: ', err);
        });
    }
    
    // Handle copy all button
    const copyAllBtn = document.getElementById('copy-all-btn');
    copyAllBtn.addEventListener('click', function() {
        // Collect all experiment code
        let allCode = '';
        codeFiles.forEach((file, index) => {
            const codeId = index === 0 ? 'code1' : `code${index + 2}`;
            const codeElement = document.getElementById(codeId);
            allCode += `# ${file.filename}\n${codeElement.textContent}\n\n`;
        });
        
        // Copy to clipboard
        navigator.clipboard.writeText(allCode).then(() => {
            // Visual feedback
            this.textContent = 'Copied!';
            this.classList.add('success');
            
            // Reset after 2 seconds
            setTimeout(() => {
                this.textContent = 'copy';
                this.classList.remove('success');
            }, 2000);
        }).catch(err => {
            console.error('Could not copy text: ', err);
        });
    });

    // Make experiment selector draggable
    const expSelector = document.getElementById('exp-selector');
    if (expSelector) {
        let isDragging = false;
        let startX, startY, startLeft, startTop;
        
        expSelector.addEventListener('mousedown', function(e) {
            // Skip if clicking a button or a control point
            if (e.target.classList.contains('control-point') || 
                e.target.classList.contains('copy-btn') ||
                e.target.classList.contains('exp-item')) {
                return;
            }
            
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = parseInt(window.getComputedStyle(expSelector).left) || 0;
            startTop = parseInt(window.getComputedStyle(expSelector).top) || 0;
            
            expSelector.style.cursor = 'grabbing';
        });
        
        document.addEventListener('mousemove', function(e) {
            if (!isDragging) return;
            
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            
            expSelector.style.position = 'absolute';
            expSelector.style.left = (startLeft + deltaX) + 'px';
            expSelector.style.top = (startTop + deltaY) + 'px';
        });
        
        document.addEventListener('mouseup', function() {
            if (isDragging) {
                isDragging = false;
                expSelector.style.cursor = 'default';
            }
        });
        
        // Make resizable using control points
        const controlPoints = expSelector.querySelectorAll('.control-point');
        controlPoints.forEach(point => {
            let isResizing = false;
            let startX, startY, startWidth, startHeight, startLeft, startTop;
            let resizeType = '';
            
            point.addEventListener('mousedown', function(e) {
                e.stopPropagation();
                isResizing = true;
                startX = e.clientX;
                startY = e.clientY;
                startWidth = expSelector.offsetWidth;
                startHeight = expSelector.offsetHeight;
                startLeft = parseInt(window.getComputedStyle(expSelector).left) || 0;
                startTop = parseInt(window.getComputedStyle(expSelector).top) || 0;
                
                // Determine resize type based on control point class
                if (point.classList.contains('control-point-top-left')) {
                    resizeType = 'tl';
                } else if (point.classList.contains('control-point-top-center')) {
                    resizeType = 'tc';
                } else if (point.classList.contains('control-point-top-right')) {
                    resizeType = 'tr';
                } else if (point.classList.contains('control-point-middle-left')) {
                    resizeType = 'ml';
                } else if (point.classList.contains('control-point-middle-right')) {
                    resizeType = 'mr';
                } else if (point.classList.contains('control-point-bottom-left')) {
                    resizeType = 'bl';
                } else if (point.classList.contains('control-point-bottom-center')) {
                    resizeType = 'bc';
                } else if (point.classList.contains('control-point-bottom-right')) {
                    resizeType = 'br';
                }
            });
            
            document.addEventListener('mousemove', function(e) {
                if (!isResizing) return;
                
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                
                expSelector.style.position = 'absolute';
                
                // Handle resizing based on the control point being dragged
                switch (resizeType) {
                    case 'tl': // top-left
                        expSelector.style.width = (startWidth - deltaX) + 'px';
                        expSelector.style.height = (startHeight - deltaY) + 'px';
                        expSelector.style.left = (startLeft + deltaX) + 'px';
                        expSelector.style.top = (startTop + deltaY) + 'px';
                        break;
                    case 'tc': // top-center
                        expSelector.style.height = (startHeight - deltaY) + 'px';
                        expSelector.style.top = (startTop + deltaY) + 'px';
                        break;
                    case 'tr': // top-right
                        expSelector.style.width = (startWidth + deltaX) + 'px';
                        expSelector.style.height = (startHeight - deltaY) + 'px';
                        expSelector.style.top = (startTop + deltaY) + 'px';
                        break;
                    case 'ml': // middle-left
                        expSelector.style.width = (startWidth - deltaX) + 'px';
                        expSelector.style.left = (startLeft + deltaX) + 'px';
                        break;
                    case 'mr': // middle-right
                        expSelector.style.width = (startWidth + deltaX) + 'px';
                        break;
                    case 'bl': // bottom-left
                        expSelector.style.width = (startWidth - deltaX) + 'px';
                        expSelector.style.height = (startHeight + deltaY) + 'px';
                        expSelector.style.left = (startLeft + deltaX) + 'px';
                        break;
                    case 'bc': // bottom-center
                        expSelector.style.height = (startHeight + deltaY) + 'px';
                        break;
                    case 'br': // bottom-right
                        expSelector.style.width = (startWidth + deltaX) + 'px';
                        expSelector.style.height = (startHeight + deltaY) + 'px';
                        break;
                }
            });
            
            document.addEventListener('mouseup', function() {
                isResizing = false;
            });
        });
    }
}); 