# Module 1: Foundations of Computer Vision

## Learning Objectives
By the end of this module, you should be able to:
- Understand how digital images are represented
- Perform basic image processing operations
- Implement edge detection algorithms
- Apply thresholding techniques for image segmentation

## Topics Covered
- Image representation and formats
- Basic image processing operations
- Color spaces (RGB, HSV, Grayscale)
- Image filtering and convolution
- Edge detection (Sobel, Canny)
- Histograms and thresholding

## 1. Image Representation and Formats

Digital images are represented as matrices of pixel values. Each pixel has:
- Position (x, y coordinates)
- Value (intensity or color)

### Image Types
1. **Binary images**: Pixels are either 0 or 1
2. **Grayscale images**: Pixels have intensity values (0-255)
3. **Color images**: Pixels have multiple channels (e.g., RGB)

Let's load and examine an image:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('../data/sample.jpg')

# Check image properties
print(f"Image shape: {img.shape}")
print(f"Image data type: {img.dtype}")
print(f"Image size: {img.size}")
```

```python
# Display the image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()
```

## 2. Color Spaces

### RGB Color Space
RGB represents colors as combinations of Red, Green, and Blue channels.

```python
# Split the image into its RGB channels
b, g, r = cv2.split(img)

# Display each channel
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(r, cmap='Reds')
axes[0].set_title('Red Channel')
axes[0].axis('off')

axes[1].imshow(g, cmap='Greens')
axes[1].set_title('Green Channel')
axes[1].axis('off')

axes[2].imshow(b, cmap='Blues')
axes[2].set_title('Blue Channel')
axes[2].axis('off')

plt.show()
```

### Grayscale Conversion

Converting to grayscale reduces the image to a single channel.

```python
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 5))
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

print(f"Grayscale image shape: {gray.shape}")
```

### HSV Color Space

HSV represents colors in terms of Hue, Saturation, and Value.

```python
# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Display HSV channels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(h, cmap='hsv')
axes[0].set_title('Hue')
axes[0].axis('off')

axes[1].imshow(s, cmap='gray')
axes[1].set_title('Saturation')
axes[1].axis('off')

axes[2].imshow(v, cmap='gray')
axes[2].set_title('Value')
axes[2].axis('off')

plt.show()
```

## 3. Basic Image Processing Operations

### Image Blending

Blending combines two images with specified weights.

```python
# Create a second image (we'll make a copy and modify it)
img2 = img.copy()
img2 = cv2.addWeighted(img2, 0.5, np.ones_like(img2)*255, 0.5, 0)

# Blend the images
blended = cv2.addWeighted(img, 0.7, img2, 0.3, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title('Modified')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
axes[2].set_title('Blended')
axes[2].axis('off')

plt.show()
```

### Geometric Transformations

#### Translation

```python
# Translation matrix
M = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
axes[1].set_title('Translated')
axes[1].axis('off')

plt.show()
```

#### Rotation

```python
# Get rotation matrix
center = (img.shape[1]//2, img.shape[0]//2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
axes[1].set_title('Rotated')
axes[1].axis('off')

plt.show()
```

#### Scaling

```python
# Resize the image
scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Original ({img.shape[1]}x{img.shape[0]})')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Scaled ({scaled.shape[1]}x{scaled.shape[0]})')
axes[1].axis('off')

plt.show()
```

## 4. Image Filtering and Convolution

### Blur Filters

#### Averaging

```python
# Averaging filter
blur = cv2.blur(img, (5, 5))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
axes[1].set_title('Averaging Blur')
axes[1].axis('off')

plt.show()
```

#### Gaussian Blur

```python
# Gaussian blur
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
axes[1].set_title('Gaussian Blur')
axes[1].axis('off')

plt.show()
```

#### Median Blur

```python
# Median blur
median = cv2.medianBlur(img, 5)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
axes[1].set_title('Median Blur')
axes[1].axis('off')

plt.show()
```

## 5. Edge Detection

### Sobel Edge Detection

```python
# Convert to grayscale for edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel X and Y
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Combined Sobel
sobel_combined = np.sqrt(sobelx**2 + sobely**2)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('Grayscale')
axes[0, 0].axis('off')

axes[0, 1].imshow(sobelx, cmap='gray')
axes[0, 1].set_title('Sobel X')
axes[0, 1].axis('off')

axes[1, 0].imshow(sobely, cmap='gray')
axes[1, 0].set_title('Sobel Y')
axes[1, 0].axis('off')

axes[1, 1].imshow(sobel_combined, cmap='gray')
axes[1, 1].set_title('Sobel Combined')
axes[1, 1].axis('off')

plt.show()
```

### Canny Edge Detection

```python
# Canny edge detection
edges = cv2.Canny(gray, 50, 150)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Grayscale')
axes[0].axis('off')

axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Canny Edges')
axes[1].axis('off')

plt.show()
```

## 6. Histograms and Thresholding

### Histograms

```python
# Calculate histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 5))
plt.plot(hist)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
```

### Thresholding

#### Simple Thresholding

```python
# Simple thresholding
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(thresh1, cmap='gray')
axes[1].set_title('Binary Threshold')
axes[1].axis('off')

axes[2].imshow(thresh2, cmap='gray')
axes[2].set_title('Inverse Binary Threshold')
axes[2].axis('off')

plt.show()
```

#### Adaptive Thresholding

```python
# Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(adaptive_thresh, cmap='gray')
axes[1].set_title('Adaptive Threshold')
axes[1].axis('off')

plt.show()
```

## Summary

In this module, we've covered:
1. Image representation and color spaces
2. Basic image processing operations
3. Image filtering and convolution
4. Edge detection techniques
5. Histogram analysis and thresholding

These foundational concepts form the building blocks for more advanced computer vision techniques. In the next module, we'll explore classical computer vision techniques like feature detection and matching.