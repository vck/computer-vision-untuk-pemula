# Module 2: Classical Computer Vision Techniques

## Learning Objectives
By the end of this module, you should be able to:
- Detect and describe key features in images
- Match features between images
- Apply geometric transformations
- Create panorama images from multiple photos
- Detect objects using classical methods

## Topics Covered
- Feature detection and description (SIFT, SURF, ORB)
- Feature matching
- Image transformations (Affine, Perspective)
- Image stitching and panorama creation
- Object detection with Haar cascades
- Camera calibration and stereo vision

## 1. Feature Detection and Description

Features are distinctive points in an image that can be reliably detected and matched across different views of the same scene.

### ORB (Oriented FAST and Rotated BRIEF)

ORB is a fast and free alternative to SIFT and SURF.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('../data/sample1.jpg', 0)  # queryImage
img2 = cv2.imread('../data/sample2.jpg', 0)  # trainImage

# Check if images were loaded
if img1 is None or img2 is None:
    # Create sample images if data files don't exist
    img1 = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    # Add some shapes to make them different
    cv2.rectangle(img1, (100, 100), (300, 300), 255, -1)
    cv2.circle(img2, (250, 250), 100, 255, -1)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Draw keypoints
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(img1_kp, cmap='gray')
axes[0].set_title(f'Image 1 - {len(kp1)} keypoints')
axes[0].axis('off')

axes[1].imshow(img2_kp, cmap='gray')
axes[1].set_title(f'Image 2 - {len(kp2)} keypoints')
axes[1].axis('off')

plt.show()
```

## 2. Feature Matching

After detecting features, we need to match them between images.

```python
# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 10))
plt.imshow(img_matches)
plt.title(f'Feature Matches - Top 10 (Total: {len(matches)})')
plt.axis('off')
plt.show()

print(f"Number of matches: {len(matches)}")
if len(matches) > 0:
    print(f"Best match distance: {matches[0].distance}")
    print(f"Worst match distance: {matches[-1].distance}")
```

## 3. Image Transformations

### Affine Transformation

Affine transformations preserve lines and parallelism.

```python
# Define three points in the original image
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

# Define corresponding points in the transformed image
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Get the affine transformation matrix
M = cv2.getAffineTransform(pts1, pts2)

# Apply the transformation
img_affine = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img1, cmap='gray')
axes[0].plot(pts1[:, 0], pts1[:, 1], 'ro')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(img_affine, cmap='gray')
axes[1].plot(pts2[:, 0], pts2[:, 1], 'ro')
axes[1].set_title('Affine Transformed Image')
axes[1].axis('off')

plt.show()
```

### Perspective Transformation

Perspective transformations can simulate viewpoint changes.

```python
# Define four points in the original image
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

# Define corresponding points in the transformed image
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# Get the perspective transformation matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the transformation
img_perspective = cv2.warpPerspective(img1, M, (300, 300))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img1, cmap='gray')
axes[0].plot(pts1[:, 0], pts1[:, 1], 'ro')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(img_perspective, cmap='gray')
axes[1].plot(pts2[:, 0], pts2[:, 1], 'ro')
axes[1].set_title('Perspective Transformed Image')
axes[1].axis('off')

plt.show()
```

## 4. Image Stitching and Panorama Creation

Let's create a simple panorama using feature matching and homography.

```python
# For panorama creation, we need to find homography between images
if len(matches) > 4:  # Need at least 4 matches for homography
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is not None:
        # Warp img1 to img2's perspective
        result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        plt.figure(figsize=(15, 10))
        plt.imshow(result, cmap='gray')
        plt.title('Simple Panorama')
        plt.axis('off')
        plt.show()
    else:
        print("Homography could not be computed")
else:
    print("Not enough matches for homography computation")
```

## 5. Object Detection with Haar Cascades

Haar cascades are machine learning-based approaches for object detection.

```python
# Load the cascade classifier for face detection
# Note: This requires the Haar cascade file which may not be available
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # For demonstration, let's create a sample image with a "face-like" pattern
    sample_img = np.zeros((300, 300), dtype=np.uint8)
    # Create a simple face-like pattern
    cv2.circle(sample_img, (150, 150), 80, 255, -1)  # Face
    cv2.circle(sample_img, (120, 130), 15, 0, -1)    # Left eye
    cv2.circle(sample_img, (180, 130), 15, 0, -1)    # Right eye
    cv2.ellipse(sample_img, (150, 200), (30, 15), 0, 0, 180, 0, -1)  # Smile
    
    # Detect faces
    faces = face_cascade.detectMultiScale(sample_img, 1.1, 4)
    
    # Draw rectangles around faces
    img_faces = sample_img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_faces, (x, y), (x+w, y+h), (255, 255, 255), 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(sample_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(img_faces, cmap='gray')
    axes[1].set_title(f'Faces Detected ({len(faces)} found)')
    axes[1].axis('off')
    
    plt.show()
    
    print(f"Number of faces detected: {len(faces)}")
except:
    print("Haar cascade file not found. Skipping face detection demo.")
    
    # Show a conceptual example
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(0.5, 0.5, 'Haar Cascade Face Detection\n\nIn practice, this would detect faces\nin real images using pre-trained models', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Conceptual Example')
    ax.axis('off')
    plt.show()
```

## 6. Camera Calibration

Camera calibration is the process of determining the intrinsic and extrinsic parameters of a camera.

```python
# Conceptual example of camera calibration

# In practice, you would use a chessboard pattern and cv2.findChessboardCorners
# Here's a conceptual representation:

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.text(0.5, 0.7, 'Camera Calibration Process', ha='center', va='center', fontsize=16, transform=ax.transAxes)

ax.text(0.3, 0.5, '1. Capture images of\nchessboard pattern\nfrom different angles', ha='center', va='center', fontsize=12, transform=ax.transAxes)
ax.text(0.7, 0.5, '2. Detect corners\nusing cv2.findChessboardCorners()', ha='center', va='center', fontsize=12, transform=ax.transAxes)
ax.text(0.3, 0.3, '3. Compute camera\nmatrix with\ncv2.calibrateCamera()', ha='center', va='center', fontsize=12, transform=ax.transAxes)
ax.text(0.7, 0.3, '4. Undistort images\nusing camera parameters', ha='center', va='center', fontsize=12, transform=ax.transAxes)

ax.set_title('Conceptual Overview')
ax.axis('off')
plt.show()

print("Camera calibration involves these steps:")
print("1. Capture multiple images of a known pattern (like a chessboard)")
print("2. Detect the corners of the pattern in each image")
print("3. Use these correspondences to compute intrinsic and extrinsic parameters")
print("4. Apply the calibration to correct for lens distortion")
```

## Summary

In this module, we've covered:
1. Feature detection and description using ORB
2. Feature matching between images
3. Geometric transformations (Affine and Perspective)
4. Image stitching and panorama creation
5. Object detection with Haar cascades
6. Camera calibration concepts

These classical computer vision techniques form the foundation for many applications and provide important context for understanding modern deep learning approaches. In the next module, we'll explore how traditional machine learning algorithms can be applied to computer vision tasks.