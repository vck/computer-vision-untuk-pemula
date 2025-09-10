# Module 3: Machine Learning for Computer Vision

## Learning Objectives
By the end of this module, you should be able to:
- Extract meaningful features from images
- Apply traditional ML algorithms to computer vision tasks
- Implement image classification systems
- Perform image segmentation using clustering
- Apply dimensionality reduction techniques

## Topics Covered
- Traditional ML algorithms for image classification
- Feature extraction techniques (HOG, LBP)
- Support Vector Machines (SVM) for image classification
- K-Means clustering for image segmentation
- Principal Component Analysis (PCA) for dimensionality reduction

## 1. Feature Extraction Techniques

### HOG (Histogram of Oriented Gradients)

HOG is a feature descriptor used in computer vision and image processing for object detection.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Create a sample image
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 30, 128, -1)

# Extract HOG features
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, channel_axis=None)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(hog_image, cmap='gray')
axes[1].set_title('HOG Features')
axes[1].axis('off')

plt.show()

print(f"HOG feature vector shape: {fd.shape}")
print(f"HOG image shape: {hog_image.shape}")
```

### LBP (Local Binary Patterns)

LBP is a type of visual descriptor used for classification.

```python
from skimage.feature import local_binary_pattern

# Parameters for LBP
radius = 3
n_points = 8 * radius

# Compute LBP
lbp = local_binary_pattern(img, n_points, radius, method='uniform')

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(lbp, cmap='gray')
axes[1].set_title('LBP Features')
axes[1].axis('off')

plt.show()

print(f"LBP image shape: {lbp.shape}")
print(f"Unique LBP values: {len(np.unique(lbp))}")
```

## 2. Support Vector Machines (SVM) for Image Classification

Let's create a simple image classification example using SVM.

```python
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create a simple dataset of shapes
def create_shape_dataset(n_samples=200):
    X = []
    y = []
    
    for i in range(n_samples):
        img = np.zeros((64, 64), dtype=np.uint8)
        
        if i % 2 == 0:  # Circle
            center = (np.random.randint(20, 44), np.random.randint(20, 44))
            radius = np.random.randint(8, 15)
            cv2.circle(img, center, radius, 255, -1)
            y.append(0)  # Class 0: Circle
        else:  # Rectangle
            pt1 = (np.random.randint(10, 25), np.random.randint(10, 25))
            pt2 = (np.random.randint(35, 50), np.random.randint(35, 50))
            cv2.rectangle(img, pt1, pt2, 255, -1)
            y.append(1)  # Class 1: Rectangle
            
        # Extract HOG features
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), channel_axis=None)
        X.append(fd)
    
    return np.array(X), np.array(y)

# Generate dataset
X, y = create_shape_dataset(200)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
```

```python
# Train SVM classifier
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Circle', 'Rectangle']))
```

## 3. K-Means Clustering for Image Segmentation

K-Means can be used for image segmentation by clustering pixels based on their color values.

```python
from sklearn.cluster import KMeans

# Create a sample image with different color regions
img_color = np.zeros((200, 200, 3), dtype=np.uint8)
img_color[0:100, 0:100] = [255, 0, 0]      # Red quadrant
img_color[0:100, 100:200] = [0, 255, 0]    # Green quadrant
img_color[100:200, 0:100] = [0, 0, 255]    # Blue quadrant
img_color[100:200, 100:200] = [255, 255, 0] # Yellow quadrant

# Add some noise
noise = np.random.randint(-20, 20, img_color.shape, dtype=np.int16)
img_color = np.clip(img_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Reshape image to be a list of pixels
pixel_list = img_color.reshape((-1, 3))

# Apply K-Means clustering
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixel_list)

# Get cluster assignments for each pixel
clustered = kmeans.cluster_centers_[kmeans.labels_]
clustered_img = clustered.reshape(img_color.shape).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB))
axes[1].set_title('K-Means Segmented Image')
axes[1].axis('off')

plt.show()

print(f"Number of clusters: {k}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
```

## 4. Principal Component Analysis (PCA) for Dimensionality Reduction

PCA can be used to reduce the dimensionality of image data while preserving important information.

```python
from sklearn.decomposition import PCA

# Create a dataset of face-like images
def create_face_dataset(n_samples=50):
    X = []
    for i in range(n_samples):
        img = np.zeros((64, 64), dtype=np.uint8)
        
        # Add some variation to create different "faces"
        # Face (ellipse)
        center = (32 + np.random.randint(-5, 5), 32 + np.random.randint(-5, 5))
        axes = (20 + np.random.randint(-3, 3), 25 + np.random.randint(-3, 3))
        cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)
        
        # Eyes
        eye_y = center[1] - 8
        left_eye_x = center[0] - 8 + np.random.randint(-2, 2)
        right_eye_x = center[0] + 8 + np.random.randint(-2, 2)
        cv2.circle(img, (left_eye_x, eye_y), 3, 0, -1)
        cv2.circle(img, (right_eye_x, eye_y), 3, 0, -1)
        
        # Mouth (ellipse)
        mouth_y = center[1] + 10
        mouth_x = center[0] + np.random.randint(-2, 2)
        mouth_axes = (8 + np.random.randint(-2, 2), 3 + np.random.randint(-1, 1))
        cv2.ellipse(img, (mouth_x, mouth_y), mouth_axes, 0, 0, 180, 0, -1)
        
        X.append(img.flatten())
    
    return np.array(X)

# Generate face dataset
face_data = create_face_dataset(50)

print(f"Face dataset shape: {face_data.shape}")
print(f"Each image is {int(np.sqrt(face_data.shape[1]))}x{int(np.sqrt(face_data.shape[1]))} pixels")
```

```python
# Apply PCA
n_components = 50  # Reduce from 4096 to 50 dimensions
pca = PCA(n_components=n_components)
face_data_pca = pca.fit_transform(face_data)

# Reconstruct images from PCA components
face_data_reconstructed = pca.inverse_transform(face_data_pca)

print(f"Original data shape: {face_data.shape}")
print(f"PCA transformed shape: {face_data_pca.shape}")
print(f"Reconstructed data shape: {face_data_reconstructed.shape}")
print(f"Variance explained by {n_components} components: {np.sum(pca.explained_variance_ratio_):.2f}")
```

```python
# Visualize original vs reconstructed images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # Original image
    axes[0, i].imshow(face_data[i].reshape(64, 64), cmap='gray')
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    
    # Reconstructed image
    axes[1, i].imshow(face_data_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].set_title('Reconstructed')
    axes[1, i].axis('off')

plt.suptitle('PCA for Image Compression')
plt.tight_layout()
plt.show()
```

```python
# Visualize the first few principal components
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    component = pca.components_[i].reshape(64, 64)
    axes[i].imshow(component, cmap='RdBu')
    axes[i].set_title(f'PC {i+1}\n({pca.explained_variance_ratio_[i]:.2%} variance)')
    axes[i].axis('off')

plt.suptitle('Principal Components')
plt.tight_layout()
plt.show()
```

## Summary

In this module, we've covered:
1. Feature extraction techniques (HOG, LBP)
2. Image classification using Support Vector Machines
3. Image segmentation with K-Means clustering
4. Dimensionality reduction using Principal Component Analysis

These traditional machine learning approaches provide a strong foundation for understanding modern deep learning techniques. In the next module, we'll dive into deep learning and neural networks for computer vision tasks.