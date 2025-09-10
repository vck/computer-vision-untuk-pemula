#!/usr/bin/env python3
"""
Module 3: Machine Learning for Computer Vision
Sample Script: Image Classification with SVM
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

def create_shape_dataset(n_samples=200):
    """Create a dataset of shapes (circles and rectangles)."""
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

def visualize_samples(X, y, n_samples=6):
    """Visualize sample images from the dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Reshape HOG features back to approximate images for visualization
    for i in range(n_samples):
        ax = axes[i // 3, i % 3]
        # Create a simple visualization
        img = np.zeros((64, 64))
        if y[i] == 0:
            cv2.circle(img, (32, 32), 20, 255, -1)
            title = "Circle"
        else:
            cv2.rectangle(img, (15, 15), (49, 49), 255, -1)
            title = "Rectangle"
        
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_svm_classifier(X_train, y_train):
    """Train an SVM classifier."""
    # Create and train SVM classifier
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    
    return clf

def evaluate_classifier(clf, X_test, y_test):
    """Evaluate the classifier."""
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Circle', 'Rectangle']))
    
    return y_pred

def main():
    """Main function to demonstrate image classification with SVM."""
    print("Module 3: Machine Learning for Computer Vision")
    print("=============================================")
    
    # Create dataset
    print("\nCreating shape dataset...")
    X, y = create_shape_dataset(200)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Visualize samples
    print("\nVisualizing sample images...")
    visualize_samples(X, y)
    
    # Train classifier
    print("\nTraining SVM classifier...")
    clf = train_svm_classifier(X_train, y_train)
    
    # Evaluate classifier
    print("\nEvaluating classifier...")
    y_pred = evaluate_classifier(clf, X_test, y_test)
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(5, len(y_test))):
        actual = "Circle" if y_test[i] == 0 else "Rectangle"
        predicted = "Circle" if y_pred[i] == 0 else "Rectangle"
        print(f"  Sample {i+1}: Actual={actual}, Predicted={predicted}")
    
    print("\nImage classification with SVM completed!")

if __name__ == "__main__":
    main()