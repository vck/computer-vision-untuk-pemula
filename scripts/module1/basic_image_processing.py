#!/usr/bin/env python3
"""
Module 1: Foundations of Computer Vision
Sample Script: Basic Image Processing Operations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_display_image(image_path):
    """Load and display an image."""
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Could not load image from {image_path}")
        # Create a sample image if file doesn't exist
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
        cv2.circle(img, (200, 200), 50, (0, 255, 0), -1)
    
    # Display image
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    return img

def convert_color_spaces(img):
    """Convert image between different color spaces."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (BGR)')
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale')
    axes[1].axis('off')
    
    axes[2].imshow(hsv[:, :, 0], cmap='hsv')
    axes[2].set_title('HSV Hue Channel')
    axes[2].axis('off')
    
    plt.show()
    
    return gray, hsv

def apply_filters(img):
    """Apply various filters to the image."""
    # Blur filters
    blur_avg = cv2.blur(img, (5, 5))
    blur_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    blur_median = cv2.medianBlur(img, 5)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(blur_avg, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Average Blur')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(blur_gaussian, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Gaussian Blur')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Median Blur')
    axes[1, 1].axis('off')
    
    plt.show()

def detect_edges(img):
    """Perform edge detection on the image."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Grayscale')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sobel_combined, cmap='gray')
    axes[0, 1].set_title('Sobel Edges')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Canny Edges')
    axes[1, 0].axis('off')
    
    # Show histogram
    axes[1, 1].hist(gray.ravel(), 256, [0, 256])
    axes[1, 1].set_title('Grayscale Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.show()

def main():
    """Main function to demonstrate basic image processing operations."""
    print("Module 1: Foundations of Computer Vision")
    print("======================================")
    
    # Load and display image
    img = load_and_display_image('../data/sample.jpg')
    
    # Convert color spaces
    print("\nConverting color spaces...")
    gray, hsv = convert_color_spaces(img)
    
    # Apply filters
    print("\nApplying filters...")
    apply_filters(img)
    
    # Detect edges
    print("\nDetecting edges...")
    detect_edges(img)
    
    print("\nBasic image processing operations completed!")

if __name__ == "__main__":
    main()