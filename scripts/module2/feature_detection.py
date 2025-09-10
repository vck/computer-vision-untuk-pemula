#!/usr/bin/env python3
"""
Module 2: Classical Computer Vision Techniques
Sample Script: Feature Detection and Matching
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_images():
    """Create sample images for feature detection."""
    # Image 1: Rectangle with circle
    img1 = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (250, 250), 255, -1)
    cv2.circle(img1, (150, 150), 60, 128, -1)
    
    # Image 2: Same shapes but shifted and rotated
    img2 = np.zeros((300, 300), dtype=np.uint8)
    # Create rectangle at different position
    rect_pts = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.float32)
    rotation_matrix = cv2.getRotationMatrix2D((150, 150), 30, 1)
    rotated_rect = cv2.transform(rect_pts, rotation_matrix)
    cv2.fillPoly(img2, [np.int32(rotated_rect)], 255)
    
    # Create circle at different position
    cv2.circle(img2, (180, 120), 60, 128, -1)
    
    return img1, img2

def detect_features_orb(img):
    """Detect features using ORB detector."""
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    kp, des = orb.detectAndCompute(img, None)
    
    # Draw keypoints
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    
    return img_kp, kp, des

def match_features(des1, des2):
    """Match features between two images."""
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

def find_homography_and_warp(img1, img2, kp1, kp2, matches):
    """Find homography and warp one image to align with another."""
    if len(matches) > 4:
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # Warp img1 to img2's perspective
            result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
            result[0:img2.shape[0], 0:img2.shape[1]] = img2
            return result
    
    return None

def main():
    """Main function to demonstrate feature detection and matching."""
    print("Module 2: Classical Computer Vision Techniques")
    print("============================================")
    
    # Create sample images
    img1, img2 = create_sample_images()
    
    # Display original images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    
    plt.show()
    
    # Detect features
    print("\nDetecting ORB features...")
    img1_kp, kp1, des1 = detect_features_orb(img1)
    img2_kp, kp2, des2 = detect_features_orb(img2)
    
    # Display images with keypoints
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1_kp, cmap='gray')
    axes[0].set_title(f'Image 1 with Keypoints ({len(kp1)} detected)')
    axes[0].axis('off')
    
    axes[1].imshow(img2_kp, cmap='gray')
    axes[1].set_title(f'Image 2 with Keypoints ({len(kp2)} detected)')
    axes[1].axis('off')
    
    plt.show()
    
    # Match features
    print("\nMatching features...")
    matches = match_features(des1, des2)
    
    # Draw matches
    if len(matches) > 0:
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(img_matches)
        plt.title(f'Top 10 Feature Matches (Total: {len(matches)})')
        plt.axis('off')
        plt.show()
        
        print(f"Number of matches: {len(matches)}")
        print(f"Best match distance: {matches[0].distance}")
        print(f"Worst match distance: {matches[-1].distance}")
        
        # Try to create panorama
        print("\nCreating panorama...")
        panorama = find_homography_and_warp(img1, img2, kp1, kp2, matches)
        
        if panorama is not None:
            plt.figure(figsize=(15, 8))
            plt.imshow(panorama, cmap='gray')
            plt.title('Simple Panorama')
            plt.axis('off')
            plt.show()
        else:
            print("Could not create panorama - insufficient matches")
    else:
        print("No matches found")
    
    print("\nFeature detection and matching completed!")

if __name__ == "__main__":
    main()