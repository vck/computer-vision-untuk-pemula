#!/usr/bin/env python3
"""
Module 6: Specialized Topics and Applications
Sample Script: Motion Detection in Video
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_video(filename="sample_video.avi", duration=10):
    """Create a sample video with moving objects."""
    # Video parameters
    width, height = 640, 480
    fps = 30
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Create background
    background = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some static elements
    cv2.rectangle(background, (50, 50), (200, 200), (0, 100, 0), -1)
    cv2.circle(background, (500, 300), 80, (100, 0, 0), -1)
    
    # Generate frames
    for i in range(total_frames):
        # Start with background
        frame = background.copy()
        
        # Add moving objects after a few frames
        if i > 30:
            # Moving circle
            x = 320 + int(200 * np.sin(i * 0.1))
            y = 240 + int(100 * np.cos(i * 0.1))
            cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)
            
            # Moving rectangle
            x = 100 + int(300 * np.sin(i * 0.05))
            y = 100 + int(200 * np.cos(i * 0.05))
            cv2.rectangle(frame, (x-20, y-20), (x+20, y+20), (255, 255, 0), -1)
        
        # Write frame
        out.write(frame)
    
    # Release everything
    out.release()
    print(f"Sample video created: {filename}")

def motion_detection(video_path):
    """Perform motion detection on a video."""
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Create background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()
    
    # Store results for visualization
    frames = []
    masks = []
    contours_list = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        
        # Threshold the mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on frame
        result_frame = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Store results (limit to first 10 frames for visualization)
        if frame_count < 10:
            frames.append(frame)
            masks.append(thresh)
            contours_list.append(result_frame)
        
        frame_count += 1
        
        # Limit processing for demo
        if frame_count > 100:
            break
    
    cap.release()
    
    return frames, masks, contours_list

def visualize_motion_detection(frames, masks, contours_list):
    """Visualize motion detection results."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i in range(3):
        # Original frame
        axes[i, 0].imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Original Frame {i+1}')
        axes[i, 0].axis('off')
        
        # Foreground mask
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title(f'Foreground Mask {i+1}')
        axes[i, 1].axis('off')
        
        # Contours
        axes[i, 2].imshow(cv2.cvtColor(contours_list[i], cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f'Motion Detection {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate motion detection."""
    print("Module 6: Specialized Topics and Applications")
    print("===========================================")
    
    # Create sample video
    print("\nCreating sample video...")
    video_filename = "sample_video.avi"
    create_sample_video(video_filename, duration=5)
    
    # Perform motion detection
    print("\nPerforming motion detection...")
    frames, masks, contours_list = motion_detection(video_filename)
    
    # Visualize results
    print("\nVisualizing motion detection results...")
    visualize_motion_detection(frames, masks, contours_list)
    
    # Clean up
    import os
    if os.path.exists(video_filename):
        os.remove(video_filename)
        print(f"\nCleaned up {video_filename}")
    
    print("\nMotion detection demonstration completed!")

if __name__ == "__main__":
    main()