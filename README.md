# Computer Vision Learning Path: From Classical to Modern

This repository contains a comprehensive learning path for computer vision, progressing from classical computer vision techniques to modern deep learning approaches. Each module builds upon previous knowledge while introducing new concepts and techniques.

## Learning Path Structure

### Module 0: Python 101 for Computer Vision (Prerequisites)
- Python basics (variables, data types, operators)
- Control structures (conditionals, loops)
- Functions and modules
- Data structures (lists, dictionaries, tuples)
- File handling
- Introduction to NumPy for numerical computing
- Introduction to Matplotlib for visualization

### Module 1: Foundations of Computer Vision
- Image representation and formats
- Basic image processing operations
- Color spaces (RGB, HSV, Grayscale)
- Image filtering and convolution
- Edge detection (Sobel, Canny)
- Histograms and thresholding

### Module 2: Classical Computer Vision Techniques
- Feature detection and description (SIFT, SURF, ORB)
- Feature matching
- Image transformations (Affine, Perspective)
- Image stitching and panorama creation
- Object detection with Haar cascades
- Camera calibration and stereo vision

### Module 3: Machine Learning for Computer Vision
- Traditional ML algorithms for image classification
- Feature extraction techniques (HOG, LBP)
- Support Vector Machines (SVM) for image classification
- K-Means clustering for image segmentation
- Principal Component Analysis (PCA) for dimensionality reduction

### Module 4: Introduction to Deep Learning
- Neural network fundamentals
- TensorFlow/Keras or PyTorch basics
- Building simple neural networks for image classification
- Convolutional Neural Networks (CNN) theory
- Transfer learning concepts

### Module 5: Advanced Deep Learning for Computer Vision
- Advanced CNN architectures (ResNet, Inception, EfficientNet)
- Object detection (R-CNN, YOLO, SSD)
- Semantic segmentation (U-Net, Mask R-CNN)
- Generative models (GANs, VAEs)
- Attention mechanisms and transformers

### Module 6: Specialized Topics and Applications
- Video analysis and tracking
- 3D computer vision and point clouds
- Medical image analysis
- Autonomous vehicles and robotics vision
- Augmented reality applications

## Repository Structure

```
.
├── learning_path.md                 # Comprehensive learning path blueprint
├── README.md                        # This file
├── docs/                            # Documentation
│   ├── learning_path.md             # Detailed learning path documentation
│   └── README.md                    # Documentation README
├── notebooks/                       # Jupyter notebooks for each module
│   ├── module0/
│   │   └── python_101.ipynb         # Python 101 for Computer Vision
│   ├── module1/
│   │   └── foundations.ipynb        # Foundations of Computer Vision
│   ├── module2/
│   │   └── classical_cv.ipynb       # Classical Computer Vision Techniques
│   ├── module3/
│   │   └── ml_cv.ipynb              # Machine Learning for Computer Vision
│   ├── module4/
│   │   └── dl_intro.ipynb           # Introduction to Deep Learning
│   ├── module5/
│   │   └── advanced_dl.ipynb        # Advanced Deep Learning for Computer Vision
│   └── module6/
│       └── specialized_applications.ipynb  # Specialized Applications
├── scripts/                         # Sample Python scripts for hands-on practice
│   ├── module0/
│   │   └── python_101.py            # Python 101 script version
│   ├── module1/
│   │   └── basic_image_processing.py # Basic image processing operations
│   ├── module2/
│   │   └── feature_detection.py     # Feature detection and matching
│   ├── module3/
│   │   └── svm_classification.py    # Image classification with SVM
│   ├── module4/
│   │   └── simple_nn.py             # Simple neural network with PyTorch
│   ├── module5/
│   │   └── resnet_demo.py           # ResNet-inspired architecture
│   └── module6/
│       └── motion_detection.py      # Motion detection in video
└── data/                            # Sample data (to be created by users)
```

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy matplotlib scikit-learn torch torchvision scikit-image
   ```
3. If you're new to Python, start with Module 0 (Python 101) to build foundational skills
4. Start with the [learning path blueprint](learning_path.md) to understand the complete curriculum
5. Work through each module in sequence, using the Jupyter notebooks and sample scripts

## Prerequisites

Before starting this curriculum, you should have:
- Basic Python programming skills (covered in Module 0)
- Understanding of linear algebra and calculus
- Familiarity with NumPy and Matplotlib

## Contributing

Feel free to contribute improvements, additional resources, or corrections by submitting a pull request.