# Module 5: Advanced Deep Learning for Computer Vision

## Learning Objectives
By the end of this module, you should be able to:
- Implement state-of-the-art CNN architectures
- Build object detection systems
- Perform semantic segmentation
- Create generative models
- Apply attention mechanisms in vision tasks

## Topics Covered
- Advanced CNN architectures (ResNet, Inception, EfficientNet)
- Object detection (R-CNN, YOLO, SSD)
- Semantic segmentation (U-Net, Mask R-CNN)
- Generative models (GANs, VAEs)
- Attention mechanisms and transformers

## 1. Advanced CNN Architectures

### ResNet (Residual Networks)

ResNets address the vanishing gradient problem through skip connections.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Residual block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# Visualize ResNet architecture concept
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.text(0.5, 0.9, 'ResNet Residual Block', ha='center', va='center', fontsize=16, transform=ax.transAxes)

# Input
ax.text(0.2, 0.8, 'Input', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Main path
ax.text(0.4, 0.7, 'Conv1
BN
ReLU', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax.text(0.6, 0.7, 'Conv2
BN', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# Skip connection
ax.text(0.4, 0.5, 'Shortcut
(Conv+BN if needed)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Output
ax.text(0.8, 0.6, 'Add
ReLU', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# Connections
ax.annotate('', xy=(0.3, 0.8), xytext=(0.2, 0.8), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.5, 0.7), xytext=(0.4, 0.7), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.7, 0.7), xytext=(0.6, 0.7), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.8, 0.6), xytext=(0.7, 0.7), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.5, 0.5), xytext=(0.3, 0.8), arrowprops=dict(arrowstyle="->", lw=1, color='red'))
ax.annotate('', xy=(0.8, 0.6), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="->", lw=1))

ax.text(0.5, 0.3, 'Output', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
ax.annotate('', xy=(0.5, 0.3), xytext=(0.8, 0.6), arrowprops=dict(arrowstyle="->", lw=1))

ax.set_title('ResNet Architecture Concept')
ax.axis('off')
plt.show()

print("ResNet Key Concepts:")
print("1. Skip connections help with gradient flow")
print("2. Enables training of very deep networks")
print("3. Addresses vanishing gradient problem")
```

## 2. Object Detection

### YOLO (You Only Look Once)

YOLO is a real-time object detection system.

```python
# Conceptual overview of YOLO
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.text(0.5, 0.95, 'YOLO Object Detection Concept', ha='center', va='center', fontsize=18, transform=ax.transAxes)

# Input image
ax.text(0.1, 0.8, 'Input Image
(448×448×3)', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Grid division
ax.text(0.3, 0.8, 'Divide into
S×S grid
(e.g., 7×7)', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# CNN backbone
ax.text(0.5, 0.8, 'CNN Backbone
(Feature extraction)', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Detection
ax.text(0.7, 0.8, 'Detection Layer
(S×S×(B×5+C))', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# Output
ax.text(0.9, 0.8, 'Bounding Boxes
+ Class Probabilities', ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink"))

# Connections
for i in range(4):
    ax.annotate('', xy=(0.3+0.2*i, 0.8), xytext=(0.1+0.2*i, 0.8), arrowprops=dict(arrowstyle="->", lw=2))

# Detailed explanation
ax.text(0.2, 0.6, 'Grid Cell Concept:', ha='center', va='center', fontsize=14, transform=ax.transAxes, weight='bold')
ax.text(0.2, 0.5, 'Each grid cell:
- Predicts B bounding boxes
- Confidence scores
- Class probabilities', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

ax.text(0.5, 0.6, 'Bounding Box:
- x, y (center coordinates)
- w, h (width, height)
- Confidence score', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

ax.text(0.8, 0.6, 'Class Prediction:
- Probability for
  each of C classes', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

# Loss function
ax.text(0.5, 0.3, 'YOLO Loss Function:
Localization + Confidence + Classification', 
        ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))

ax.set_title('YOLO Architecture Overview')
ax.axis('off')
plt.tight_layout()
plt.show()

print("YOLO Advantages:")
print("1. Real-time detection")
print("2. Global context consideration")
print("3. End-to-end training")
print("
YOLO Limitations:")
print("1. Struggles with small objects")
print("2. Limited to a fixed number of bounding boxes per grid cell")
```

## 3. Semantic Segmentation

### U-Net

U-Net is a convolutional network architecture for biomedical image segmentation.

```python
# Visualize U-Net architecture
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.text(0.5, 0.98, 'U-Net Architecture', ha='center', va='center', fontsize=18, transform=ax.transAxes)

# Contracting path (left side)
contracting_blocks = [
    (0.1, 0.8, 'Input
Image'),
    (0.2, 0.7, 'Conv+ReLU
Conv+ReLU
MaxPool'),
    (0.3, 0.6, 'Conv+ReLU
Conv+ReLU
MaxPool'),
    (0.4, 0.5, 'Conv+ReLU
Conv+ReLU
MaxPool'),
    (0.5, 0.4, 'Conv+ReLU
Conv+ReLU
MaxPool')
]

# Bottleneck
bottleneck = (0.6, 0.3, 'Conv+ReLU
Conv+ReLU')

# Expanding path (right side)
expanding_blocks = [
    (0.7, 0.4, 'UpConv
Concat
Conv+ReLU
Conv+ReLU'),
    (0.8, 0.5, 'UpConv
Concat
Conv+ReLU
Conv+ReLU'),
    (0.9, 0.6, 'UpConv
Concat
Conv+ReLU
Conv+ReLU'),
    (1.0, 0.7, 'UpConv
Concat
Conv+ReLU
Conv+ReLU')
]

# Output
output_block = (1.1, 0.8, '1x1 Conv
Segmentation
Map')

# Draw contracting path
for i, (x, y, text) in enumerate(contracting_blocks):
    ax.text(x, y, text, ha='center', va='center', fontsize=8, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    if i > 0:
        ax.annotate('', xy=(x, y), xytext=(contracting_blocks[i-1][0], contracting_blocks[i-1][1]), 
                    arrowprops=dict(arrowstyle="->", lw=1))

# Draw bottleneck
ax.text(bottleneck[0], bottleneck[1], bottleneck[2], ha='center', va='center', fontsize=8, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax.annotate('', xy=(bottleneck[0], bottleneck[1]), xytext=(contracting_blocks[-1][0], contracting_blocks[-1][1]), 
            arrowprops=dict(arrowstyle="->", lw=1))

# Draw expanding path
prev_x, prev_y = bottleneck[0], bottleneck[1]
for i, (x, y, text) in enumerate(expanding_blocks):
    ax.text(x, y, text, ha='center', va='center', fontsize=8, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.annotate('', xy=(x, y), xytext=(prev_x, prev_y), arrowprops=dict(arrowstyle="->", lw=1))
    
    # Skip connections
    if i < len(contracting_blocks) - 1:
        skip_x, skip_y = contracting_blocks[-(i+2)][0], contracting_blocks[-(i+2)][1]
        ax.annotate('', xy=(x-0.05, y+0.05), xytext=(skip_x+0.05, skip_y-0.05), 
                    arrowprops=dict(arrowstyle="->", lw=1, color='red', linestyle='dashed'))
        ax.text((x+skip_x)/2, (y+skip_y)/2, 'Skip
Connection', ha='center', va='center', 
                fontsize=6, transform=ax.transAxes, color='red')
    
    prev_x, prev_y = x, y

# Draw output
ax.text(output_block[0], output_block[1], output_block[2], ha='center', va='center', fontsize=8, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
ax.annotate('', xy=(output_block[0], output_block[1]), xytext=(expanding_blocks[-1][0], expanding_blocks[-1][1]), 
            arrowprops=dict(arrowstyle="->", lw=1))

# U-Net characteristics
ax.text(0.3, 0.1, 'Characteristics:', ha='center', va='center', fontsize=14, transform=ax.transAxes, weight='bold')
ax.text(0.3, 0.05, '• Contracting path captures context
• Expanding path enables precise localization
• Skip connections combine global and local information', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

ax.set_title('U-Net Architecture Diagram')
ax.axis('off')
plt.tight_layout()
plt.show()

print("U-Net Key Features:")
print("1. Symmetrical U-shaped architecture")
print("2. Skip connections between contracting and expanding paths")
print("3. Excellent for biomedical image segmentation")
print("4. Works well with limited training data")
```

## 4. Generative Models

### GANs (Generative Adversarial Networks)

GANs consist of two neural networks competing against each other.

```python
# GAN architecture visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.text(0.5, 0.95, 'Generative Adversarial Network (GAN)', ha='center', va='center', fontsize=18, transform=ax.transAxes)

# Generator
ax.text(0.2, 0.7, 'Generator

- Takes random noise
- Generates fake images
- Tries to fool Discriminator', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# Discriminator
ax.text(0.8, 0.7, 'Discriminator

- Takes real/fake images
- Outputs probability
- Tries to detect fakes', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# Random noise
ax.text(0.1, 0.5, 'Random Noise
(z ~ N(0,1))', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Generated image
ax.text(0.35, 0.5, 'Generated Image
G(z)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Real images
ax.text(0.65, 0.5, 'Real Images
from dataset', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Discriminator output
ax.text(0.9, 0.5, 'Probability
Real/Fake', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink"))

# Connections
ax.annotate('', xy=(0.25, 0.65), xytext=(0.15, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate('', xy=(0.45, 0.55), xytext=(0.25, 0.65), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate('', xy=(0.55, 0.55), xytext=(0.75, 0.65), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate('', xy=(0.85, 0.65), xytext=(0.55, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color='red'))
ax.annotate('', xy=(0.85, 0.65), xytext=(0.75, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color='blue'))
ax.annotate('', xy=(0.95, 0.55), xytext=(0.85, 0.65), arrowprops=dict(arrowstyle="->", lw=2))

# Loss functions
ax.text(0.3, 0.3, 'Generator Loss:
minimize -log(D(G(z)))', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

ax.text(0.7, 0.3, 'Discriminator Loss:
maximize log(D(x)) + log(1-D(G(z)))', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

# Minimax game
ax.text(0.5, 0.15, 'Minimax Game:
min_G max_D V(D,G) = E[log(D(x))] + E[log(1-D(G(z)))]', 
        ha='center', va='center', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))

ax.set_title('GAN Architecture and Training Process')
ax.axis('off')
plt.tight_layout()
plt.show()

print("GAN Training Dynamics:")
print("1. Generator tries to create realistic images")
print("2. Discriminator tries to distinguish real from fake")
print("3. Equilibrium: Generator produces realistic images that fool Discriminator")
```

## 5. Attention Mechanisms and Transformers

### Vision Transformers (ViT)

Vision Transformers apply the transformer architecture to image classification.

```python
# Vision Transformer concept
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.text(0.5, 0.98, 'Vision Transformer (ViT) Concept', ha='center', va='center', fontsize=18, transform=ax.transAxes)

# Input image
ax.text(0.1, 0.8, 'Input Image
(224×224×3)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Patch splitting
ax.text(0.25, 0.8, 'Split into Patches
(16×16×3 each)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# Linear projection
ax.text(0.4, 0.8, 'Linear Projection
(Patch Embeddings)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Position embeddings
ax.text(0.55, 0.8, 'Add Position
Embeddings', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# Class token
ax.text(0.7, 0.8, 'Add Class Token', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink"))

# Transformer encoder
ax.text(0.85, 0.8, 'Transformer
Encoder (L blocks)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))

# MLP head
ax.text(0.85, 0.6, 'MLP Head
(Classification)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

# Connections
for i in range(5):
    ax.annotate('', xy=(0.25+0.15*i, 0.8), xytext=(0.1+0.15*i, 0.8), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate('', xy=(0.85, 0.7), xytext=(0.85, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate('', xy=(0.85, 0.6), xytext=(0.85, 0.7), arrowprops=dict(arrowstyle="->", lw=2))

# Transformer encoder details
ax.text(0.5, 0.5, 'Transformer Encoder Block:', ha='center', va='center', fontsize=14, transform=ax.transAxes, weight='bold')

# Multi-head attention
ax.text(0.3, 0.4, 'Multi-Head
Self-Attention', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Layer norm
ax.text(0.4, 0.4, 'Layer
Norm', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# MLP
ax.text(0.6, 0.4, 'MLP
(2 layers)', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Layer norm
ax.text(0.7, 0.4, 'Layer
Norm', ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# Connections
ax.annotate('', xy=(0.35, 0.4), xytext=(0.3, 0.4), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.5, 0.4), xytext=(0.4, 0.4), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.65, 0.4), xytext=(0.6, 0.4), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.7, 0.4), xytext=(0.65, 0.4), arrowprops=dict(arrowstyle="->", lw=1))

# Skip connections
ax.annotate('', xy=(0.4, 0.45), xytext=(0.3, 0.45), arrowprops=dict(arrowstyle="->", lw=1, color='red'))
ax.text(0.35, 0.47, 'Skip', ha='center', va='center', fontsize=8, transform=ax.transAxes, color='red')

ax.annotate('', xy=(0.7, 0.45), xytext=(0.6, 0.45), arrowprops=dict(arrowstyle="->", lw=1, color='red'))
ax.text(0.65, 0.47, 'Skip', ha='center', va='center', fontsize=8, transform=ax.transAxes, color='red')

# Attention mechanism detail
ax.text(0.2, 0.2, 'Self-Attention:
Q = XW_Q
K = XW_K
V = XW_V
Attention(Q,K,V) = softmax(QK^T/√d_k)V', 
        ha='center', va='center', fontsize=9, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

# Benefits
ax.text(0.7, 0.2, 'ViT Benefits:
- Global attention
- Sequence modeling
- Transfer learning
- Scalability', 
        ha='center', va='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))

ax.set_title('Vision Transformer Architecture')
ax.axis('off')
plt.tight_layout()
plt.show()

print("Vision Transformer Key Concepts:")
print("1. Treats images as sequences of patches")
print("2. Uses self-attention to model global relationships")
print("3. Competitive with CNNs when trained on large datasets")
print("4. Better transfer learning capabilities")
```

## Summary

In this module, we've explored:
1. Advanced CNN architectures like ResNet with skip connections
2. Object detection systems including YOLO
3. Semantic segmentation with U-Net
4. Generative models like GANs
5. Attention mechanisms and Vision Transformers

These advanced techniques represent the current state-of-the-art in computer vision and form the foundation for many modern applications. In the final module, we'll explore specialized applications and cutting-edge research directions in computer vision.