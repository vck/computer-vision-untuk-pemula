#!/usr/bin/env python3
"""
Module 5: Advanced Deep Learning for Computer Vision
Sample Script: ResNet-inspired Architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

# Simple ResNet implementation
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        # Average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def make_layer(self, out_channels, blocks, stride):
        """Create a layer with residual blocks."""
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

def create_sample_data(batch_size=32):
    """Create sample image data."""
    # Create random images (batch_size, 3, 32, 32)
    images = torch.randn(batch_size, 3, 32, 32)
    # Create random labels
    labels = torch.randint(0, 10, (batch_size,))
    
    return images, labels

def train_model(model, num_epochs=5):
    """Train the model for a few epochs."""
    # Create sample data
    images, labels = create_sample_data(32)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    losses = []
    
    print("Training model...")
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return losses

def visualize_resnet_architecture():
    """Visualize the ResNet architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.text(0.5, 0.9, 'Simple ResNet Architecture', ha='center', va='center', fontsize=16, transform=ax.transAxes)
    
    # Layers
    layers = [
        (0.1, 0.8, 'Input\n(3×32×32)'),
        (0.25, 0.8, 'Conv1\n(64 filters)'),
        (0.4, 0.8, 'ResBlock\n(64 filters)'),
        (0.55, 0.8, 'ResBlock\n(128 filters)'),
        (0.7, 0.8, 'ResBlock\n(256 filters)'),
        (0.85, 0.8, 'FC\n(10 classes)')
    ]
    
    # Draw layers
    for i, (x, y, text) in enumerate(layers):
        ax.text(x, y, text, ha='center', va='center', fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        if i > 0:
            ax.annotate('', xy=(x, y), xytext=(layers[i-1][0], layers[i-1][1]), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Residual block detail
    ax.text(0.4, 0.6, 'Residual Block:\nConv→BN→ReLU\nConv→BN\nSkip Connection\nReLU', 
            ha='center', va='center', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Skip connection
    ax.annotate('', xy=(0.45, 0.65), xytext=(0.35, 0.65), 
                arrowprops=dict(arrowstyle="->", lw=2, color='red'))
    ax.text(0.4, 0.7, 'Skip', ha='center', va='center', fontsize=8, transform=ax.transAxes, color='red')
    
    ax.set_title('ResNet Architecture Overview')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate ResNet architecture."""
    print("Module 5: Advanced Deep Learning for Computer Vision")
    print("=================================================")
    
    # Visualize ResNet architecture
    print("\nVisualizing ResNet architecture...")
    visualize_resnet_architecture()
    
    # Initialize model
    print("\nInitializing ResNet model...")
    model = SimpleResNet(num_classes=10)
    print(f"Model architecture:\n{model}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nTraining model with sample data...")
    losses = train_model(model, num_epochs=5)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nResNet demonstration completed!")

if __name__ == "__main__":
    main()