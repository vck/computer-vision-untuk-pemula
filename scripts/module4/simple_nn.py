#!/usr/bin/env python3
"""
Module 4: Introduction to Deep Learning
Sample Script: Simple Neural Network with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def create_dataset():
    """Create a classification dataset."""
    # Create a more complex classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=10, n_clusters_per_class=1, random_state=42)
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def train_model(model, X_train, y_train, X_test, y_test, num_epochs=100):
    """Train the neural network model."""
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Calculate test accuracy
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test).sum().item() / len(y_test)
                test_accuracies.append(accuracy)
            model.train()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')
    
    return train_losses, test_accuracies

def plot_training_results(train_losses, test_accuracies):
    """Plot training loss and test accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot test accuracy
    ax2.plot(range(10, 101, 10), test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate a simple neural network."""
    print("Module 4: Introduction to Deep Learning")
    print("=====================================")
    
    # Create dataset
    print("\nCreating dataset...")
    X_train, y_train, X_test, y_test = create_dataset()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize model
    print("\nInitializing neural network...")
    model = SimpleNN(input_size=20, hidden_size=64, num_classes=2)
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("\nTraining model...")
    train_losses, test_accuracies = train_model(model, X_train, y_train, X_test, y_test, num_epochs=100)
    
    # Evaluate final model
    print("\nEvaluating final model...")
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_outputs = model(X_train)
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_accuracy = (train_predicted == y_train).sum().item() / len(y_train)
        
        # Test accuracy
        test_outputs = model(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test).sum().item() / len(y_test)
    
    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # Plot results
    print("\nPlotting training results...")
    plot_training_results(train_losses, test_accuracies)
    
    print("\nSimple neural network demonstration completed!")

if __name__ == "__main__":
    main()