import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import configuration
from config import convnet_config

# Use configuration from config.py
model_config = convnet_config

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=model_config['num_channels'],
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, model_config['num_classes'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load Data
    # Using "../data" to store data outside of src directory
    train_data = datasets.MNIST(root="../data", train=True, transform=ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)

    test_data = datasets.MNIST(root="../data", train=False, transform=ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0)

    # Initialize network
    model = CNN().to(device)

    # Loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    model.train()

    # Train the model
    total_step = len(train_loader)

    for epoch in range(model_config['ephocs']):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda
            data = data.to(device)
            targets = targets.to(device)

            # forward
            scores, _ = model(data)
            loss = loss_func(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update weight based on computed gradient
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, model_config['ephocs'], batch_idx + 1, total_step, loss.item()
                    )
                )

    # Model evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            test_output, _ = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / float(total)
        print(
            "Test Accuracy of the model on the 10000 test images: %.2f" % accuracy
        )

    # Save the model
    torch.save(model.state_dict(), '../models/cnn.pt')
    print("Model saved to ../models/cnn.pt")

if __name__ == '__main__':
    main()