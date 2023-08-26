import torch
import torch.nn as F
import torch.nn as nn
from torch import flatten
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

# Model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization

# Hyperparameters

model_config = {
    "learning_rate": 0.01,
    "num_ephocs": 10,
    "num_classes": 20,
    "num_channels": 3
}

# Set Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data

train_data = datasets.MNIST(root="data",train=True,transform=ToTensor(),download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=5)

test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=5)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])

model.train()

# Train the model

total_step = len(train_loader)

for epoch in range(model_config['num_ephocs']):
    for batch_idx, (data, targets) in enumerate(train_loader):

        # gives batch data, normalize x when iterate train_loader
        
        # Get data to cuda
        data = data.to(device)
        targets = targets.to(device)

        # Get data to correct shape
        #data = data.reshape(data.shape[0], -1)

        # forward

        scores = model(data)
        print(scores)
        loss = loss_func(scores[0], targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update weight based on computed gradient
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, model_config['num_ephocs'], batch_idx + 1, total_step, loss.item()
                )
            )
"""

# Model evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        test_output, last_layer = model(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        print(
            "Test Accuracy of the model on the 10000 test images: %.2f" % accuracy
        )
"""