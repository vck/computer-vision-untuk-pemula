import torch.nn as F
import torch.nn as nn
import torch
from torch import flatten
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class LeNet(F.Module):
    def __init__(self, num_channels, classes):

        # call parent constructor
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layer

        self.conv1 = F.Conv2d(
            in_channels=num_channels, out_channels=20, kernel_size=(5, 5)
        )

        self.relu1 = F.ReLU()
        self.maxpool1 = F.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layer
        self.conv2 = F.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))

        self.relu2 = F.ReLU()
        self.maxpool2 = F.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first set of FC => RELU layers
        self.fc1 = F.Linear(in_features=800, out_features=500)
        self.relu3 = F.ReLU()

        # initialize the softmax classifier
        self.fc2 = F.Linear(in_features=500, out_features=classes)
        self.logSoftmax = F.LogSoftmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the prev layer thorugh the second

        # set of CONV => RELU => POOL layers

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the prev layer and pass it through FC => RELU layers

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # prediction

        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


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


train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())


loaders = {
    "train": torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=True, num_workers=5
    ),
    "test": torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=True, num_workers=5
    ),
}

model = LeNet(num_channels=3, classes=20)
cnn = CNN()
loss_func = nn.CrossEntropyLoss()
from torch import optim

optimizer = optim.Adam(cnn.parameters(), lr=0.01)

print(cnn)

from torch.autograd import Variable

num_epochs = 10


def train(num_epochs, cnn, loaders):

    cnn.train()

    # Train the model
    total_step = len(loaders["train"])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders["train"]):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

    torch.save(cnn, "cnn.pt")

train(num_epochs, cnn, loaders)


def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders["test"]:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            print(
                "Test Accuracy of the model on the 10000 test images: %.2f" % accuracy
            )


test()
