import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input channels=1 (grayscale), output channels=32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # output channels=64
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

        # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output channels=128

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # Adjust the size depending on the input size and pooling layers
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv layer + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer + ReLU + Pooling
        x = self.pool(F.relu(self.conv3(x)))  # Third conv layer + ReLU + Pooling
        x = x.view(-1, 128 * 6 * 6)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # First fully connected layer + ReLU
        x = self.dropout(x)  # Dropout layer
        x = self.fc2(x)  # Output layer
        return x


class SimpleVGG(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleVGG, self).__init__()

        # First conv block
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second conv block
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third conv block
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth conv block
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fifth conv block
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))  # Use AdaptiveAvgPool to ensure the output size is always 1x1

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional blocks
        x = self.pool1(F.relu(self.conv1_1(x)))
        x = self.pool1(F.relu(self.conv1_2(x)))

        x = self.pool2(F.relu(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2(x)))

        x = self.pool3(F.relu(self.conv3_1(x)))
        x = self.pool3(F.relu(self.conv3_2(x)))
        x = self.pool3(F.relu(self.conv3_3(x)))

        x = self.pool4(F.relu(self.conv4_1(x)))
        x = self.pool4(F.relu(self.conv4_2(x)))
        x = self.pool4(F.relu(self.conv4_3(x)))

        x = self.pool5(F.relu(self.conv5_1(x)))
        x = self.pool5(F.relu(self.conv5_2(x)))
        x = self.pool5(F.relu(self.conv5_3(x)))

        # Ensure x has a valid size before flattening
        x = x.view(-1, 512 * 1 * 1)  # Flatten the tensor

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
