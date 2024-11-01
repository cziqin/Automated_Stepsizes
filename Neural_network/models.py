import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(torch.nn.Module):
    def __init__(self, classes=10):
        super().__init__()

        in_channels = 3
        kernel_size = 5
        in1, in2, in3 = 512, 84, 84

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=kernel_size, padding=1)  # , padding=2)
        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=kernel_size, padding=1)

        self.dropout = nn.Dropout2d(0.25)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.dense1 = nn.Linear(in1, in2)
        self.dense2 = nn.Linear(in3, classes)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(self.act(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.act(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.act(self.batch_norm3(self.conv3(x))))

        x = x.view(-1, 512)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout2d(0.3)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout2d(0.3)
        self.pool1 = nn.MaxPool2d(2)  # Reduces the size to 16x16

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout2d(0)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU()
        self.drop4 = nn.Dropout2d(0)
        self.pool2 = nn.MaxPool2d(2)  # Reduces the size to 8x8

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = nn.Linear(128, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool1(self.drop2(self.act2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.act3(self.bn3(self.conv3(x))))
        x = self.pool2(self.drop4(self.act4(self.bn4(self.conv4(x)))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the layer
        x = self.fc(x)
        return x



if __name__ == "__main__":
    res =CIFAR10CNN()
    t = torch.randn((10, 3, 224, 224))
    t = res(t)
    print(t.shape)
