import torch
import torch.nn as nn
import torch.nn.functional as F

class WeatherCNNClassifier(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        super(WeatherCNNClassifier, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Convolutional Block 4 (extra depth)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  # After 4 times max pooling (256x16x16)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Output logits for multi-class classification

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # -> 128x128

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # -> 64x64

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # -> 32x32

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # -> 16x16

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)  # logits
        return x


if __name__ == "__main__":
    model = WeatherCNNClassifier()
    print(model)
    x = torch.randn(4, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)  # Expected: [4, 5]

