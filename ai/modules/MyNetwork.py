import torch.nn as nn
import torch.nn.functional as F


class MyCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)  # (8,28,28)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # (16,14,14)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # (32,7,7)
        self.fc = nn.Linear(32, 10)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B,8,28,28)
        x = F.relu(self.max_pool(self.conv2(x)))  # (B,16,14,14)
        x = F.relu(self.max_pool(self.conv3(x)))  # (B,32,7,7)
        x = F.avg_pool2d(x, kernel_size=7)  # (B,32,1,1)
        x = x.flatten(start_dim=1)  # (B,32)
        logit = self.fc(x)

        return logit