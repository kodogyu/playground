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

# 오토인코더 모델 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [batch, 16, 14, 14]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [batch, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)                      # [batch, 64, 1, 1]
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),            # [batch, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [batch, 16, 14, 14]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # [batch, 1, 28, 28]
            nn.Sigmoid()  # Output을 0~1 사이로 제한
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x