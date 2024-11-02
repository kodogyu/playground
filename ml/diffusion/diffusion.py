import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# 가우시안 노이즈 추가
def add_noise(imgs, t, noise_schedule):
    noise = torch.randn_like(imgs)
    noisy_imgs = imgs * (1 - noise_schedule[t]) + noise * noise_schedule[t]
    return noisy_imgs

# U-Net과 같은 간단한 네트워크
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    # 데이터 로드
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # 모델, 손실함수, 옵티마이저 정의
    model = SimpleUNet().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 노이즈 스케줄 설정
    timesteps = 1000
    noise_schedule = np.linspace(1e-4, 0.02, timesteps)

    # checkpoint 경로 설정
    checkpoint_path = "checkpoints"

    # 학습 루프
    epochs = 5
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to('cuda')

            # 랜덤 타임스텝 선택
            t = torch.randint(0, timesteps, (1,)).item()

            # 데이터에 노이즈 추가
            noisy_data = add_noise(data, t, noise_schedule).to('cuda')

            # 모델을 통해 노이즈 복원
            optimizer.zero_grad()
            output = model(noisy_data)

            # 손실 계산 및 역전파
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

        # checkpoint 저장
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict()
            }
        torch.save(checkpoint, f"{checkpoint_path}/diffusion_epoch{epoch+1}")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")


    print("Training complete!")
