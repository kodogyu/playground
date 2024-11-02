import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from diffusion import SimpleUNet, add_noise


def main(device) :
    # 데이터 로드
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    # model 정의
    model = SimpleUNet().to(device)
    model = model.eval()

    # checkpoint 로드
    checkpoint = torch.load("checkpoints/diffusion_epoch5")
    model.load_state_dict(checkpoint["model_state_dict"])

    # 가우시안 노이즈 생성
    # gaussian_noise = torch.randn((1,28,28)).to(device)
    data, _ = next(iter(train_loader))
    data = data.to(device)

    # 랜덤 타임스텝 선택
    timesteps = 1000
    # noise_schedule = np.linspace(1e-4, 0.02, timesteps)
    noise_schedule = np.linspace(0, 1, timesteps)
    # t = torch.randint(0, timesteps, (1,)).item()
    t = 999

    # 데이터에 노이즈 추가
    noisy_data = add_noise(data, t, noise_schedule).to(device)


    # model 통과
    with torch.no_grad():
        # output = model(gaussian_noise)
        output = model(noisy_data)

    # print("gaussian_nosie.shape:", gaussian_noise.shape)
    print("noisy_data.shape:", noisy_data.shape)
    print("output.shape:", output.shape)

    # 결과 시각화
    # plt.subplot(1,2,1).imshow(gaussian_noise.squeeze().detach().cpu().numpy(), cmap='gray')
    # plt.title("gaussian noise")
    plt.subplot(1,2,1).imshow(noisy_data.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.title("noisy_data")
    plt.subplot(1,2,2).imshow(output.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.title("model output")
    plt.show()

if __name__ == "__main__":
    device = 'cuda'
    print(f"device: {device}")

    main(device=device)