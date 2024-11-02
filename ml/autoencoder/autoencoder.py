import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

# wandb 설정
project_name = "AutoEncoder_practice"
wandb.init(project=project_name)

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 1e-3
num_epochs = 1000

# 체크포인트 경로
checkpoint_path = "checkpoints/"
ckpt_save_interval = 5

# 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='../datasets', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 오토인코더 모델 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
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

# 모델, 손실 함수 및 옵티마이저 정의
model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    loss_acc = 0
    tepoch = tqdm(train_loader)
    for data in tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        img, _ = data
        img = img.cuda()
        
        # 순전파
        output = model(img)
        loss = criterion(output, img)
        loss_acc += loss.item()
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        tepoch.set_postfix(loss=loss.item())

    # wandb 로깅
    wandb.log({"Train Loss": loss_acc/len(train_loader), "epoch": epoch+1})

    # 체크포인트 저장
    if epoch % ckpt_save_interval == 0:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, checkpoint_path + f"autoencoder_epoch{epoch}")
        print("checkpoint saved")

    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 테스트: 일부 이미지에 대해 오토인코더의 재구성 결과 시각화
model.eval()
with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.cuda()
        output = model(img)
        break

# 원본 이미지와 재구성된 이미지 시각화
img = img.cpu().numpy()
output = output.cpu().numpy()

# 시각화 설정
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 이미지
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(img[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # 재구성된 이미지
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(output[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()
