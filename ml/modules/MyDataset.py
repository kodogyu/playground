import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from mlxtend.data import loadlocal_mnist

class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform):
        self.images, self.labels = loadlocal_mnist(images_path, labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28)
        label = torch.tensor(self.labels[idx], dtype=int)

        if self.transform:
            image = self.transform(image)

        label = F.one_hot(label, 10).to(float)

        return image, label