import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import random
import numpy as np
from mlxtend.data import loadlocal_mnist
from tqdm import tqdm
import wandb

import sys
sys.path.insert(0, "../")
from modules.MyDataset import MNISTDataset
from modules.MyNetwork import MyCNNClassifier

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# Test function
def get_test_acc(model, dataloader, criterion):
    model.eval()

    with torch.no_grad():
        correct_cnt = 0
        samples_cnt = 0
        loss_acc = 0
        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.to(device)

            answer = torch.argmax(label, dim=1)

            # inference
            pred = model(image)
            pred_label = torch.argmax(pred, dim=1)
            
            # accumulate results
            correct = (pred_label == answer)
            correct_cnt += correct.sum().item()
            samples_cnt += label.shape[0]

            # loss
            loss = criterion(pred, label)
            loss_acc += loss.item()

        # accuracy
        test_acc = correct_cnt/samples_cnt
        test_avg_loss = loss_acc/len(dataloader)

    return test_acc, test_avg_loss

# Train function
def train(config):
    # random seed
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Dataset
    transform = torchvision.transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNISTDataset("../datasets/MNIST/raw/train-images-idx3-ubyte", "../datasets/MNIST/raw/train-labels-idx1-ubyte", transform)
    test_dataset = MNISTDataset("../datasets/MNIST/raw/t10k-images-idx3-ubyte", "../datasets/MNIST/raw/t10k-labels-idx1-ubyte", transform)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Network
    model = MyCNNClassifier().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        # train accuracy & loss
        train_acc, train_loss = train_epoch(model=model, dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, config=config)
        wandb.log({"Training accuracy": train_acc, "epoch": epoch+1})
        wandb.log({"Training loss": train_loss, "epoch": epoch+1})

        # test accuracy & loss
        test_acc, test_loss = get_test_acc(model=model, dataloader=test_dataloader, criterion=criterion)
        wandb.log({"Test accuracy": test_acc, "epoch": epoch+1})
        wandb.log({"Test loss": test_loss, "epoch": epoch+1})

        # save checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, f"checkpoints/ckpt_epoch{epoch}.pth")
    return test_acc

def train_epoch(model, dataloader, optimizer, criterion, config):
    model.train()

    correct_cnt = 0
    samples_cnt = 0
    loss_acc = 0
    tepoch = tqdm(dataloader)
    for idx, (image, label) in enumerate(tepoch):
        image = image.to(device)
        label = label.to(device)
        
        # forward
        optimizer.zero_grad()
        pred = model(image)

        # backward
        loss = criterion(pred, label)
        loss.backward()

        # update
        optimizer.step()

        # accuracy
        answer = torch.argmax(label, dim=1).detach().cpu().numpy()
        pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
        correct_cnt += (answer == pred_label).sum()
        samples_cnt += label.shape[0]

        # logging
        tepoch.set_postfix({"loss": loss.item()})
        loss_acc += loss.item()

    train_acc = correct_cnt / samples_cnt
    train_avg_loss = loss_acc/len(dataloader)
    return train_acc, train_avg_loss
