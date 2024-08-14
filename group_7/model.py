# ---------------------------------------------------------------------------- #
# Group members:
# Cecilie   cevei21@student.sdu.dk
# Josefine  ander22@student.sdu.dk
# Mathias   krist21@student.sdu.dk
# Simone    sileb18@student.sdu.dk
# Stinne    stzac22@student.sdu.dk
# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader , Dataset
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# ---------------------------------------------------------------------------- #

# ------------------------------- Using Device ------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------- Custom Dataset ------------------------------ #
class ChestXRayDataset(Dataset):
    def _init_(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
        self.labels = [0 if 'NORMAL' in img else 1 for img in self.images]

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------------------- Model Class ------------------------------- #
input_dim = (256,256)                                       # Image dimension
channel_dim = 3                                             # 1 for grayscale, 3 for RGB

class group_7(nn.Module):
    def _init_(self):
        super(group_7,self)._init_()
        # Layers
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(channel_dim, 16, 3, 1, 1)    # 3x3 kernel, stride 1, padding 1

        # Fully connected layers
        self.fc1 = nn.Linear(64*64, 1024)                   # 64x64 image dimension
        self.fc2 = nn.Linear(1024, 512)                     # 1024 input dimension
        self.fc3 = nn.Linear(512, 10)                       # 512 input dimension

        # Pooling layer
        self.maxpool = nn.MaxPool2d(2, 2)                   # 2x2 kernel, stride 2
        self.softmax = nn.Softmax(dim=1)                    # Softmax layer

        # Activation function
        self.relu = nn.ReLU()


    def forward(self,x):
        # Layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x
    
# --------------------- Define function to train network --------------------- #

def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in group_7(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# ---------------------- Define function to test network ---------------------- #

def test(model, test_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in group_7(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# ------------------------------- Training Loop ------------------------------- #

def training_loop(model, train_loader, test_loader, loss_fn, optimizer, device, epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, loss_fn, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies


# -------------------------------- Dataloader -------------------------------- #

# Define paths to training, validation, and testing directories
train_dir = '/path/to/data/training'
val_dir = '/path/to/data/validation'
test_dir = '/path/to/data/testing'

# Create datasets
train_dataset = ChestXRayDataset(img_dir=train_dir, transform=train_transform)
val_dataset = ChestXRayDataset(img_dir=val_dir, transform=val_test_transform)
test_dataset = ChestXRayDataset(img_dir=test_dir, transform=val_test_transform)

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)


# Data augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), #This can help the model become invariant to changes in the patient's position when the chest X-rays are taken.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # We have used this because it is the standard normalization for the pre-trained models
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1) # This can help the model become invariant to changes in the color of the X-ray images.
])


# Create an instance of your custom dataset
train_dataset = ChestXRayDataset(train_dir, transform=transform)
val_dataset = ChestXRayDataset(val_dir, transform=transform)
test_dataset = ChestXRayDataset(test_dir, transform=transform)

dataset = CustomDataset(data_path='group_7\data', transform=transform)

# ------------------------------- Plot Resoults ------------------------------ #

def plot_results(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()