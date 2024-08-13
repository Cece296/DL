import cv2
import numpy as np
import glob
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, img_size, class_names, path=None, transformations=None, num_per_class: int = -1):
        self.img_size = img_size
        self.path = path
        self.num_per_class = num_per_class
        self.class_names = class_names
        self.transforms = transformations
        self.data = []
        self.labels = []

        if path:
            self.readImages()

        self.standard_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def readImages(self):
        for id, class_name in self.class_names.items():
            print(f'Loading images from class: {id} : {class_name}')
            img_path = glob.glob(f'{self.path}{class_name}/*.jpg')
            if self.num_per_class > 0:
                img_path = img_path[:self.num_per_class]
            self.labels.extend([id] * len(img_path))
            for filename in img_path:
                img = Image.open(filename).convert('RGB')
                img = img.resize(self.img_size)
                self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transforms:
            img = self.transforms(img)
        else:
            img = self.standard_transforms(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label

def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
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


def test(model, test_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
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


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, patience=5):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = test(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'model5.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load the best model
    model.load_state_dict(torch.load('model5.pth'))

    # Plot learning and loss curves
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.savefig('learning_curves5.png')
    plt.show()

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
img_size = (224, 224)
batch_size = 16
val_split = 0.8
num_per_class = 100
num_epochs = 20

# Define transformations for training and validation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
train_path = "./data/train/"
test_path = "./data/train/"

class_names = [name[len(train_path):] for name in glob.glob(f'{train_path}*')]
class_names = dict(zip(range(len(class_names)), class_names))

full_dataset = CustomDataset(img_size=img_size, path=train_path, class_names=class_names, transformations=train_transform,
                             num_per_class=num_per_class)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.transforms = val_transform

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define model
resnet_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(full_dataset.class_names))
resnet_model.to(device)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

# Train the model with early stopping and learning curves
trained_model = train_model(model=resnet_model, train_loader=train_data_loader, val_loader=val_data_loader,
                            loss_fn=loss_fn, optimizer=optimizer, num_epochs=num_epochs, device=device)

# Evaluate final test performance
dataset_test = CustomDataset(img_size=img_size, path=test_path, class_names=class_names, transformations=val_transform,
                             num_per_class=num_per_class)
test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

test_loss, test_accuracy = test(trained_model, test_data_loader, loss_fn, device)
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")