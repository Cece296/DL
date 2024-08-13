import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
import os


# Create a CustomDataset class for the dataset using PyTorch’s Dataset. This class should be responsible for:
# i.Reading of images
# ii.Preprocessing of images
# iii.Image augmentation
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        for label in os.listdir(root_dir):
            for image in os.listdir(os.path.join(root_dir, label)):
                self.data.append(os.path.join(root_dir, label, image))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = plt.imread(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    

# Create a transform object that will be used to preprocess the images in the dataset. The transform object should be responsible for:
# i.Resizing the images to 224x224
# ii.Converting the images to PyTorch tensors
# iii.Normalizing the images using the mean and standard deviation of the ImageNet dataset
# iv.Randomly flipping the images horizontally
# v.Randomly rotating the images by 10 degrees
# vi.Randomly applying affine transformations on the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1))
])

# Create a CustomDataset object for the dataset using the CustomDataset class and the transform object.
trainset = CustomDataset(root_dir='data', transform=transform)

# Create a DataLoader object for the dataset using the CustomDataset object. Set the batch_size to 4 and shuffle to True.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# Load the ResNet-50 model from torchvision.models and modify the final fully connected layer (classifier) of the ResNet-50 to match the number of classes in your dataset (16 classes).
resnet_model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

#Modify the final fully connected layer (classifier) of the ResNet-50 to match the number of classes in your dataset (16 classes).
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 16)

#Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

#Train the model
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(trainset)
        epoch_acc = running_corrects.double() / len(trainset)
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return model

train_model(resnet_model, criterion, optimizer, num_epochs=25)

#Save the trained model
torch.save(resnet_model.state_dict(), 'resnet_model.pth')


