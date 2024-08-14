# ---------------------------------------------------------------------------- #
# Group members:
# Cecilie cevei21@student.sdu.dk
# Josefine ander22@student.sdu.dk
# Mathias krist21@student.sdu.dk
# Simone sileb18@student.sdu.dk
# Stinne stzac22@student.sdu.dk
#
# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------- #

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(input_dim)])


input_dim = (h,w)
channel_dim = c

class group_7(nn.Module):
    def _init_(self):
        super(group_7,self)._init_()
        #layers
    def forward(self,x):
        #layers
        return x

""" Task 1 The Dataset
    All images are x-ray images taken in the chest region of children aged 1-5 from the Guangzhou Women and Childrens Medical Center. 
    There are 1100 images of healthy humans, as well as 1100 images of people with pneumonia.  
    Keep in mind, that the image format is jpeg, and there are 3 color channels. 
    You need to organize the data into directories as shown on Figure 1. 
    You need to determine the training/validation/testing split yourselves but need to justify your split choice.
"""

# Load the dataset
data_path = 'data'
train_data = datasets.ImageFolder(root=data_path, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Split dataset into "normal" and "pneumonia"
normal_data = [data for data in train_data if data[1] == 0]
pneumonia_data = [data for data in train_data if data[1] == 1]


# Split the data into training, validation and testing
train_size = int(0.7 * len(train_data))
val_size = int(0.15 * len(train_data))
test_size = len(train_data) - train_size - val_size

train_data = torch.utils.data.random_split(train_data, [train_size, val_size, test_size])
val_data = torch.utils.data.random_split(train_data, [train_size, val_size, test_size])
test_data = torch.utils.data.random_split(train_data, [train_size, val_size, test_size])

# Justify the split choice
# We chose to split the data into 70% training, 15% validation and 15% testing. This is a common split choice in machine learning.
# The training set is used to train the model, the validation set is used to tune the hyperparameters and the testing set is used to evaluate the model.
# We chose to use 70% of the data for training because we want the model to learn from a large amount of data. We chose to use 15% of the data for validation
# because we want to tune the hyperparameters on a smaller dataset. We chose to use 15% of the data for testing because we want to evaluate the model on a smaller dataset.

""" Task 2 The Model """
# Define the model
model = group_7()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer

# Train the model
def train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item()))

# Test the model
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy: {} %'.format(100 * correct / total))

# Hyperparameters
learning_rate = 0.001
epochs = 5

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train(model, train_loader, criterion, optimizer, epochs)

# Test the model
test(model, test_loader)

""" Task 3 The Evaluation """
# Plot the training loss
plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()



