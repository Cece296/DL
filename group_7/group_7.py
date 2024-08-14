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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #


input_dim = (256,256)
channel_dim = 3             # 1 for grayscale, 3 for RGB

class group_7(nn.Module):
    def _init_(self):
        super(group_7,self)._init_()
        #layers
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(channel_dim, 16, 3, 1, 1)    # 3x3 kernel, stride 1, padding 1

        # Fully connected layers
        self.fc1 = nn.Linear(64*64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        # Pooling layer
        self.maxpool = nn.MaxPool2d(2, 2)    # 2x2 kernel, stride 2
        self.softmax = nn.Softmax(dim=1)    # Softmax layer

        # Activation function
        self.relu = nn.ReLU()


    def forward(self,x):
        #layers
        x = torch.flatten(x, 1)
        x = self.conv1(x)

        

        return x



