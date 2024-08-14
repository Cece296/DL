# This file is made to divide the images into a train, a validation and a test folder.
# As well as separated by whether they are healthy or have pneumonia.
# They are split in parts of 70% train, 20% validation and 10% test.
# This is chosen based on it being a default balanced split in general.

# The images are randomly assigned to the different folders.

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import os
from pathlib import Path
import shutil
import random

# ---------------------------------------------------------------------------- #
#Retrieve the directory path.
pathSource = Path(__file__).parent
dataDirec = pathSource / 'data'     # The data folder has to be in the same directory as this file


# Create the train, validation and testing folders with normal and pneumonia subfolders
os.makedirs(dataDirec / 'training' / 'normal', exist_ok=True)
os.makedirs(dataDirec / 'training' / 'pneumonia', exist_ok=True)
os.makedirs(dataDirec / 'validation' / 'normal', exist_ok=True)
os.makedirs(dataDirec / 'validation' / 'pneumonia', exist_ok=True)
os.makedirs(dataDirec / 'testing' / 'normal', exist_ok=True)
os.makedirs(dataDirec / 'testing' / 'pneumonia', exist_ok=True)


# Get the names of the images in the data folder and split them into normal and pneumonia images
# This is done by checking the last part of the name, which is either normal or pneumonia
normalImages = [name for name in os.listdir(dataDirec) if name.rsplit('_',1)[-1] == 'normal.jpg']       # looks at the last part of the name if it's "normal"
pneumoniaImages = [name for name in os.listdir(dataDirec) if name.rsplit('_',1)[-1] == 'pneumonia.jpg'] # looks at the last part of the name if it's "pneumonia"


# function that moves the files into the subfolders
def moveFiles(list, name:str):
    random.shuffle(list)         # shuffles the list of images to make it random
    for index, imageName in enumerate(list):
        # This is used to move the images from the main folder into the subfolders
        if index <= 770:            # 70% of the data is used for training
            shutil.move(dataDirec / imageName, dataDirec / 'training' / name / imageName)   # moves the first 770 images to the training folder
        elif 770 < index <= 990:    # 20% of the data is used for validation
            shutil.move(dataDirec / imageName, dataDirec / 'validation' / name / imageName) # moves the next 220 images to the validation folder
        else:                       # 10% of the data is used for testing
            shutil.move(dataDirec / imageName, dataDirec / 'testing' / name / imageName)    # moves the last 110 images to the testing folder


# moves the images into the correct folders
moveFiles(normalImages, "normal")
moveFiles(pneumoniaImages, "pneumonia")