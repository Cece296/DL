# This file is made to divide the images into a train, a validation and a test folder.
# As well as separated by whether they are healthy or have pneumonia.
# They are split in parts of 70% train, 20% validation and 10% test.
# This is chosen based on it being a default balanced split in general.

# The images are randomly assigned to the different folders.

# This was primarily made by

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import os
from pathlib import Path
from os import walk
import shutil
import random
from PIL import Image

# ---------------------------------------------------------------------------- #

# function that moves the files into the subfolders
def move_files(my_list, name:str):
    random.shuffle(my_list) # shuffles the list of images to make it random
    for index, image_name in enumerate(my_list):
        # This is used to move the images from the main folder into the subfolders
        if index <= 770: # 70% of the data is used for training
            shutil.move(data_dir / image_name, data_dir / 'training' / name / image_name) 
        elif 770 < index <= 990: # 20% of the data is used for validation
            shutil.move(data_dir / image_name, data_dir / 'validation' / name / image_name)
        else: # 10% of the data is used for testing
            shutil.move(data_dir / image_name, data_dir / 'test' / name / image_name)


#Retrieve the directory path.
src = Path(__file__).parent
data_dir = src / 'data' # The data folder is in the same directory as the script


# Create the train, validation and testing folders with normal and pneumonia subfolders
os.makedirs(data_dir / 'training' / 'normal', exist_ok=True)
os.makedirs(data_dir / 'training' / 'pneumonia', exist_ok=True)
os.makedirs(data_dir / 'validation' / 'normal', exist_ok=True)
os.makedirs(data_dir / 'validation' / 'pneumonia', exist_ok=True)
os.makedirs(data_dir / 'testing' / 'normal', exist_ok=True)
os.makedirs(data_dir / 'testing' / 'pneumonia', exist_ok=True)


# Get the names of the images in the data folder and split them into normal and pneumonia images
# This is done by checking the last part of the name, which is either normal or pneumonia
normal_images = [name for name in os.listdir(data_dir) if name.rsplit('_',1)[-1] == 'normal.jpg'] # looks at the last part of the name if it's "normal"
pneumonia_images = [name for name in os.listdir(data_dir) if name.rsplit('_',1)[-1] == 'pneumonia.jpg'] # looks at the last part of the name if it's "pneumonia"


# moves the images into the correct folders
move_files(normal_images, "normal")
move_files(pneumonia_images, "pneumonia")