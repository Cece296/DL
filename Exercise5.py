# split the data/train folder that has 16 class folders inside 
# and each of them should be split into 80% training and 20% validation

import os
import shutil
import random

# Path to the data/train folder
data_path = 'data/train'
# Path to the data folder
data_folder = 'data'
# Path to the train folder
train_folder = 'train'
# Path to the validation folder
validation_folder = 'validation'

# Create the train and validation folders
if not os.path.exists(data_folder):
    os.mkdir(data_folder)
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
if not os.path.exists(validation_folder):
    os.mkdir(validation_folder)

# Get the list of class folders
class_folders = os.listdir(data_path)

# Split the data into training and validation sets
for class_folder in class_folders:
    # Path to the class folder
    class_path = os.path.join(data_path, class_folder)
    # Path to the train class folder
    train_class_path = os.path.join(train_folder, class_folder)
    # Path to the validation class folder
    validation_class_path = os.path.join(validation_folder, class_folder)
    # Create the train and validation class folders
    if not os.path.exists(train_class_path):
        os.mkdir(train_class_path)
    if not os.path.exists(validation_class_path):
        os.mkdir(validation_class_path)
    # Get the list of images in the class folder
    images = os.listdir(class_path)
    # Shuffle the images
    random.shuffle(images)
    # Split the images into training and validation sets
    num_images = len(images)
    num_train_images = int(0.8 * num_images)
    train_images = images[:num_train_images]
    validation_images = images[num_train_images:]
    # Copy the training images to the train class folder
    for image in train_images:
        image_path = os.path.join(class_path, image)
        train_image_path = os.path.join(train_class_path, image)
        shutil.copy(image_path, train_image_path)
    # Copy the validation images to the validation class folder
    for image in validation_images:
        image_path = os.path.join(class_path, image)
        validation_image_path = os.path.join(validation_class_path, image)
        shutil.copy(image_path, validation_image_path)
    print(f'Split {class_folder} into {num_train_images} training images and {num_images - num_train_images} validation images')

print('Done')



