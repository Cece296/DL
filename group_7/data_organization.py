"""
This code is made to asssign the images to the train, validation and test folders.
We have chosen to assign 70% to train and 15% to validation and test, respectively. 

# VI KØRER FUCKING 70-20-10 VED VORES

We assign the images randomly as we cannot be sure if the images is ordered in a way.
This could be that all the first photos are from a hospital with a new scanner and the rest from 
an older more blurry scanner. As we have no ID on the people from the images we assume 
that the images match a unique person. 

We have chosen to have the most data to train our model. The exact distribution is made from
considerations made on researching the web and from what we have discussed in the lectures.

This contribution was made by CAFOL19, INPHI16
"""
import os
from pathlib import Path
from os import walk
import shutil
import random
from PIL import Image

"""
Function that splits the dataset into training, validation and testing folders.
"""
def move_files(my_list, name:str):
    random.shuffle(my_list) # randomly rearrange the list

    for index, image_name in enumerate(my_list):
        # This is used to move the images from the main folder into the subfolders
        if index <= 770:
            shutil.move(data_dir / image_name, data_dir / 'training' / name / image_name)
        elif 770 < index <= 935:
            shutil.move(data_dir / image_name, data_dir / 'testing' / name / image_name)
        else:
            shutil.move(data_dir / image_name, data_dir / 'validation' / name / image_name)

"""
Function that prints out the minimum, maximum and average width and height of the images. This was used for decision making on the reshaping size.
"""
def image_sizes(my_list, name:str):
    height = []
    width = []

    for image in my_list:
        # This is used in order to get the different sizes of the images, so that we can get the avg., min and max height and width
        img = Image.open(data_dir/image)
        img_h, img_w = img.size
        height.append(img_h)
        width.append(img_w)
        img.close()
        
    print(f"{name}: \nThe average height is {sum(height)/1100}, and the average width is {sum(width)/1100}, the minimum is {min(height), min(width)}, an maximum is {max(height), max(width)}.")


"""
Retrieve the directory path.
"""    
src = Path(__file__).parent
data_dir = src / 'data'

"""
Saving all the image names, having normal and pneumonia in seperate list, such that
we can make sure to get the same amount of respectively sick and healthy individuals in 
each folder
"""
normal = [name for name in os.listdir(data_dir) if name.rsplit('_',1)[-1] == 'normal.jpg']
pneumonia = [name for name in os.listdir(data_dir) if name.rsplit('_',1)[-1] == 'pneumonia.jpg']

"""
Retrieves the average, minimum and maximum sizes of the images.
"""
image_sizes(normal, "normal")
image_sizes(pneumonia, "pneumonia")

"""
Moves the dataset into the correct folders.
"""
move_files(normal, "normal")
move_files(pneumonia, "pneumonia")