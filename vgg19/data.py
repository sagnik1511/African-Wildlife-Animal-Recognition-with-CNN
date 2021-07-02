"""
All the folders of images are into a single folder named as 'images'

the current working directory looks like

-images/
    -buffalo/
        -001.jpg
        -001.txt
        .
        .
        .
        
    -elephant/
        -001.jpg
        -001.txt
        .
        .
        .
        
    -rhino/
        -001.jpg
        -001.txt
        .
        .
        .
        
    -zebra/
        -001.jpg
        -001.txt
        .
        .
        .
        
        
So, basically all the label files are going to be
deleted and all the folder will only contain jpg
images.


Codes are created by ©️ Sagnik Roy 2021 .
Do share your feedbacks.

"""

# Libraries
import os
import pandas as pd
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Source
img_dir = "/content/images/"

# Hyperparameters

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
SEED = 42    # To find similar result the seed has been taken

      
""" 
Now the files will be deleted as the directory tree will convert into

-images/
    -buffalo/
        -001.jpg
        .
        .
        .
        
    -elephant/
        -001.jpg
        .
        .
        .
        
    -rhino/
        -001.jpg
        .
        .
        .
        
    -zebra/
        -001.jpg
        .
        .
        .
        

"""

label_files = glob(f"{img_dir}*/*.txt")

for filepath in label_files:
  os.remove(filepath)

"""

After converting the directory as we need, the data
is fed into the data generator to augment the images

Their is an 85-15 train-validation split


"""


# Augmentation
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    horizontal_flip = True ,
    zoom_range = 0.1,
    validation_split = 0.15
)


# Train Data Generator
train_data = train_datagen.flow_from_directory(
    '/content/images',
    target_size = (IMG_HEIGHT , IMG_WIDTH),
    seed = SEED,
    subset = 'training'
)

#Validation Data Generator
validation_data = train_datagen.flow_from_directory(
    '/content/images',
    target_size = (IMG_HEIGHT , IMG_WIDTH),
    seed = SEED,
    subset = 'validation'
)


# Classes of the Data
classes = pd.DataFrame.from_dict(train_data.class_indices , orient = 'index').T.columns


# Testing
for patches , labels in train_data:
  assert patches.shape == (BATCH_SIZE * 2 , IMG_HEIGHT , IMG_WIDTH , 3)
  assert labels.shape == (BATCH_SIZE * 2 , len(classes))
  break
    
print('Data Processed Successfully............')
