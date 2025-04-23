import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
from ast import literal_eval

# the r prefix is used to tell python to interpret the string as a raw string, so the 
#backslash isn't treated as an escape character
train_csv_path = r'C:\Users\marco\Desktop\spaceDebrisDetection\data\train.csv'
val_csv_path = r'C:\Users\marco\Desktop\spaceDebrisDetection\data\val.csv'
#creating a dataframe from the csv file (pandas)
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
#printing the first 5 rows of the dataframe
print(train_df.head())
print(val_df.head())


def show_images(images, num):
    images_to_show = np.random.choice(images, num) #randomly select num images from the list of images

    for image_id in images_to_show:
        #genera un path per ogni immagine
        image_path = os.path.join(r'C:\Users\marco\Desktop\spaceDebrisDetection\data\train', f"{image_id}.jpg")
        image = Image.open(image_path)
        bboxes = literal_eval(train_df.loc[train_df['ImageID'] == image_id]['bboxes'].values[0])
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle([bbox[0], bbox[2], bbox[1], bbox[3]], width=1)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.show()

show_images(train_df['ImageID'].unique(), 1)
