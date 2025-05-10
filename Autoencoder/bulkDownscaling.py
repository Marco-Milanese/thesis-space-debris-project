import os
from PIL import Image
import pandas as pd
import numpy as np
from math import floor


train_csv_path = r'C:\Users\marco\Desktop\spaceDebrisDetection\data\train.csv'
train_df = pd.read_csv(train_csv_path)

scalingFactor = 2

#c = 0
for image_id in train_df['ImageID'].unique():
#while c < 5000: 
    
    #generates a path for each image
    image_path = os.path.join(r'C:\Users\marco\Desktop\spaceDebrisDetection\data\train', f"{image_id}.jpg")
    hiResImage = Image.open(image_path)
    #scales the image size by the scaling
    w, h = hiResImage.size
    newW = floor(w // scalingFactor)
    newH = floor(h // scalingFactor)
    #downsample with Bicubic Interpolation
    lowResImage = hiResImage.resize((newW, newH), Image.BICUBIC)
    lowResImage.save(os.path.join(r'C:\Users\marco\Desktop\spaceDebrisDetection\data\lowResTrain', f"{image_id}.jpg"))
    #c += 1
    print(f"Image {image_id} downsampled")
    