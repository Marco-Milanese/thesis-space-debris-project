import os
from PIL import Image
import pandas as pd
import numpy as np
import tkinter as tk
from math import floor
from tkinter import simpledialog


train_csv_path = r'C:\Users\marco\Desktop\spaceDebrisDetection\data\train.csv'
train_df = pd.read_csv(train_csv_path)

def downsample_images(images, num, scalingFactor, mode=None):
    images_to_show = np.random.choice(images, num) #randomly select num images from the list of images

    for image_id in images_to_show:
        #generates a path for each image
        image_path = os.path.join(r'C:\Users\marco\Desktop\spaceDebrisDetection\data\train', f"{image_id}.jpg")
        hiResImage = Image.open(image_path)
        #scales the image size by the scaling
        w, h = hiResImage.size
        newW = floor(w // scalingFactor)
        newH = floor(h // scalingFactor)
        #downsample with Bicubic Interpolation
        lowResImage = hiResImage.resize((newW, newH), Image.BICUBIC)

        if mode == "merge":
            #generates a new image which is the combination of the low and high resolution images
            mergedW = w + newW
            mergedH = max(h, newH)

            mergedImage = Image.new('RGB', (mergedW, mergedH))
            mergedImage.paste(hiResImage)
            mergedImage.paste(lowResImage, (w, 0))

            mergedImage.show()
        else:
            #shows the high and low resolution images in separate windows
            hiResImage.show()
            lowResImage.show()


root = tk.Tk()
root.withdraw()  # Hide the main window
# Show a single input dialog for both inputs
class InputDialog(simpledialog.Dialog):
    def body(self, master):
        self.geometry("300x150")  # Set the initial size of the dialog window
        tk.Label(master, text="Scaling Factor:").grid(row=0)
        tk.Label(master, text="Mode (merge or separate):").grid(row=1)

        self.scale_entry = tk.Entry(master)
        self.mode_entry = tk.Entry(master)

        self.scale_entry.grid(row=0, column=1)
        self.mode_entry.grid(row=1, column=1)
        return self.scale_entry  # Focus on the first entry field

    def apply(self):
        self.result = (float(self.scale_entry.get()),self.mode_entry.get())

dialog = InputDialog(root, title="Input Parameters")
if dialog.result:
    scale, mode = dialog.result

downsample_images(train_df['ImageID'].unique(), 1, scale, mode)



