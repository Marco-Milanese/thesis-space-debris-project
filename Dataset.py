
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image


class SpaceDebrisDataset(Dataset):
    def __init__(self, csv_file, lowResDirectory, hiResDirectory):
        # Load the dataset from the CSV file
        self.csv = pd.read_csv(csv_file).drop(columns=['bboxes'])
        self.lowResDirectory = lowResDirectory
        self.hiResDirectory = hiResDirectory
    
    def __len__(self):
        # The length of the dataset is the number of rows in the CSV file
        return len(self.csv)
    
    def __getitem__(self, index):
        # Generate the paths for the hi/low resolution images
        lowResPath = os.path.join(self.lowResDirectory, str(self.csv.iloc[index, 0]) + '.jpg')
        hiResPath = os.path.join(self.hiResDirectory, str(self.csv.iloc[index, 0]) + '.jpg')
        # Load the images as tensors from the paths
        lowResImage = torchvision.transforms.ToTensor()(Image.open(lowResPath))
        hiResImage = torchvision.transforms.ToTensor()(Image.open(hiResPath))
        
        return lowResImage, hiResImage