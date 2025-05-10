from PIL import Image
import os
import pandas as pd
from math import floor

trainLen = len(pd.read_csv("data/train.csv"))
valLen = len(pd.read_csv("data/val.csv"))
testLen = 5000

def to1Channel(len, LRinputDir, HRinputDir, LRoutputDir, HRoutputDir):
    for i in range(len):
        LRimage = Image.open(os.path.join(LRinputDir, str(i) + '.jpg'))
        HRimage = Image.open(os.path.join(HRinputDir, str(i) + '.jpg'))
        LRimage1ch = LRimage.convert('L')
        HRimage1ch = HRimage.convert('L')
        LRimage1ch.save(os.path.join(LRoutputDir, str(i) + '.jpg'))
        HRimage1ch.save(os.path.join(HRoutputDir, str(i) + '.jpg'))
        print(f"Converted {i} images to 1 channel from {HRinputDir}")

to1Channel(1, "data/LowResTrain", "data/train", "data/LowResTrain1ch", "data/Train1ch")
to1Channel(1, "data/LowResVal", "data/val", "data/LowResVal1ch", "data/Val1ch")
to1Channel(1, "data/LowResTest", "data/test", "data/LowResTest1ch", "data/Test1ch")