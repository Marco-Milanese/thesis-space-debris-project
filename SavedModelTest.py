import torch
from PIL import Image
import os
from torchvision.transforms import ToTensor, ToPILImage
from Autoencoder import Autoencoder
import pandas as pd
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load('./Autoencoder.pth'))

numTestImages = 5000

randNum = random.randint(0, numTestImages - 1)
randImagePath = os.path.join('./data/LowResTest1ch', f"{randNum}.jpg")

randImage = ToTensor()(Image.open(randImagePath)).to(device)

outImage = model(randImage)

to_pil_image = ToPILImage()

Image.open(randImagePath).show()
to_pil_image(outImage[0].cpu().squeeze(0)).show()