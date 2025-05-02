import torch
from PIL import Image
import os
from torchvision.transforms import ToTensor, ToPILImage
from Autoencoder import Autoencoder
import pandas as pd
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load('./Autoencoder.pth', map_location=device))

numTestImages = 20000

randNum = random.randint(0, numTestImages - 1)
randImagePath = os.path.join('./data/LowResTrain1ch', f"{randNum}.jpg")
highResImage = Image.open(os.path.join('./data/Train1ch', f"{randNum}.jpg"))

randImage = ToTensor()(Image.open(randImagePath))
dlImage = randImage.unsqueeze(0).to(device) # Add a batch dimension
outImage = model(dlImage)

to_pil_image = ToPILImage()

highResImage.show()
to_pil_image(outImage[0].cpu().squeeze(0)).show()