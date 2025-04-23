import torch
from Autoencoder import Autoencoder
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image


inputImage = ToTensor()(Image.open('./data/LowResTrain1ch/0.jpg'))
outputImage = ToTensor()(Image.open('./data/Train1ch/0.jpg'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99)) 

model.train()
inputImage.to(device)
outputImage.to(device)

outputs = model(inputImage)
loss = lossFunction(outputs, outputImage)

