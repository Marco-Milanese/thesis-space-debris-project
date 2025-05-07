import torch
from PIL import Image
import os
from torchvision.transforms import ToTensor, ToPILImage
from Autoencoder import Autoencoder
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

import matplotlib.pyplot as plt

# Convert the output image to numpy
out_numpy = outImage[0].cpu().squeeze(0).detach().numpy()
to_pil_image = ToPILImage()

highResImage.show()
to_pil_image(randImage).show()
out = outImage[0].cpu().squeeze(0)
print("Min pixel value:", out.min().item())
print("Max pixel value:", out.max().item())
#out = out.clamp(0, 1)
to_pil_image(out).show()

# Display the image using matplotlib

plt.imshow(out_numpy, cmap='gray')
plt.axis('on')  # Turn off axis
plt.show()
