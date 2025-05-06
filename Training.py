from Dataset import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Autoencoder import Autoencoder
from torchvision.transforms import ToPILImage
import os
from datetime import datetime


# Select the gpu if available
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print("TPU available")
except:
    print("No TPU detected, using GPU if available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Load the Training and Validation datasets
TrainingSet = SpaceDebrisDataset('./data/train.csv', './data/LowResTrain1ch', './data/Train1ch')
TrainLen = len(TrainingSet)
print(f'Training set size: {TrainLen}')

ValSet = SpaceDebrisDataset('./data/val.csv', './data/LowResVal1ch', './data/Val1ch')
ValLen = len(ValSet)
print(f'Validation set size: {ValLen}')

batch_size = 128 # batch size chosen as 2^7, good for Colab GPU memory
epochs = 10

#test
#batch_size = 5
#epochs = 1

# Load the datasets into dataloaders
trainDataLoader = DataLoader(TrainingSet, batch_size, shuffle=True)
valDataLoader = DataLoader(ValSet, batch_size, shuffle=True)

model = Autoencoder().to(device)

# Load the pre-trained model if available
if os.path.exists('./Autoencoder.pth'):
    print("Loading pre-trained model")
    model.load_state_dict(torch.load('./Autoencoder.pth', map_location="cpu"))
    model.to(device)
else:
    print('No pre-trained model')
    
# Defining the loss function and optimizer
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99)) # Adam optimizer with the specified betas and learning rate
trainLossSum = 0.0

for epoch in range(epochs):
    # Training phase
    batchNumber = 0
    model.train()
    for data in trainDataLoader:
        batchNumber += 1
        totalBatch = len(trainDataLoader)
        print(f"Epoch {epoch+1}/{epochs} Batch {batchNumber}/{totalBatch}")
        #print('Training')
        lowResImages, hiResImages = data 
        # Pushing the data to the available device
        lowResImages = lowResImages.to(device) 
        hiResImages = hiResImages.to(device)

        # Forward pass
        #print("Forward pass")
        outputs = model(lowResImages)
        outMin = outputs.min()
        trainLoss = lossFunction(outputs, hiResImages) + abs(outMin) 
        trainLossSum = trainLossSum + trainLoss
        # Backward pass and optimization
        #print("Backward pass")
        optimizer.zero_grad()
        trainLoss.backward()
        #print("Optimizer step")
        optimizer.step()
        #xm.optimizer_step(optimizer)

    # Validation phase to prevent overfitting
    model.eval()
    with torch.no_grad():
        valLossSum = 0.0
        for data in valDataLoader:
            lowResImages, hiResImages = data
            lowResImages = lowResImages.to(device)
            hiResImages = hiResImages.to(device)

            # Forward pass
            outputs = model(lowResImages)
            outMin = outputs.min()
            valLoss = lossFunction(outputs, hiResImages) + abs(outMin) 
            valLossSum = valLossSum + valLoss
    
    torch.save(model.state_dict(), 'Autoencoder.pth')

    # Auto saving of the model during Colab training
    os.system('git add Autoencoder.pth')
    current_time = datetime.now().strftime("%A, %d %B %Y")
    os.system(f'git commit Autoencoder.pth -m "AutoSave of the model during training - Epoch {epoch+1}/{epochs}, Batch {batchNumber}/{totalBatch} - {current_time}"')
    os.system('git push -u origin main')

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {trainLossSum/len(trainDataLoader):.4f}, Validation Loss: {valLossSum/len(valDataLoader):.4f}")

# Display the output of the last validation batch
"""
to_pil_image = ToPILImage()

finalInput = to_pil_image(hiResImages[0].cpu().squeeze(0))  
finalGenerated = to_pil_image(outputs[0].cpu().squeeze(0)) 

finalGenerated.show()
finalInput.show()
"""