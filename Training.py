from Dataset import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Autoencoder import Autoencoder
from torchvision.transforms import ToPILImage
import os


# Select the gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Load the Training and Validation datasets
TrainingSet = SpaceDebrisDataset('./data/trainTest.csv', './data/LowResTrain1ch', './data/Train1ch')
print('Training set size: {len(TrainingSet)}')

ValSet = SpaceDebrisDataset('./data/valTest.csv', './data/LowResVal1ch', './data/Val1ch')
print('Validation set size: {len(ValSet)}')

batch_size = 264 # batch size chosen as 2^8 good for Colab GPU memory
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
    model.load_state_dict(torch.load('./Autoencoder.pth', map_location=device))
else:
    print('No pre-trained model')
    
# Defining the loss function and optimizer
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99)) # Adam optimizer with the specified betas and learning rate

for epoch in range(epochs):
    # Training phase
    batchNumber = 0
    model.train()
    for data in trainDataLoader:
        batchNumber += 1
        totalBatch = len(trainDataLoader)
        print(f"Batch {batchNumber}/{totalBatch}")
        print('Training')
        lowResImages, hiResImages = data 
        # Pushing the data to the available device
        lowResImages = lowResImages.to(device) 
        hiResImages = hiResImages.to(device)

        # Forward pass
        print("Forward pass")
        outputs = model(lowResImages)
        loss = lossFunction(outputs, hiResImages)

        # Backward pass and optimization
        print("Backward pass")
        optimizer.zero_grad()
        loss.backward()
        print("Optimizer step")
        optimizer.step()

    # Validation phase to prevent overfitting
    model.eval()
    with torch.no_grad():
        valLoss = 0.0
        for data in valDataLoader:
            lowResImages, hiResImages = data
            lowResImages = lowResImages.to(device)
            hiResImages = hiResImages.to(device)

            # Forward pass
            outputs = model(lowResImages)
            valLoss = valLoss + lossFunction(outputs, hiResImages).item()
    
    torch.save(model.state_dict(), 'Autoencoder.pth')

    # Auto saving of the model during Colab training
    os.system('git add Autoencoder.pth')
    os.system('git commit Autoencoder.pth -m "AutoSave of the model during training"')
    os.system('git push -u origin main')

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss:.4f}, Validation Loss: {valLoss/len(valDataLoader):.4f}")

# Display the output of the last validation batch
"""to_pil_image = ToPILImage()

finalInput = to_pil_image(hiResImages[0].cpu().squeeze(0))  
finalGenerated = to_pil_image(outputs[0].cpu().squeeze(0)) 

finalGenerated.show()
finalInput.show()"""
