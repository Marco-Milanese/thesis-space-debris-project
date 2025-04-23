from Dataset import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Autoencoder import Autoencoder


# Select the gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Training and Validation datasets
TrainingSet = SpaceDebrisDataset('./data/trainTest.csv', './data/lowResTrain1ch', './data/Train1ch')


# testing with lowRes for the hiRes so that the images are the same size
#TrainingSet = SpaceDebrisDataset('./data/trainTest.csv', './data/lowResTrain', './data/lowResTrain')
print(len(TrainingSet))

ValSet = SpaceDebrisDataset('./data/valTest.csv', './data/lowResVal1ch', './data/Val1ch')
print(len(ValSet))

#batch_size = 200 # batch size chosen just because it divides well into 20k
#epochs = 1500 # number of epochs specified in the paper

#test
batch_size = 1 
epochs = 1


# Load the datasets into dataloaders
trainDataLoader = DataLoader(TrainingSet, batch_size, shuffle=True)
valDataLoader = DataLoader(ValSet, batch_size, shuffle=True)


model = Autoencoder().to(device)
    
# Defining the loss function and optimizer
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99)) # Adam optimizer with the specified betas and learning rate

for epoch in range(epochs):
    # Training phase
    model.train()
    for data in trainDataLoader:
        print('Training...')
        lowResImages, hiResImages = data 
        # Pushing the data to the available device
        lowResImages = lowResImages.to(device) 
        hiResImages = hiResImages.to(device)

        # Forward pass
        outputs = model(lowResImages)
        loss = lossFunction(outputs, hiResImages)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
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

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {valLoss/len(valDataLoader):.4f}")