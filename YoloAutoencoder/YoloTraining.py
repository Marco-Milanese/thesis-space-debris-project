from YoloDataLoader import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from YoloAutoencoderV2 import Autoencoder
import os
from datetime import datetime
from YoloLoss import YoloLoss
from lossLogger import logLosses


# Select the GPU if available
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

# Load the datasets into dataloaders
trainDataLoader = DataLoader(TrainingSet, batch_size, shuffle=True)
valDataLoader = DataLoader(ValSet, batch_size, shuffle=True)
model = Autoencoder().to(device)

# Load the pre-trained model if available
if os.path.exists('./YoloAutoencoderV2.pth'):
    print("Loading pre-trained model")
    model.load_state_dict(torch.load('./YoloAutoencoderV2.pth', map_location="cpu"))
    model.to(device)
else:
    print('No pre-trained model')
    
# Defining the loss function and optimizer
YoloLoss = YoloLoss()
MseLoss = nn.MSELoss()
lambdaSR = 5
# Adam optimizer with the specified betas and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

for epoch in range(epochs):
    # Training phase
    batchNumber = 0
    trainLossSum = 0.0
    valLossSum = 0.0
    model.train()
    for data in trainDataLoader:
        batchNumber += 1
        totalBatch = len(trainDataLoader)
        print(f"Epoch {epoch+1}/{epochs} Batch {batchNumber}/{totalBatch}")

        # Pushing the data to the available device
        lowResImages, hiResImages, bboxes= data 
        lowResImages = lowResImages.to(device) 
        hiResImages = hiResImages.to(device)
        bboxes = bboxes.to(device)
        # Forward pass
        outputs = model(lowResImages)
        trainLoss = YoloLoss(outputs[1], bboxes) + lambdaSR * MseLoss(outputs[0], hiResImages)
        trainLossSum = trainLossSum + trainLoss

        # Backward pass and optimization
        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

    # Validation phase to prevent overfitting
    model.eval()
    with torch.no_grad():
        for data in valDataLoader:
            lowResImages, hiResImages, bboxes= data
            lowResImages = lowResImages.to(device)
            hiResImages = hiResImages.to(device)
            bboxes = bboxes.to(device)
            # Forward pass
            outputs = model(lowResImages)
            valLoss = YoloLoss(outputs[1], bboxes) + lambdaSR * MseLoss(outputs[0], hiResImages)
            valLossSum = valLossSum + valLoss

    trainLosses = {
    "total": trainLossSum.item() / len(trainDataLoader),
    "detection": YoloLoss(outputs[1], bboxes).item() / len(trainDataLoader),
    "reconstruction": (lambdaSR * MseLoss(outputs[0], hiResImages)).item() / len(trainDataLoader),
    }
    valLosses = {
    "total": valLossSum.item() / len(valDataLoader),
    "detection": YoloLoss(outputs[1], bboxes).item() / len(valDataLoader),
    "reconstruction": (lambdaSR * MseLoss(outputs[0], hiResImages)).item() / len(valDataLoader),
    }
    logLosses("YoloAutoencoderV2TrainingLogs.json", epoch + 1, trainLosses, valLosses)
    torch.save(model.state_dict(), 'YoloAutoencoderV2.pth')
    """
    # Auto saving of the model during Colab training
    os.system('git add  YoloAutoencoder.pth')
    current_time = datetime.now().strftime("%A, %d %B %Y")
    os.system(f'git commit YoloAutoencoder.pth -m "AutoSave of the model during training - Epoch {epoch+1}/{epochs}, Batch {batchNumber}/{totalBatch} - {current_time}"')
    os.system('git push -u origin main')
    """
    """
    # Show the last reconstructed image
    import matplotlib.pyplot as plt
    last_low_res_image = lowResImages[-1].cpu().detach().numpy().squeeze()
    last_reconstructed_image = outputs[0][-1].cpu().detach().numpy().squeeze()
    last_high_res_image = hiResImages[-1].cpu().detach().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(last_low_res_image, cmap='gray')
    axes[0].set_title("Low-Res Input")
    axes[0].axis('off')

    axes[1].imshow(last_reconstructed_image, cmap='gray')
    axes[1].set_title("Reconstructed Output")
    axes[1].axis('off')

    axes[2].imshow(last_high_res_image, cmap='gray')
    axes[2].set_title("High-Res Ground Truth")
    axes[2].axis('off')

    plt.show()
    """
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {trainLossSum/len(trainDataLoader):.4f}, Validation Loss: {valLossSum/len(valDataLoader):.4f}")