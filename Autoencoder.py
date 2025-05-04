from Dataset import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from ConvolutionalBlockAttentionModule import ChannelAttention, SpatialAttention


#256x256x3-->256x256x64-->128x128x128-->64x64x256

# New Layer Dimensions:
# Encoder: 256x256x1-->256x256x32-->128x128x64-->64x64x128
# Decoder: 64x64x128-->128x128x64-->256x256x32-->512x512x1

# The channels have been reduced from 3 to 1, since the input images are grayscale
# The output layer now has 512x512x1 since the model will generate images with twice the resolution of the input





# Used ((W-F+2P)/S)+1 to calculate the filter size of each layer based on the dimensions
# W = input size, F = filter size, P = padding, S = stride

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        #self.inputLayer = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)

        # Trying a bigger kernel size to reduce the noise in the generated image
        self.inputLayer = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3)

        self.enc1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        #CBAM

        self.channelAttention = ChannelAttention(128, 8)
        self.spatialAttention = SpatialAttention()

        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.outputLayer = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)


    def forward(self, x):
        # Used ReLU activation function as specified in the paper
        # Added skip connections as specified in the network diagram

        #print('\n Input Image: \n')
        #print(x.shape)
        x = F.relu(self.inputLayer(x))
        #print('\n Input Layer: \n')
        #print(x.shape)
        # fist skip connection
        skip1 = x
        x = F.relu(self.enc1(x))
        #print('\n Encoder 1:  \n')
        #print(x.shape)
        # second skip connection
        skip2 = x
        x = F.relu(self.enc2(x))
        #print('\n Encoder 2: \n')
        #print(x.shape)
        x = self.channelAttention(x) * x
        x = self.spatialAttention(x) * x
        x = F.relu(self.dec1(x))
        #print('\n Decoder 1: \n')
        #print(x.shape)
        # second skip connection
        x = x + skip2
        x = F.relu(self.dec2(x))
        #print('\n Decoder 2: \n')
        #print(x.shape)
        # first skip connection
        x = x + skip1
        x = self.outputLayer(x)
        #print('\n Output Layer: \n ')
        #print(x.shape)
        return x
