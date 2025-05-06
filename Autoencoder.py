from Dataset import SpaceDebrisDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from ConvolutionalBlockAttentionModule import ChannelAttention, SpatialAttention
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import gaussian_blur



#256x256x3-->256x256x64-->128x128x128-->64x64x256

# New Layer Dimensions:
# Encoder: 256x256x1-->256x256x32-->128x128x64-->64x64x128
# Decoder: 64x64x128-->128x128x64-->256x256x32-->512x512x1

# The channels have been reduced from 3 to 1, since the input images are grayscale
# The output layer now has 512x512x1 since the model will generate images with twice the resolution of the input

# Used ((W-F+2P)/S)+1 to calculate the filter size of each layer based on the dimensions
# W = input size, F = filter size, P = padding, S = stride

def AttentionInfo(index, spAttMask = None, chAttMask = None, show = False):
    if spAttMask != None:
        spMin = spAttMask.min()
        spMax = spAttMask.max()
        print(f'\n Spatial Attention {index} Min-Max: {spMin}  -  {spMax} \n')
    if chAttMask != None:
        chMin = chAttMask.min()
        chMax = chAttMask.max()
        print(f'\n Channel Attention {index} Min-Max: {chMin}  -  {chMax} \n')
    if show:
        to_pil_image = ToPILImage()
        mask = to_pil_image(spAttMask[0].cpu().squeeze(0))
        mask.show()


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

        self.channelAttention1 = ChannelAttention(128, 8)
        self.channelAttention2 = ChannelAttention(64, 8)
        self.channelAttention3 = ChannelAttention(32, 8)
        self.spatialAttention = SpatialAttention()

        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.outputLayer = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)


    def forward(self, x):
        # Used ReLU activation function as specified in the paper
        # Added skip connections as specified in the network diagram

        gaussSpAtt = self.spatialAttention(x)
        x = x + gaussSpAtt * gaussian_blur(x, kernel_size=7, sigma=(0.1, 2.0))
        AttentionInfo(0, gaussSpAtt, None, False)
        x = F.relu(self.inputLayer(x))
        chAtt = self.channelAttention3(x)
        x = chAtt * x
        spAtt = self.spatialAttention(x)
        x = spAtt * x
        AttentionInfo(1, spAtt, chAtt, False)

        # fist skip connection
        skip1 = x
        x = F.relu(self.enc1(x))
       
        # second skip connection
        skip2 = x
        x = F.relu(self.enc2(x))

        chAtt = self.channelAttention1(x)
        x = chAtt * x

        x = F.relu(self.dec1(x))
       
        # second skip connection
        x = x + skip2
        x = F.relu(self.dec2(x))
        # first skip connection
        x = x + skip1
        chAtt = self.channelAttention3(x)
        x = chAtt * x
        spAtt = self.spatialAttention(x)
        x = spAtt * x
        x = self.outputLayer(x)
       
        return x
