import torch.nn as nn
import torch.nn.functional as F
from ConvolutionalBlockAttentionModule import ChannelAttention, SpatialAttention, AttentionInfo



# New Layer Dimensions:
# Encoder: 256x256x1-->256x256x32-->128x128x64-->64x64x128
# Decoder: 64x64x128-->128x128x64-->256x256x32-->512x512x1

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.inputLayer = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)

        # Trying a bigger kernel size to reduce the noise in the generated image
        #self.inputLayer = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3)

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

        x = F.relu(self.inputLayer(x))

        # fist skip connection
        skip1 = x
        x = F.relu(self.enc1(x))
       
        # second skip connection
        skip2 = x
        x = F.relu(self.enc2(x))


        chAtt = self.channelAttention(x)
        x = chAtt * x
        spAtt = self.spatialAttention(x, 3)
        x = spAtt * x
        AttentionInfo(1, spAtt, chAtt)
        x = F.relu(self.dec1(x))
       
        # second skip connection
        x = x + skip2
        x = F.relu(self.dec2(x))

        # first skip connection
        x = x + skip1
        x = self.outputLayer(x)
        
        return x
