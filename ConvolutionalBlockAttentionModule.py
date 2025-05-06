import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, inChannels, redRatio):
        super(ChannelAttention, self).__init__()
        # The pooling layers output a 1x1xC tensor, meaning a single value for each channel
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(nn.Conv2d(inChannels, inChannels // redRatio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inChannels // redRatio, inChannels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avgPool(x))
        max = self.mlp(self.maxPool(x))

        add = avg + max
        x = self.sigmoid(add)

        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernelSize=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernelSize, stride=1, padding=(kernelSize-1)//2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True).values
        x = torch.cat([avg, max], dim=1)
        x = self.conv(x)
        sigTemp = 10
        x = torch.sigmoid((x - 0.5) * sigTemp)

        return x