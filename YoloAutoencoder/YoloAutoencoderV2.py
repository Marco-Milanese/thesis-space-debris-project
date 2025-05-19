import torch.nn as nn
from CbamForYolo import CBAM, SpatialAttention

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #CBAM(inChannels=32, redRatio=4, kernelSize=7)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
                              
     
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            CBAM(inChannels=256, redRatio=16, kernelSize=7)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAM(inChannels=128, redRatio=16, kernelSize=7)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAM(inChannels=32, redRatio=8, kernelSize=7)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )


        # detection head
        self.detection = nn.Sequential(
            CBAM(inChannels=512, redRatio=16, kernelSize=7),
            nn.Conv2d(512, 5, 1)
        )




    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        # print("Shape after enc1:", x.shape)
        skip1 = x
        x = self.enc2(x)
        # print("Shape after enc2:", x.shape)
        x = self.enc3(x)
        skip2 = x
        # print("Shape after enc3:", x.shape)
        x = self.enc4(x)
        # print("Shape after enc4:", x.shape)
        xDec = self.enc5(x)
        # print("Shape after enc5:", x.shape)

        # Detection head
        bboxes = self.detection(xDec)
        # print("Shape after detection head:", bboxes.shape)

        # Decoder
        #x = self.dec1(x) + skip2
        # print("Shape after dec1:", x.shape)
        x = self.dec2(x) + skip2
        # print("Shape after dec2:", x.shape)
        x = self.dec3(x)
        # print("Shape after dec3:", x.shape)
        x = self.dec4(x) + skip1
        # print("Shape after dec4:", x.shape)
        x = self.dec5(x)
        # print("Shape after dec5:", x.shape)
        generated_image = self.dec6(x)
        # print("Shape after dec6 (generated image):", generated_image.shape)

        return generated_image, bboxes
