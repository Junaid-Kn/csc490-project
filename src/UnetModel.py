import torch.nn as nn
import torch

from DoubleConv import DoubleConv


class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()

        # Encoder - deeper with more channels
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder - with proper skip connections
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 + 512 (skip)

        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)   # 512 = 256 + 256 (skip)

        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 256 = 128 + 128 (skip)

        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # 128 = 64 + 64 (skip)

        # Final convolution with sigmoid for output normalization
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)

        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)

        enc3_out = self.enc3(pool2_out)
        pool3_out = self.pool3(enc3_out)

        enc4_out = self.enc4(pool3_out)
        pool4_out = self.pool4(enc4_out)

        # Bottleneck
        bottleneck_out = self.bottleneck(pool4_out)

        # Decoder
        up4 = self.up_conv4(bottleneck_out)
        merge4 = torch.cat([enc4_out, up4], dim=1)
        dec4_out = self.dec4(merge4)

        up3 = self.up_conv3(dec4_out)
        merge3 = torch.cat([enc3_out, up3], dim=1)
        dec3_out = self.dec3(merge3)

        up2 = self.up_conv2(dec3_out)
        merge2 = torch.cat([enc2_out, up2], dim=1)
        dec2_out = self.dec2(merge2)

        up1 = self.up_conv1(dec2_out)
        merge1 = torch.cat([enc1_out, up1], dim=1)
        dec1_out = self.dec1(merge1)

        # Final output
        output = self.final_conv(dec1_out)

        return output
