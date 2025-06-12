import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DownBlock(nn.Module):
    # DOWNSAMPLING BLOCK: Conv2D - InstanceNorm - LeakyReLU
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    # SMOOTH UPSAMPLING BLOCK: Upsample - Conv2D - InstanceNorm - ReLU
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)


class UNetGenerator(nn.Module):
    # U-Net generator with smooth upsampling
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        # ENCODER
        self.down1 = DownBlock(in_channels, 64, normalize=False)  # 1 → 64
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512, normalize=False)

        # DECODER with skip connections
        self.up1 = UpBlock(512, 512, dropout=0.5)      # concat with d7
        self.up2 = UpBlock(1024, 512, dropout=0.5)     # concat with d6
        self.up3 = UpBlock(1024, 512, dropout=0.5)     # concat with d5
        self.up4 = UpBlock(1024, 512)                  # concat with d4
        self.up5 = UpBlock(1024, 256)                  # concat with d3
        self.up6 = UpBlock(512, 128)                   # concat with d2
        self.up7 = UpBlock(256, 64)                    # concat with d1

        # FINAL layer with Tanh
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # ENCODING
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # DECODING with skip connections
        u1 = self.up1(d8)
        u1 = torch.cat((u1, d7), dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat((u2, d6), dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat((u3, d5), dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat((u4, d4), dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat((u5, d3), dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat((u6, d2), dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat((u7, d1), dim=1)

        return self.final(u7)








