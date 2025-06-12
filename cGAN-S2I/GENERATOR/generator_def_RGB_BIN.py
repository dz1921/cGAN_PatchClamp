import torch
import torch.nn as nn

class DownBlock(nn.Module):
    # DOWNSAMPLING BLOCK: conv2D - InstanceNorm - LeakyReLU
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = []

        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        )

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    # UPSAMPLING BLOCK: convtranspose2D - InstanceNorm - ReLU
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = []

        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        )

        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)
    
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=3):
        super().__init__()

        # Encoder (Downsampling)
        self.down1 = DownBlock(input_nc, 64, normalize=False)  # No norm in first layer
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512, normalize=False)  # bottleneck

        # Decoder (Upsampling)
        self.up1 = UpBlock(512, 512)
        self.up2 = UpBlock(1024, 512)
        self.up3 = UpBlock(1024, 512)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        # Final output layer (ConvTranspose2d)
        self.final = nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)  # [N, 64, 128, 128]
        d2 = self.down2(d1) # [N, 128, 64, 64]
        d3 = self.down3(d2) # [N, 256, 32, 32]
        d4 = self.down4(d3) # [N, 512, 16, 16]
        d5 = self.down5(d4) # [N, 512, 8, 8]
        d6 = self.down6(d5) # [N, 512, 4, 4]
        d7 = self.down7(d6) # [N, 512, 2, 2]
        d8 = self.down8(d7) # [N, 512, 1, 1]

        # Upsample + skip connections
        u1 = self.up1(d8)           # [N, 512, 2, 2]
        u1 = torch.cat((u1, d7), 1) # [N, 1024, 2, 2]

        u2 = self.up2(u1)           # [N, 512, 4, 4]
        u2 = torch.cat((u2, d6), 1) # [N, 1024, 4, 4]

        u3 = self.up3(u2)           # [N, 512, 8, 8]
        u3 = torch.cat((u3, d5), 1) # [N, 1024, 8, 8]

        u4 = self.up4(u3)           # [N, 512, 16, 16]
        u4 = torch.cat((u4, d4), 1) # [N, 1024, 16, 16]

        u5 = self.up5(u4)           # [N, 256, 32, 32]
        u5 = torch.cat((u5, d3), 1) # [N, 512, 32, 32]

        u6 = self.up6(u5)           # [N, 128, 64, 64]
        u6 = torch.cat((u6, d2), 1) # [N, 256, 64, 64]

        u7 = self.up7(u6)           # [N, 64, 128, 128]
        u7 = torch.cat((u7, d1), 1) # [N, 128, 128, 128]

        out = self.final(u7)        # [N, 3, 256, 256]
        return self.tanh(out)






