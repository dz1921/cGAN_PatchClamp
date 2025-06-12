import torch
import torch.nn as nn

class DownBlock(nn.Module):
    # DOWNSAMPLING BLOCK: conv2D - norm - LeakyReLU
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = []

        # CONVOLUTION (kernel = 4, stride = 2, padding = 1)
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        )

        # NORMALISATION
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # DROPOUT
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)
    

class UpBlock(nn.Module):
    # UPSAMPLING BLOCK: convtranspose2D - norm - dropout - ReLU
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = []

        # TRANSPOSED CONVOLUTION
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))

        # NORMALISATION
        layers.append(nn.BatchNorm2d(out_channels))

        # ReLU
        layers.append(nn.ReLU(inplace=True))

        # DROPOUT
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)
    

class UNetGenerator(nn.Module):
    # pix2pix-style generator with skip connections (U-Net)
    # default: input 1 channel, output 1 channel (greyscale) at 256x256 resolution
        
    def __init__(self, in_channels=1, out_channels=1):  # 1 for greyscale
        super().__init__()

        # DOWNSAMPLING - ENCODER
        self.down1 = DownBlock(in_channels, 64, normalize=False)       # 1 -> 64
        self.down2 = DownBlock(64, 128)                                # 64 -> 128
        self.down3 = DownBlock(128, 256)                               # 128 -> 256
        self.down4 = DownBlock(256, 512)                               # 256 -> 512
        self.down5 = DownBlock(512, 512)                               # 512 -> 512
        self.down6 = DownBlock(512, 512)                               # 512 -> 512
        self.down7 = DownBlock(512, 512)                               # 512 -> 512
        self.down8 = DownBlock(512, 512, normalize=False)              # 512 -> 512

        # UPSAMPLING - DECODER
        self.up1 = UpBlock(512, 512, dropout=0.5)    # skip connection with down7
        self.up2 = UpBlock(1024, 512, dropout=0.5)   # skip connection with down6
        self.up3 = UpBlock(1024, 512, dropout=0.5)   # skip connection with down5
        self.up4 = UpBlock(1024, 512)                # skip connection with down4
        self.up5 = UpBlock(1024, 256)                # skip connection with down3
        self.up6 = UpBlock(512, 128)                 # skip connection with down2
        self.up7 = UpBlock(256, 64)                  # skip connection with down1
        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)

        # TANH ACTIVATION
        self.tanh = nn.Tanh()

    def forward(self, x):

        # DOWNSAMPLING
        d1 = self.down1(x)  # shape: [N, 64, 128, 128]
        d2 = self.down2(d1) # shape: [N, 128, 64, 64]
        d3 = self.down3(d2) # shape: [N, 256, 32, 32]
        d4 = self.down4(d3) # shape: [N, 512, 16, 16]
        d5 = self.down5(d4) # shape: [N, 512, 8, 8]
        d6 = self.down6(d5) # shape: [N, 512, 4, 4]
        d7 = self.down7(d6) # shape: [N, 512, 2, 2]
        d8 = self.down8(d7) # shape: [N, 512, 1, 1]

        # UPSAMPLING and SKIP CONNECTIONS
        u1 = self.up1(d8)                # [N, 512, 2, 2]
        u1 = torch.cat((u1, d7), dim=1)  # [N, 1024, 2, 2]

        u2 = self.up2(u1)                # [N, 512, 4, 4]
        u2 = torch.cat((u2, d6), dim=1)  # [N, 1024, 4, 4]

        u3 = self.up3(u2)                # [N, 512, 8, 8]
        u3 = torch.cat((u3, d5), dim=1)  # [N, 1024, 8, 8]

        u4 = self.up4(u3)                # [N, 512, 16, 16]
        u4 = torch.cat((u4, d4), dim=1)  # [N, 1024, 16, 16]

        u5 = self.up5(u4)                # [N, 256, 32, 32]
        u5 = torch.cat((u5, d3), dim=1)  # [N, 512, 32, 32]

        u6 = self.up6(u5)                # [N, 128, 64, 64]
        u6 = torch.cat((u6, d2), dim=1)  # [N, 256, 64, 64]

        u7 = self.up7(u6)                # [N, 64, 128, 128]
        u7 = torch.cat((u7, d1), dim=1)  # [N, 128, 128, 128]

        # Final layer - TANH
        out = self.final(u7)             # [N, 1, 256, 256] (greyscale output)
        out = self.tanh(out)
        return out
