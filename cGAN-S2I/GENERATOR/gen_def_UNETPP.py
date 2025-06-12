import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class UNetPPGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=64, dropout=0.3):
        super().__init__()
        f = base_filters  # Starting number of feature maps (64)

        # Encoder blocks
        self.conv00 = ConvBlock(in_channels, f, dropout)      # 1 × 256×256 → 64 × 256×256
        self.conv10 = ConvBlock(f, f * 2, dropout)            # 64 × 128×128
        self.conv20 = ConvBlock(f * 2, f * 4, dropout)        # 128 × 64×64
        self.conv30 = ConvBlock(f * 4, f * 8, dropout)        # 256 × 32×32
        self.conv40 = ConvBlock(f * 8, f * 16, dropout)       # 512 × 16×16

        # Decoder blocks with nested dense skip connections
        self.up01 = UpBlock(f * 2, f)                         # 128 → 64
        self.conv01 = ConvBlock(f * 2, f, dropout)            # concat: 64 + 64 = 128

        self.up11 = UpBlock(f * 4, f * 2)                     # 256 → 128
        self.conv11 = ConvBlock(f * 4, f * 2, dropout)        # concat: 128 + 128 = 256
        self.up02 = UpBlock(f * 2, f)                         # 128 → 64
        self.conv02 = ConvBlock(f * 3, f, dropout)            # concat: 64 + 64 + 64 = 192

        self.up21 = UpBlock(f * 8, f * 4)                     # 512 → 256
        self.conv21 = ConvBlock(f * 8, f * 4, dropout)        # concat: 256 + 256 = 512
        self.up12 = UpBlock(f * 4, f * 2)                     # 256 → 128
        self.conv12 = ConvBlock(f * 6, f * 2, dropout)        # concat: 128*3 = 384

        self.up03 = UpBlock(f * 2, f)                         # 128 → 64
        self.conv03 = ConvBlock(f * 4, f, dropout)            # concat: 64*4 = 256

        self.up31 = UpBlock(f * 16, f * 8)                    # 1024 → 512
        self.conv31 = ConvBlock(f * 16, f * 8, dropout)       # concat: 512 + 512 = 1024

        self.up22 = UpBlock(f * 8, f * 4)                     # 512 → 256
        self.conv22 = ConvBlock(f * 12, f * 4, dropout)       # concat: 256*3 = 768

        self.up13 = UpBlock(f * 4, f * 2)                     # 256 → 128
        self.conv13 = ConvBlock(f * 8, f * 2, dropout)        # concat: 128*4 = 512

        self.up04 = UpBlock(f * 2, f)                         # 128 → 64
        self.conv04 = ConvBlock(f * 5, f, dropout)            # concat: 64*5 = 320

        self.final = nn.Conv2d(f, out_channels, kernel_size=1)  # Final 1×1 conv to RGB
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Input x: (1 × 256 × 256)

        # Encoder feature maps (after each maxpool and ConvBlock)
        x00 = self.conv00(x)                                 # 64 × 256 × 256
        x10 = self.conv10(nn.MaxPool2d(2)(x00))              # 128 × 128 × 128
        x20 = self.conv20(nn.MaxPool2d(2)(x10))              # 256 × 64 × 64
        x30 = self.conv30(nn.MaxPool2d(2)(x20))              # 512 × 32 × 32
        x40 = self.conv40(nn.MaxPool2d(2)(x30))              # 1024 × 16 × 16

        # Decoder (Nested Dense Skip Connections)

        # Level 0, step 1: x00 + up(x10)
        x01 = self.conv01(torch.cat([x00, self.up01(x10)], 1))       # 64 × 256 × 256

        # Level 1, step 1: x10 + up(x20)
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], 1))       # 128 × 128 × 128

        # Level 0, step 2: x00 + x01 + up(x11)
        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], 1))  # 64 × 256 × 256

        # Level 2, step 1: x20 + up(x30)
        x21 = self.conv21(torch.cat([x20, self.up21(x30)], 1))       # 256 × 64 × 64

        # Level 1, step 2: x10 + x11 + up(x21)
        x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], 1))  # 128 × 128 × 128

        # Level 0, step 3: x00 + x01 + x02 + up(x12)
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], 1))  # 64 × 256 × 256

        # Level 3, step 1: x30 + up(x40)
        x31 = self.conv31(torch.cat([x30, self.up31(x40)], 1))       # 512 × 32 × 32

        # Level 2, step 2: x20 + x21 + up(x31)
        x22 = self.conv22(torch.cat([x20, x21, self.up22(x31)], 1))  # 256 × 64 × 64

        # Level 1, step 3: x10 + x11 + x12 + up(x22)
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up13(x22)], 1))  # 128 × 128 × 128

        # Final output path (Level 0, step 4): concat all level 0 outputs + up(x13)
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up04(x13)], 1))  # 64 × 256 × 256

        # Final output: 3 × 256 × 256 (RGB image)
        return self.tanh(self.final(x04))