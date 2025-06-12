import torch
import torch.nn as nn

# CBAM: Convolutional Block Attention Module (channel and spatial attention)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        # Channel attention: focuses on informative feature maps using global pooling
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention: identifies 'where' to focus in spatial dimensions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        ca = self.channel_attention(avg_pool + max_pool)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x


# Double conv block with CBAM and dropout
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.cbam(x)  # Apply CBAM refinement


# Upsampling block: bilinear + conv projection
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.up(x)


# UNet++ generator with CBAM at each ConvBlock
class UNetPPGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=64, dropout=0.3):
        super().__init__()
        f = base_filters

        # Encoder pathway (contracting path)
        self.conv00 = ConvBlock(in_channels, f, dropout)
        self.conv10 = ConvBlock(f, f * 2, dropout)
        self.conv20 = ConvBlock(f * 2, f * 4, dropout)
        self.conv30 = ConvBlock(f * 4, f * 8, dropout)
        self.conv40 = ConvBlock(f * 8, f * 16, dropout)

        # Decoder pathway (expanding path) with nested skip connections
        self.up01 = UpBlock(f * 2, f)
        self.conv01 = ConvBlock(f * 2, f, dropout)

        self.up11 = UpBlock(f * 4, f * 2)
        self.conv11 = ConvBlock(f * 4, f * 2, dropout)
        self.up02 = UpBlock(f * 2, f)
        self.conv02 = ConvBlock(f * 3, f, dropout)

        self.up21 = UpBlock(f * 8, f * 4)
        self.conv21 = ConvBlock(f * 8, f * 4, dropout)
        self.up12 = UpBlock(f * 4, f * 2)
        self.conv12 = ConvBlock(f * 6, f * 2, dropout)
        self.up03 = UpBlock(f * 2, f)
        self.conv03 = ConvBlock(f * 4, f, dropout)

        self.up31 = UpBlock(f * 16, f * 8)
        self.conv31 = ConvBlock(f * 16, f * 8, dropout)
        self.up22 = UpBlock(f * 8, f * 4)
        self.conv22 = ConvBlock(f * 12, f * 4, dropout)
        self.up13 = UpBlock(f * 4, f * 2)
        self.conv13 = ConvBlock(f * 8, f * 2, dropout)
        self.up04 = UpBlock(f * 2, f)
        self.conv04 = ConvBlock(f * 5, f, dropout)

        # Final projection to output image
        self.final = nn.Conv2d(f, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  # Ensure output is in range [-1, 1]

    def forward(self, x):
        # Encoder pathway
        x00 = self.conv00(x)
        x10 = self.conv10(nn.MaxPool2d(2)(x00))
        x20 = self.conv20(nn.MaxPool2d(2)(x10))
        x30 = self.conv30(nn.MaxPool2d(2)(x20))
        x40 = self.conv40(nn.MaxPool2d(2)(x30))

        # UNet++ nested decoder connections
        x01 = self.conv01(torch.cat([x00, self.up01(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], dim=1))
        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up21(x30)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], dim=1))
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up31(x40)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up22(x31)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up13(x22)], dim=1))
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up04(x13)], dim=1))

        # Final output projection with tanh activation
        return self.tanh(self.final(x04))

