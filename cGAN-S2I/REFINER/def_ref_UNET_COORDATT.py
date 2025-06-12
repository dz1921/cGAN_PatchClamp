import torch
import torch.nn as nn
import torch.nn.functional as F

# Coordinate Attention Module
# This mechanism captures long-range dependencies along height and width separately,
# helping the network focus on spatially important features.
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        # Pooling across height and width independently to create spatial context descriptors
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # Outputs shape (N, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # Outputs shape (N, C, 1, W)

        # Reduce the number of channels for intermediate representation
        mip = max(8, in_channels // reduction)

        # Shared 1×1 convolution to process both pooled features
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        # Separate projections for height and width attention
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool along height and width separately
        x_h = self.pool_h(x)  # → (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # → (N, C, W, 1)

        # Concatenate and process through conv + BN + ReLU
        y = torch.cat([x_h, x_w], dim=2)  # → (N, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split features back and restore shape
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention to input
        return identity * a_h * a_w


# Double convolution block followed by Coordinate Attention
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )
        self.ca = CoordAttention(out_ch)

    def forward(self, x):
        x = self.conv(x)
        return self.ca(x)


# Downsampling block: max pooling followed by a DoubleConv
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


# Upsampling block: bilinear upsampling followed by concatenation and DoubleConv
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # Upsample x1 to match spatial resolution of x2
        x1 = self.up(x1)

        # Calculate padding needed due to potential size mismatch
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # Pad x1 to align with x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension and apply conv
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Full U-Net style Background Refiner using CoordAttention
class UNetBackgroundRefiner(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=64):
        super().__init__()
        f = base_filters  # Base number of filters for first layer

        # Encoder path (downsampling)
        self.inc = DoubleConv(in_channels, f)       # 4 → 64
        self.down1 = Down(f, f * 2)                 # 64 → 128
        self.down2 = Down(f * 2, f * 4)             # 128 → 256
        self.down3 = Down(f * 4, f * 8)             # 256 → 512
        self.down4 = Down(f * 8, f * 8)             # 512 → 512

        # Decoder path (upsampling with skip connections)
        self.up1 = Up(f * 16, f * 4)                # (512 + 512) → 256
        self.up2 = Up(f * 8, f * 2)                 # (256 + 256) → 128
        self.up3 = Up(f * 4, f)                     # (128 + 128) → 64
        self.up4 = Up(f * 2, f)                     # (64 + 64)   → 64

        # Final projection layer: maps features to RGB output and applies tanh
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)  # 64 → 3
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoding path (feature extraction and downsampling)
        x1 = self.inc(x)     # [B, 64, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 128, 128]
        x3 = self.down2(x2)  # [B, 256, 64, 64]
        x4 = self.down3(x3)  # [B, 512, 32, 32]
        x5 = self.down4(x4)  # [B, 512, 16, 16]

        # Decoding path with skip connections
        x = self.up1(x5, x4) # [B, 256, 32, 32]
        x = self.up2(x, x3)  # [B, 128, 64, 64]
        x = self.up3(x, x2)  # [B, 64, 128, 128]
        x = self.up4(x, x1)  # [B, 64, 256, 256]

        # Final output layer with tanh to constrain values in [-1, 1]
        x = self.tanh(self.outc(x))  # [B, 3, 256, 256]
        return x

