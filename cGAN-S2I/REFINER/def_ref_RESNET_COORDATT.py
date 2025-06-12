import torch
import torch.nn as nn
import torch.nn.functional as F

# Coordinate Attention Module
# Captures long-range dependencies along height and width separately
# while preserving precise positional information
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()

        # Apply global average pooling along height and width independently
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pool to (H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Pool to (1, W)

        # Reduce dimensionality (but keep a minimum of 8 channels)
        mip = max(8, in_channels // reduction)

        # Shared transformation of concatenated spatial descriptors
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        # Separate projections to generate height and width attention maps
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x  # Save input for residual-style modulation
        n, c, h, w = x.size()

        # Generate 1D spatial context along height and width
        x_h = self.pool_h(x)                    # Shape: [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Shape: [N, C, W, 1] → [N, C, 1, W]

        # Concatenate and transform spatial descriptors
        y = torch.cat([x_h, x_w], dim=2)  # Shape: [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into height and width pathways
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Restore to [N, C, 1, W]

        # Generate attention maps and apply them
        a_h = self.conv_h(x_h).sigmoid()  # [N, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [N, C, 1, W]

        # Modulate input features with both attention maps
        return identity * a_h * a_w


# Residual Block with optional Coordinate Attention
# Helps preserve low-level features and enables deep refinement
class ResBlock(nn.Module):
    def __init__(self, ch, use_attention=False):
        super().__init__()

        # First convolutional path
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(ch, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional path
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(ch, affine=True)

        # Final activation
        self.relu_out = nn.ReLU(inplace=True)

        # Optional coordinate attention
        self.attn = CoordAttention(ch) if use_attention else nn.Identity()

    def forward(self, x):
        residual = x  # Save input for skip connection

        # Apply conv → norm → ReLU → conv → norm
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        # Add residual connection
        out = out + residual

        # Apply attention if enabled
        out = self.attn(out)

        # Final non-linearity
        return self.relu_out(out)


# ResNet-style Background Refiner
# Designed to reduce checkerboard and patch artefacts from GAN outputs
class ResNetBackgroundRefiner(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=64, num_blocks=6, use_attention=True):
        """
        Args:
            in_channels: Number of input channels (e.g. image + mask = 3 + 1 = 4)
            out_channels: Final output image channels (typically 3 for RGB)
            base_channels: Number of channels for the first convolutional layer
            num_blocks: Number of residual blocks in the bottleneck
            use_attention: Whether to apply coordinate attention in residual blocks
        """
        super().__init__()

        # Initial feature extraction (7×7 convolution)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True)
        )

        # Middle bottleneck: a series of residual blocks with optional attention
        blocks = [ResBlock(base_channels, use_attention=use_attention) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # Output projection and image normalisation to [-1, 1]
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Input: tensor of shape [N, 4, H, W] (e.g. image + mask)
        Output: tensor of shape [N, 3, H, W] with refined RGB content
        """
        x = self.head(x)           # Initial conv + norm + ReLU
        x = self.res_blocks(x)     # Deep residual refinement
        return self.tail(x)        # Final RGB output


