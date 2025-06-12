import torch
import torch.nn as nn


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        # Global average and max pooling across spatial dimensions (H, W)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        # Shared multi-layer perceptron applied to both pooled features
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights from both average and max pooled features
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale  # Apply channel-wise modulation


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2  # Ensures output retains original dimensions
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Aggregate across channels using average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(combined))
        return x * scale  # Apply spatial-wise modulation


# CBAM: Convolutional Block Attention Module
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# Residual Block with CBAM enhancement
class ResnetCBAMBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_bias=False):
        super(ResnetCBAMBlock, self).__init__()
        self.cbam = CBAM(dim)
        self.block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        block = []
        # If padding is not zero, padding must be applied manually before conv
        p = 1 if padding_type == 'zero' else 0
        if padding_type == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block += [nn.ReplicationPad2d(1)]

        # First convolutional layer
        block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        # Second convolutional layer with same padding logic
        if padding_type == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block += [nn.ReplicationPad2d(1)]

        block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
        return nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        out = self.cbam(out)  # Apply CBAM after residual body
        return x + out  # Standard skip connection


# Full CBAM-ResNet Generator
class CBAMResNetGenerator(nn.Module):
    def __init__(self,
                 input_nc=1,       # Number of input channels (e.g., mask only or mask + heatmap)
                 output_nc=3,      # Number of output channels (e.g., RGB image)
                 ngf=64,           # Base number of generator filters
                 n_downsampling=3, # Number of downsampling layers
                 n_blocks=9,       # Number of ResNet blocks
                 norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect'):
        super(CBAMResNetGenerator, self).__init__()

        use_bias = (norm_layer == nn.InstanceNorm2d)
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),  # Preserves spatial size when using a 7×7 convolution
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
            norm_layer(ngf),
            activation
        ]

        # Downsampling: halve spatial resolution, double channel count
        for i in range(n_downsampling):
            in_ch = ngf * (2 ** i)
            out_ch = ngf * (2 ** (i + 1))
            model += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(out_ch),
                activation
            ]

        # Residual blocks: maintain feature resolution and apply CBAM
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetCBAMBlock(ngf * mult, padding_type, norm_layer, use_bias)]

        # Upsampling: gradually increase spatial resolution while reducing the number of feature channels
        for i in range(n_downsampling):
            in_ch = ngf * (2 ** (n_downsampling - i))  # Number of input channels (e.g., 512 → 256 → 128)
            out_ch = in_ch // 2                        # Halve the number of channels at each step

            model += [
                # Smooth upsampling via bilinear interpolation (non-learnable)
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

                # Convolution to refine and adapt the upsampled features (learnable)
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias),

                # Normalisation and activation, as used in the encoder
                norm_layer(out_ch),
                activation
            ]

        # Output layer: maps back to desired output channel count with tanh activation
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, bias=use_bias),
            nn.Tanh()  # Output is scaled to [-1, 1] range
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)