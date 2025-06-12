import torch
import torch.nn as nn
import torch.nn.functional as F


# SPADE: Spatially-Adaptive Denormalisation
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        # InstanceNorm without affine transform (i.e. param-free)
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128  # hidden layer size for modulation MLP

        # Shared MLP to process the segmentation map
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Generate spatially-varying scale (gamma) and bias (beta)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Resize segmentation map to match input resolution
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')

        # Apply instance normalisation
        normalized = self.param_free_norm(x)

        # Get modulation parameters from segmentation map
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Apply spatially-adaptive modulation
        return normalized * (1 + gamma) + beta


# CBAM: Convolutional Block Attention Module (channel + spatial attention)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention: global pooling followed by shared MLP
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention: convolution over concatenated channel-wise stats
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention stage
        ca = self.channel_attention(x) * x

        # Compute spatial descriptors from channel-attended output
        max_pool = torch.max(ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(ca, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)

        # Apply spatial attention and return result
        sa = self.spatial_attention(sa_input) * ca
        return sa


# Downsampling block: convolution -> optional norm -> LeakyReLU -> optional dropout
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


# Upsampling block: transpose conv -> SPADE -> ReLU -> CBAM -> optional dropout
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc, dropout=0.0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.spade = SPADE(out_channels, label_nc)
        self.cbam = CBAM(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, segmap):
        # Upsample using transposed convolution
        x = self.deconv(x)

        # Apply SPADE normalisation with segmentation guidance
        x = self.spade(x, segmap)
        x = self.relu(x)

        # Refine features using CBAM
        x = self.cbam(x)

        # Optionally apply dropout
        x = self.dropout(x)
        return x


# Generator architecture combining UNet, SPADE, and CBAM
class SPADECBAMUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, label_nc=1):
        super().__init__()

        # Encoder: downsampling path
        self.down1 = DownBlock(in_channels, 64, normalize=False)  # No norm on input
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512, normalize=False)  # Bottleneck

        # Decoder: upsampling with SPADE + CBAM + skip connections
        self.up1 = UpBlock(512, 512, label_nc, dropout=0.5)
        self.up2 = UpBlock(1024, 512, label_nc, dropout=0.5)
        self.up3 = UpBlock(1024, 512, label_nc, dropout=0.5)
        self.up4 = UpBlock(1024, 512, label_nc)
        self.up5 = UpBlock(1024, 256, label_nc)
        self.up6 = UpBlock(512, 128, label_nc)
        self.up7 = UpBlock(256, 64, label_nc)

        # Final output convolution
        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()  # Output image in [-1, 1] range

    def forward(self, x, segmap):
        # Downsampling path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Upsampling with skip connections and conditional guidance
        u1 = self.up1(d8, segmap)
        u1 = torch.cat((u1, d7), dim=1)

        u2 = self.up2(u1, segmap)
        u2 = torch.cat((u2, d6), dim=1)

        u3 = self.up3(u2, segmap)
        u3 = torch.cat((u3, d5), dim=1)

        u4 = self.up4(u3, segmap)
        u4 = torch.cat((u4, d4), dim=1)

        u5 = self.up5(u4, segmap)
        u5 = torch.cat((u5, d3), dim=1)

        u6 = self.up6(u5, segmap)
        u6 = torch.cat((u6, d2), dim=1)

        u7 = self.up7(u6, segmap)
        u7 = torch.cat((u7, d1), dim=1)

        # Final output projection
        out = self.final(u7)
        return self.tanh(out)
