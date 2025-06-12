import torch
import torch.nn as nn


# Channel Attention: learns to emphasise informative feature channels
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Global average and max pooling compress each feature map to a single value
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        # Shared MLP used for both pooled descriptors
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Produce attention maps from average and max pooled features, then combine
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


# Spatial Attention: focuses on "where" to attend within each feature map
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # A 7×7 convolution processes the concatenated channel-wise max and mean
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial descriptors: average and maximum over channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and generate spatial attention map
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


# CBAM: Convolutional Block Attention Module (channel + spatial attention)
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Sequentially apply channel and spatial attention
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# Downsampling block: conv -> (optional norm) -> leaky ReLU -> (optional dropout)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


# Upsampling block: transposed conv -> norm -> ReLU -> (optional dropout)
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)


# UNet-style generator architecture with CBAM-enhanced skip connections
class UNetCBAMGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        # Contracting path (encoder): downsample image while increasing depth
        self.down1 = DownBlock(in_channels, 64, normalize=False)  # no norm on first layer
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512, normalize=False)  # bottleneck layer

        # Expanding path (decoder): upsample and merge with encoder features
        self.up1 = UpBlock(512, 512, dropout=0.5)
        self.cbam1 = CBAM(1024)  # after concatenating with down7 (512+512)

        self.up2 = UpBlock(1024, 512, dropout=0.5)
        self.cbam2 = CBAM(1024)

        self.up3 = UpBlock(1024, 512, dropout=0.5)
        self.cbam3 = CBAM(1024)

        self.up4 = UpBlock(1024, 512)
        self.cbam4 = CBAM(1024)

        self.up5 = UpBlock(1024, 256)
        self.cbam5 = CBAM(512)  # now skip connections are 256+256

        self.up6 = UpBlock(512, 128)
        self.cbam6 = CBAM(256)

        self.up7 = UpBlock(256, 64)
        self.cbam7 = CBAM(128)

        # Final layer: map 128-channel output to RGB (or other desired channels)
        self.final = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()  # restrict output to [-1, 1] range

    def forward(self, x):
        # Encoder forward pass (store outputs for skip connections)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with CBAM-enhanced skip connections
        u1 = self.up1(d8)
        u1 = self.cbam1(torch.cat((u1, d7), dim=1))

        u2 = self.up2(u1)
        u2 = self.cbam2(torch.cat((u2, d6), dim=1))

        u3 = self.up3(u2)
        u3 = self.cbam3(torch.cat((u3, d5), dim=1))

        u4 = self.up4(u3)
        u4 = self.cbam4(torch.cat((u4, d4), dim=1))

        u5 = self.up5(u4)
        u5 = self.cbam5(torch.cat((u5, d3), dim=1))

        u6 = self.up6(u5)
        u6 = self.cbam6(torch.cat((u6, d2), dim=1))

        u7 = self.up7(u6)
        u7 = self.cbam7(torch.cat((u7, d1), dim=1))

        # Final output projection
        return self.tanh(self.final(u7))
