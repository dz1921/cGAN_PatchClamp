import torch
import torch.nn as nn


# CBAM MODULE
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# CBAM-Enhanced ResNet Block
class ResnetCBAMBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_bias=False):
        super(ResnetCBAMBlock, self).__init__()
        self.cbam = CBAM(dim)
        self.block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        block = []
        p = 1 if padding_type == 'zero' else 0
        if padding_type == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block += [nn.ReplicationPad2d(1)]

        block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

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
        out = self.cbam(out)
        return x + out

class CBAMResNetGenerator(nn.Module):
    def __init__(self,
                 input_nc=1,
                 heatmap_nc=1,
                 output_nc=3,
                 ngf=64,
                 n_downsampling=3,
                 n_blocks=9,
                 norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect'):
        super(CBAMResNetGenerator, self).__init__()

        self.input_nc_total = input_nc + heatmap_nc
        use_bias = (norm_layer == nn.InstanceNorm2d)
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.input_nc_total, ngf, kernel_size=7, bias=use_bias),
            norm_layer(ngf),
            activation
        ]

        # Downsampling
        for i in range(n_downsampling):
            in_ch = ngf * (2 ** i)
            out_ch = ngf * (2 ** (i + 1))
            model += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(out_ch),
                activation
            ]

        # ResNet blocks with CBAM
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetCBAMBlock(ngf * mult, padding_type, norm_layer, use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            in_ch = ngf * (2 ** (n_downsampling - i))
            out_ch = in_ch // 2
            model += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=use_bias),
                norm_layer(out_ch),
                activation
            ]

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, heatmap=None):
        if heatmap is not None:
            x = torch.cat([x, heatmap], dim=1)  # Concatenate along channel axis
        return self.model(x)
