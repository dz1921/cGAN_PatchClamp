import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM MODULE
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared_mlp(self.avg_pool(x))
        max_ = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)


# SPADE MODULE
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


# SPADE + CBAM ResNet Block
class SPADECBAMResnetBlock(nn.Module):
    def __init__(self, dim, label_nc):
        super().__init__()
        self.spade1 = SPADE(dim, label_nc)
        self.spade2 = SPADE(dim, label_nc)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.cbam = CBAM(dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, segmap):
        dx = self.spade1(x, segmap)
        dx = self.relu(dx)
        dx = self.conv1(dx)
        dx = self.spade2(dx, segmap)
        dx = self.relu(dx)
        dx = self.conv2(dx)
        dx = self.cbam(dx)
        return x + dx


class SPADECBAMResNetGenerator(nn.Module):
    def __init__(self, input_nc=1, heatmap_nc=1, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, label_nc=1):
        super().__init__()
        total_input_nc = input_nc + heatmap_nc

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(total_input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Downsampling layers
        self.downsampling = []
        for i in range(n_downsampling):
            mult = 2 ** i
            self.downsampling += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        self.downsampling = nn.Sequential(*self.downsampling)

        # Residual SPADE + CBAM blocks
        mult = 2 ** n_downsampling
        self.resnet_blocks = nn.ModuleList(
            [SPADECBAMResnetBlock(ngf * mult, label_nc) for _ in range(n_blocks)]
        )

        # Upsampling layers
        self.upsampling = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.upsampling += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        self.upsampling = nn.Sequential(*self.upsampling)

        # Final output layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, segmap, heatmap=None):
        if heatmap is not None:
            x_input = torch.cat([segmap, heatmap], dim=1)
        else:
            x_input = segmap

        x = self.head(x_input)
        x = self.downsampling(x)
        for block in self.resnet_blocks:
            x = block(x, segmap)  # segmap still used for SPADE normalisation
        x = self.upsampling(x)
        x = self.final(x)
        return x