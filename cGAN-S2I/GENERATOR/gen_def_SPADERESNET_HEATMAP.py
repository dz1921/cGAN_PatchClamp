import torch
import torch.nn as nn
import torch.nn.functional as F

# SPADE Normalisation
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

# SPADE ResNet Block
class SPADEResnetBlock(nn.Module):
    def __init__(self, dim, label_nc):
        super().__init__()
        self.spade1 = SPADE(dim, label_nc)
        self.spade2 = SPADE(dim, label_nc)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x, segmap):
        dx = self.spade1(x, segmap)
        dx = self.relu(dx)
        dx = self.conv1(dx)
        dx = self.spade2(dx, segmap)
        dx = self.relu(dx)
        dx = self.conv2(dx)
        return x + dx

# SPADE ResNet Generator with Heatmap Input
class SPADEResNetGenerator(nn.Module):
    def __init__(self, input_nc=1, heatmap_nc=1, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, label_nc=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc + heatmap_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Downsampling
        self.downsampling = []
        for i in range(n_downsampling):
            mult = 2 ** i
            self.downsampling += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        self.downsampling = nn.Sequential(*self.downsampling)

        # Residual blocks with SPADE
        mult = 2 ** n_downsampling
        self.resnet_blocks = nn.ModuleList(
            [SPADEResnetBlock(ngf * mult, label_nc) for _ in range(n_blocks)]
        )

        # Upsampling
        self.upsampling = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.upsampling += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        self.upsampling = nn.Sequential(*self.upsampling)

        # Final layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, segmap, heatmap):
        # Concatenate binary mask and heatmap as input
        x = torch.cat([segmap, heatmap], dim=1)
        x = self.head(x)
        x = self.downsampling(x)
        for block in self.resnet_blocks:
            x = block(x, segmap)
        x = self.upsampling(x)
        return self.final(x)