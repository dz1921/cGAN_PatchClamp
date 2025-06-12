import torch
import torch.nn as nn
import torch.nn.functional as F


# SPADE normalisation layer
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        # Param-free normalisation (InstanceNorm without affine transformation)
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128  # Number of hidden channels in modulation MLP

        # Shared MLP that processes the segmentation map
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Separate branches to produce spatially-varying scale and bias
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Resize segmentation map to match input feature map
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')

        # Apply param-free normalisation
        normalized = self.param_free_norm(x)

        # Generate modulation parameters from segmentation
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Modulate normalised features using SPADE
        out = normalized * (1 + gamma) + beta
        return out


# ResNet block with SPADE normalisation
class SPADEResnetBlock(nn.Module):
    def __init__(self, dim, label_nc):
        super().__init__()
        self.spade1 = SPADE(dim, label_nc)
        self.spade2 = SPADE(dim, label_nc)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x, segmap):
        # First SPADE-normalised convolution
        dx = self.spade1(x, segmap)
        dx = self.relu(dx)
        dx = self.conv1(dx)

        # Second SPADE-normalised convolution
        dx = self.spade2(dx, segmap)
        dx = self.relu(dx)
        dx = self.conv2(dx)

        # Residual connection
        return x + dx


# Generator with SPADE ResNet blocks
class SPADEResNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, label_nc=1):
        super().__init__()

        # Initial convolution on input (usually the segmentation map)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Downsampling path: halves spatial resolution while doubling channel count
        self.downsampling = []
        for i in range(n_downsampling):
            mult = 2 ** i
            self.downsampling += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        self.downsampling = nn.Sequential(*self.downsampling)

        # Bottleneck: series of ResNet blocks, each modulated by SPADE
        mult = 2 ** n_downsampling
        self.resnet_blocks = nn.ModuleList(
            [SPADEResnetBlock(ngf * mult, label_nc) for _ in range(n_blocks)]
        )

        # Upsampling path: doubles spatial resolution while halving channel count
        self.upsampling = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.upsampling += [
                # Transposed convolution used for learnable upsampling
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        self.upsampling = nn.Sequential(*self.upsampling)

        # Final convolutional layer producing RGB output
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()  # Output scaled to range [-1, 1]
        )

    def forward(self, segmap):
        # Initial feature extraction from segmentation
        x = self.head(segmap)

        # Encode into bottleneck representation
        x = self.downsampling(x)

        # Apply series of SPADE-modulated residual blocks
        for block in self.resnet_blocks:
            x = block(x, segmap)

        # Decode back to high-resolution image
        x = self.upsampling(x)

        # Final projection to RGB image
        x = self.final(x)
        return x

