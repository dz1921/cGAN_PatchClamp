import torch
import torch.nn as nn
import torch.nn.functional as F

# SPADE Normalisation Module
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, 3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, 3, padding=1)

    def forward(self, x, segmap):
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')
        normed = self.param_free_norm(x)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normed * (1 + gamma) + beta

# SPADE-based Conv Block
class SPADEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = SPADE(out_channels, label_nc)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = SPADE(out_channels, label_nc)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, segmap):
        x = self.conv1(x)
        x = self.norm1(x, segmap)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x, segmap)
        x = self.relu2(x)
        return x

# UpSampling block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

# UNet++ Generator with SPADE
class UNetPPGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=64, dropout=0.3):
        super().__init__()
        f = base_filters
        self.label_nc = in_channels

        # Encoder
        self.conv00 = SPADEConvBlock(in_channels, f, self.label_nc, dropout)
        self.conv10 = SPADEConvBlock(f, f * 2, self.label_nc, dropout)
        self.conv20 = SPADEConvBlock(f * 2, f * 4, self.label_nc, dropout)
        self.conv30 = SPADEConvBlock(f * 4, f * 8, self.label_nc, dropout)
        self.conv40 = SPADEConvBlock(f * 8, f * 16, self.label_nc, dropout)

        # Decoder
        self.up01 = UpBlock(f * 2, f)
        self.conv01 = SPADEConvBlock(f * 2, f, self.label_nc, dropout)

        self.up11 = UpBlock(f * 4, f * 2)
        self.conv11 = SPADEConvBlock(f * 4, f * 2, self.label_nc, dropout)
        self.up02 = UpBlock(f * 2, f)
        self.conv02 = SPADEConvBlock(f * 3, f, self.label_nc, dropout)

        self.up21 = UpBlock(f * 8, f * 4)
        self.conv21 = SPADEConvBlock(f * 8, f * 4, self.label_nc, dropout)
        self.up12 = UpBlock(f * 4, f * 2)
        self.conv12 = SPADEConvBlock(f * 6, f * 2, self.label_nc, dropout)

        self.up03 = UpBlock(f * 2, f)
        self.conv03 = SPADEConvBlock(f * 4, f, self.label_nc, dropout)

        self.up31 = UpBlock(f * 16, f * 8)
        self.conv31 = SPADEConvBlock(f * 16, f * 8, self.label_nc, dropout)
        self.up22 = UpBlock(f * 8, f * 4)
        self.conv22 = SPADEConvBlock(f * 12, f * 4, self.label_nc, dropout)

        self.up13 = UpBlock(f * 4, f * 2)
        self.conv13 = SPADEConvBlock(f * 8, f * 2, self.label_nc, dropout)

        self.up04 = UpBlock(f * 2, f)
        self.conv04 = SPADEConvBlock(f * 5, f, self.label_nc, dropout)

        self.final = nn.Conv2d(f, out_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, seg):
        # Encoder
        x00 = self.conv00(x, seg)
        x10 = self.conv10(F.max_pool2d(x00, 2), seg)
        x20 = self.conv20(F.max_pool2d(x10, 2), seg)
        x30 = self.conv30(F.max_pool2d(x20, 2), seg)
        x40 = self.conv40(F.max_pool2d(x30, 2), seg)

        # Decoder
        x01 = self.conv01(torch.cat([x00, self.up01(x10)], 1), seg)
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], 1), seg)
        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], 1), seg)

        x21 = self.conv21(torch.cat([x20, self.up21(x30)], 1), seg)
        x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], 1), seg)
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], 1), seg)

        x31 = self.conv31(torch.cat([x30, self.up31(x40)], 1), seg)
        x22 = self.conv22(torch.cat([x20, x21, self.up22(x31)], 1), seg)
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up13(x22)], 1), seg)
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up04(x13)], 1), seg)

        return self.tanh(self.final(x04))
