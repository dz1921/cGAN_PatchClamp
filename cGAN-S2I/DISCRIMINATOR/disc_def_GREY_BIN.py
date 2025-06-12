import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    pix2pix-style discriminator (PatchGAN)
    Input: (input_condition, real_or_fake_image) concatenated along channel dimensions
    Output: (N, 1, H, W) patch-level real/fake prediction.
    """
    def __init__(self, in_channels=3):  # 1 (mask) + 3 (RGB image) = 4
        super().__init__()

        # Layer 1: no normalisation
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 4
        # stride=1 here, which helps keep the patch size around 70×70
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final output layer (stride=1)
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
        # No activation, BCE loss applied externally

    def forward(self, condition, image):
        # condition: (N, 1, H, W) mask
        # image: (N, 3, H, W) real or fake  image

        # Concatenate along channel dimension: shape - (N, 4, H, W)
        x = torch.cat((condition, image), dim=1)

        # Forward pass
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final(x)  # (N, 1, H', W')
        return x





