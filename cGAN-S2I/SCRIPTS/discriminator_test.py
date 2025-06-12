import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from DISCRIMINATOR.discriminator_def_RGB_BIN import PatchGANDiscriminator

if __name__ == "__main__":
    mask = torch.randn(2, 1, 256, 256)
    real_or_fake_img = torch.randn(2, 3, 256, 256)

    netD = PatchGANDiscriminator(in_channels=4)

    out = netD(mask, real_or_fake_img)
    print("Discriminator output shape:", out.shape)
    # (N, 1, 30, 30) for 256×256 input
