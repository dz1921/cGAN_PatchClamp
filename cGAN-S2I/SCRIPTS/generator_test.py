import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GENERATOR.generator_def_RGB_BIN import UNetGenerator, init_weights 
import torch

if __name__ == "__main__":
    netG = UNetGenerator(in_channels=1, out_channels=3)
    init_weights(netG, init_type="normal", init_gain=0.02)

    x = torch.randn(2, 1, 256, 256)
    y = netG(x)
    print(f"Output shape: {y.shape}")  # Should be [2, 3, 256, 256]