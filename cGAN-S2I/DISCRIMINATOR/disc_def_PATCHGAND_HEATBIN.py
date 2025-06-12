import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator that accepts a segmentation mask, a heatmap, and an image.
    Input: (mask, heatmap, real_or_fake_image) → concatenated to (N, 5, H, W)
    Output: patch-level real/fake prediction (N, 1, H', W')
    """
    def __init__(self, in_channels=5):  # 1 (mask) + 1 (heatmap) + 3 (image) = 5
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, mask, heatmap, image):
        """
        Args:
            mask (Tensor): shape (N, 1, H, W)
            heatmap (Tensor): shape (N, 1, H, W)
            image (Tensor): shape (N, 3, H, W)
        Returns:
            Tensor: shape (N, 1, H', W') with patch-level real/fake logits
        """
        x = torch.cat((mask, heatmap, image), dim=1)  # shape (N, 5, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final(x)
        return x