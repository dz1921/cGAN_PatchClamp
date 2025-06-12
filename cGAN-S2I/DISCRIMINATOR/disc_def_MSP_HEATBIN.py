import torch
import torch.nn as nn

# NLayerDiscriminator (PatchGAN)
class NLayerDiscriminator(nn.Module):
    """
    A PatchGAN discriminator
    It classifies overlapping patches of the (condition, image) pair as real or fake.
    """
    def __init__(self, 
                 input_nc=5,    # 1 (seg) + 1 (heatmap) + 3 (image)
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()

        use_bias = (norm_layer == nn.InstanceNorm2d)
        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, segmap, heatmap, img):
        """
        Args:
            segmap (Tensor): Segmentation map (B, 1, H, W).
            heatmap (Tensor): Heatmap image (B, 1, H, W).
            img (Tensor): Real or generated image (B, 3, H, W).
        Returns:
            Patch-level prediction (real/fake).
        """
        x = torch.cat((segmap, heatmap, img), dim=1)  # [B, 5, H, W]
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator using multiple NLayerDiscriminators.
    """
    def __init__(self, 
                 input_nc=5,         # 1 (seg) + 1 (heatmap) + 3 (image)
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.InstanceNorm2d,
                 num_discriminators=3):
        super().__init__()
        self.num_discriminators = num_discriminators

        self.discriminators = nn.ModuleList()
        for _ in range(num_discriminators):
            disc = NLayerDiscriminator(
                input_nc=input_nc, 
                ndf=ndf, 
                n_layers=n_layers, 
                norm_layer=norm_layer
            )
            self.discriminators.append(disc)

    def forward(self, segmap, heatmap, img):
        """
        Args:
            segmap (Tensor): Segmentation map (B, 1, H, W).
            heatmap (Tensor): Heatmap (B, 1, H, W).
            img (Tensor): Generated or real RGB image (B, 3, H, W).
        Returns:
            List of patch predictions (1 per scale).
        """
        result = []
        input_downsampled = torch.cat((segmap, heatmap, img), dim=1)  # [B, 5, H, W]

        for i, disc in enumerate(self.discriminators):
            # Split back into components for each disc's forward
            seg_d = input_downsampled[:, 0:1]
            heat_d = input_downsampled[:, 1:2]
            img_d = input_downsampled[:, 2:5]
            out = disc(seg_d, heat_d, img_d)
            result.append(out)

            if i != self.num_discriminators - 1:
                input_downsampled = nn.functional.avg_pool2d(input_downsampled, kernel_size=2, stride=2, count_include_pad=False)

        return result