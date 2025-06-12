import torch
import torch.nn as nn

# NLayerDiscriminator (PatchGAN)
class NLayerDiscriminator(nn.Module):
    """
    A PatchGAN discriminator used for conditional GANs.
    It classifies overlapping patches of the (condition, image) pair as real or fake.
    """
    def __init__(self, 
                 input_nc=4,    # 1 (segmentation map) + 3 (RGB image)
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.InstanceNorm2d):
        """
        Args:
            input_nc (int): Number of input channels (segmentation + image).
            ndf (int): Number of filters in the first convolution layer.
            n_layers (int): Number of downsampling layers.
            norm_layer (nn.Module): Normalisation layer to use (InstanceNorm2d).
        """
        super(NLayerDiscriminator, self).__init__()

        # Use bias if normalisation is InstanceNorm2d
        use_bias = (norm_layer == nn.InstanceNorm2d)

        kw = 4       # kernel size
        padw = 1     # padding

        # Initial convolution (no normalisation at first layer)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # Downsampling layers
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)  # Clamp the number of channels to 512
            sequence += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layer without downsampling (to keep feature map relatively large)
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]

        # Output convolution layer (no sigmoid as BCEWithLogitsLoss externally applied)
        sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, segmap, img):
        """
        Forward pass of the NLayerDiscriminator.
        Args:
            segmap (Tensor): Segmentation map (B, 1, H, W).
            img (Tensor): Image (real or generated) (B, 3, H, W).
        Returns:
            Patch-level real/fake predictions.
        """
        
        x = torch.cat((segmap, img), dim=1)
        return self.model(x)

# Multi-Scale Discriminator
class MultiScaleDiscriminator(nn.Module):
    """
    A multi-scale discriminator that consists of multiple NLayerDiscriminators.
    """
    def __init__(self, 
                 input_nc=4,        # 1 (segmentation map) + 3 (RGB image)
                 ndf=64,            
                 n_layers=3,       
                 norm_layer=nn.InstanceNorm2d,
                 num_discriminators=3):  # number of discriminators (scales)
        super().__init__()
        self.num_discriminators = num_discriminators

        # Build multiple NLayerDiscriminators
        self.discriminators = nn.ModuleList()
        for _ in range(num_discriminators):
            disc = NLayerDiscriminator(
                input_nc=input_nc, 
                ndf=ndf, 
                n_layers=n_layers, 
                norm_layer=norm_layer
            )
            self.discriminators.append(disc)

    def forward(self, segmap, img):
        """
        Forward pass of the MultiScaleDiscriminator.
        Args:
            segmap (Tensor): Segmentation map (B, 1, H, W).
            img (Tensor): Image (real or generated) (B, 3, H, W).
        Returns:
            List of patch-level predictions from each discriminator.
        """
        result = []
        input_downsampled = torch.cat((segmap, img), dim=1)

        for i, disc in enumerate(self.discriminators):
            # Feed the current scale to the i-th discriminator
            out = disc(segmap, img)
            result.append(out)

            # Downscale the input by a factor of 2 for the next discriminator
            if i != self.num_discriminators - 1:
                input_downsampled = nn.functional.avg_pool2d(input_downsampled, kernel_size=2, stride=2, count_include_pad=False)

        return result  # list of patch predictions from each scale

