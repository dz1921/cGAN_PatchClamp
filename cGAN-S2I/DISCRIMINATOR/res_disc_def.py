import torch
import torch.nn as nn
import functools


# NLayerDiscriminator (PatchGAN)
class NLayerDiscriminator(nn.Module):
    """
    A PatchGAN discriminator used in Pix2Pix/Pix2PixHD.
    It classifies overlapping patches of the image as real or fake.
    """
    def __init__(self, 
                 input_nc=3,     # number of input image channels (RGB)
                 ndf=64,         
                 n_layers=3,     
                 norm_layer=nn.InstanceNorm2d, 
                 use_sigmoid=False):
        """
        Args:
            input_nc (int): number of input channels.
            ndf (int): number of filters in the first convolution layer.
            n_layers (int): number of downsampling layers in the discriminator.
            norm_layer (nn.Module): normalisation layer to use (InstanceNorm2d or BatchNorm2d).
            use_sigmoid (bool): whether to apply a sigmoid at the output (common in older GANs).
        """
        super(NLayerDiscriminator, self).__init__()

        
        self.use_sigmoid = use_sigmoid
        use_bias = (norm_layer == nn.InstanceNorm2d)
        kw = 4       # kernel size
        padw = 1     # padding

        # Initial convolution (no normalisation at first layer)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # Downsampling layers
        nf = ndf  # current number of filters
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)  # clamp the max channels to 512
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

        # Output convolution layer
        sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """
        Forward pass of the NLayerDiscriminator.
        Returns a feature map (patch-level real/fake predictions).
        """
        return self.model(x)
    
    #Multiple discriminators
    class MultiScaleDiscriminator(nn.Module):
        def __init__(self, 
                    input_nc=3, 
                    ndf=64, 
                    n_layers=3, 
                    norm_layer=nn.InstanceNorm2d, 
                    use_sigmoid=False,
                    num_discriminators=3):
            """
            Args:
                input_nc (int): number of channels of the input image
                ndf (int): number of filters in the first conv layer
                n_layers (int): number of downsampling layers in each discriminator
                norm_layer (nn.Module): normaliation layer
                use_sigmoid (bool): whether to apply sigmoid in the last layer
                num_discriminators (int): how many discriminators (scales) to use
            """
            super().__init__()
            self.num_discriminators = num_discriminators

            # Build multiple NLayerDiscriminators
            self.discriminators = nn.ModuleList()
            for _ in range(num_discriminators):
                disc = NLayerDiscriminator(
                    input_nc=input_nc, 
                    ndf=ndf, 
                    n_layers=n_layers, 
                    norm_layer=norm_layer, 
                    use_sigmoid=use_sigmoid
                )
                self.discriminators.append(disc)

        def forward(self, x):
            """
            Forward pass of the MultiScaleDiscriminator.
            Args:
                x (Tensor): input image tensor of shape [B, C, H, W]
            Returns:
                List of patch-level predictions (one per scale).
            """
            result = []
            input_downsampled = x
            for i, disc in enumerate(self.discriminators):
                # Feed the current scale to the i-th discriminator
                out = disc(input_downsampled)
                result.append(out)

                # Downscale by factor of 2 for the next discriminator
                if i != self.num_discriminators - 1:
                    input_downsampled = nn.functional.avg_pool2d(input_downsampled, kernel_size=2, stride=2, count_include_pad=False)

            return result  # list of patch predictions, from highest resolution to lowest

