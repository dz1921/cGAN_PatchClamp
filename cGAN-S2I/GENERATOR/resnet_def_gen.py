import torch
import torch.nn as nn


# Helper: Padding choice function (Pix2PixHD typically uses reflection padding)
def get_padding_layer(padding_type):
    if padding_type == 'reflect':
        return nn.ReflectionPad2d
    elif padding_type == 'replicate':
        return nn.ReplicationPad2d
    elif padding_type == 'zero':
        # ZeroPad2d expects a single integer or a tuple of 4 ints (left, right, top, bottom)
        return nn.ZeroPad2d
    else:
        raise NotImplementedError(f"Padding type [{padding_type}] is not recognized.")


# ResnetBlock (Pix2PixHD style)
class ResnetBlock(nn.Module):
    """
    ResNet block:
      1) reflection (or other) padding
      2) convolution 3x3
      3) instance normalisation
      4) ReLU
      5) reflection (or other) padding
      6) convolution 3x3
      7) instance normalisation
      8) skip connection
    """
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_bias=False):
        """
        Args:
            dim (int): Number of input and output channels.
            padding_type (str): Type of padding ('reflect').
            norm_layer (nn.Module): Normalisation layer ( nn.InstanceNorm2d).
            use_bias (bool): Whether conv layers use bias. False if using a norm layer.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        # First padding + conv + norm + ReLU
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            pass
        else:
            raise NotImplementedError(f"Padding type [{padding_type}] is not recognized.")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=(0 if padding_type != 'zero' else 1), bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        # Second padding + conv + norm
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            pass

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=(0 if padding_type != 'zero' else 1), bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)  # Skip connection


# ResNetGenerator (Pix2PixHD style)
class ResNetGenerator(nn.Module):
    """
    A ResNet-based generator architecture based on Pix2PixHD:
      1) Initial 7x7 convolution block
      2) n_downsampling times downsampling
      3) n_blocks times ResNet blocks
      4) n_downsampling times upsampling
      5) Final 7x7 convolution + Tanh
    """
    def __init__(self, 
                 input_nc=1, 
                 output_nc=3, 
                 ngf=64, 
                 n_downsampling=3, 
                 n_blocks=9, 
                 norm_layer=nn.InstanceNorm2d, 
                 padding_type='reflect'):
        """
        Args:
            input_nc (int): # of channels in input images.
            output_nc (int): # of channels in output images.
            ngf (int): # of filters in the first conv layer.
            n_downsampling (int): How many times to downsample the image by factor of 2.
            n_blocks (int): How many ResNet blocks.
            norm_layer (nn.Module): Normalisation layer (nn.InstanceNorm2d).
            padding_type (str): Padding type for ResNet blocks ('reflect').
        """
        super(ResNetGenerator, self).__init__()

        
        use_bias = (norm_layer == nn.InstanceNorm2d)

        
        activation = nn.ReLU(True)

        # Initial 7x7 convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
            norm_layer(ngf),
            activation
        ]

        # Downsampling
        # Each downsampling step doubles the number of filters
        for i in range(n_downsampling):
            in_ch = ngf * (2 ** i)
            out_ch = ngf * (2 ** (i + 1))
            model += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(out_ch),
                activation
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(mult * ngf, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias)]

        # Upsampling
        # Each upsampling step halves the number of filters
        for i in range(n_downsampling):
            in_ch = ngf * (2 ** (n_downsampling - i))
            out_ch = in_ch // 2
            model += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=use_bias),
                norm_layer(out_ch),
                activation
            ]

        # Final 7x7 convolution + Tanh
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

