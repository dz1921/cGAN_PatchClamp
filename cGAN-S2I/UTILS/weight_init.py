import torch.nn as nn

def init_weights(net, init_type = "normal", init_gain=0.02):
    #INIT NETWORK WEIGHTS
    """
    net - The neural network model whose weights need to be initialised.
    init_type - Specifies the type of weight initialisation (like "normal", "xavier", "kaiming")
    init_gain - The standard deviation (std) for the normal distribution in weight initialisation (default: 0.02, as in Pix2Pix paper).
    """
    
    def init_func(m):
        """
        Applied to each of the layers in the network net and applies approproate weight initialisation
        based on its type.
        """
        classname = m.__class__.__name__
        print(f"Initializing: {classname}")
        if hasattr(m, 'weight') and m.weight is not None:
            print(f"Applying weight initialization to {classname}")

            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') 
            else:
                raise NotImplementedError(f"Init method [{init_type}] is not implemented")
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
        # Skip InstanceNorm2d if it has no affine parameters
        if isinstance(m, nn.InstanceNorm2d) and not m.affine:
            print(f"Skipping {classname} (no affine parameters)")
        # Handle BatchNorm2d or InstanceNorm2d when affine=True
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) and m.affine:
            nn.init.normal_(m.weight, mean=1.0, std=init_gain)
            nn.init.constant_(m.bias, 0)

    net.apply(init_func)