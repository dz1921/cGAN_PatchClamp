cGAN_S2I/
├── CONFIG/                # YAML configs (hyperparameters)
├── UTILS/                 # Helpers: dataset loading, config parsing, metrics
├── GENERATOR/             # Generator model definitions
├── DISCRIMINATOR/         # Discriminator model definitions
├── REFINER/               # Refiner model definitions
├── TRAIN/                 # Training scripts
├── TEST/                  # Testing scripts
├── SCRIPTS/               # Sanity checks and data augmentation
└── requirements.txt

The main functions are present in TRAIN and TEST where training scripts and evaluation scripts are present so for using this code just run those scripts. In utility there are functions that are also standalone scripts like the 4GON_UI that could be used to prepare data for the training and testing but are not directly part of the main functions.

**U-Net**
The UNetGenerator is a symmetric encoder-decoder model based on the U-Net architecture. It takes a single-channel input image (e.g. binary mask or heatmap) and generates a 3-channel RGB output. The model uses downsampling via strided convolutions and upsampling via transposed convolutions, with skip connections between encoder and decoder layers to retain spatial detail.

Encoder (Downsampling)
  -The encoder consists of 8 convolutional blocks:

  -Each block includes: Conv2D -> InstanceNorm -> LeakyReLU (except the first block, which omits normalisation).

  -The spatial resolution is halved at each step.

Layer	Output shape	Notes
d1	(64, 128, 128)	No normalisation
d2	(128, 64, 64)	
d3	(256, 32, 32)	
d4	(512, 16, 16)	
d5	(512, 8, 8)	
d6	(512, 4, 4)	
d7	(512, 2, 2)	
d8	(512, 1, 1)	Bottleneck, no norm

Decoder (Upsampling)
  -The decoder mirrors the encoder with 7 transposed convolutional blocks:

  -Each block includes: ConvTranspose2D -> InstanceNorm -> ReLU.

  -Skip connections concatenate encoder outputs with decoder inputs at matching resolutions.

Layer	Output shape	Skip connection from
u1	(512, 2, 2)	d7
u2	(512, 4, 4)	d6
u3	(512, 8, 8)	d5
u4	(512, 16, 16)	d4
u5	(256, 32, 32)	d3
u6	(128, 64, 64)	d2
u7	(64, 128, 128)	d1

Final Output
  -A final ConvTranspose2D layer maps the last decoder output to a 3-channel image of shape (3, 256, 256).

A Tanh activation maps values to [-1, 1].

Input/Output
Input: (1, 256, 256) – 1-channel segmentation map or heatmap.

Output: (3, 256, 256) – RGB image with pixel values in [-1, 1].

This model is used as the generator within the cGAN training pipeline.


**ResNet**
The ResNetGenerator follows the Pix2PixHD-style ResNet-based generator design. It processes a 1-channel input (e.g. a segmentation mask or heatmap) into a 3-channel RGB image through a series of convolutional, residual, and upsampling layers.

Overview
The generator consists of the following stages:

Initial Convolution Block

  -A single 7×7 convolution layer with reflection padding.

  -Followed by instance normalisation and ReLU activation.

  -Output: (ngf, 256, 256) -> e.g. (64, 256, 256)

Downsampling

  -n_downsampling (default 3) strided 3×3 convolutions, each halving spatial dimensions and doubling channels.

  -Output progression:
  (64, 256, 256) -> (128, 128, 128) -> (256, 64, 64) -> (512, 32, 32)

ResNet Blocks

  -n_blocks (default 9) residual blocks with:
  
  -Reflection padding
  
  -3×3 convolutions
  
  -InstanceNorm + ReLU
  
  -Skip connections: output = input + residual

Upsampling

  -n_downsampling transposed convolutions, each doubling spatial size and halving channel count.
  
  -Output progression:
  (512, 32, 32) -> (256, 64, 64) -> (128, 128, 128) -> (64, 256, 256)

Final Convolution Block

 - A final 7×7 convolution with reflection padding maps features to 3 output channels.
  
  -A Tanh activation maps pixel values to the range [-1, 1].

Summary Table
Stage	Type	Output Shape (default ngf=64)
Initial conv	7×7 Conv + Norm + ReLU	(64, 256, 256)
Downsample 1	3×3 Strided Conv	(128, 128, 128)
Downsample 2	3×3 Strided Conv	(256, 64, 64)
Downsample 3	3×3 Strided Conv	(512, 32, 32)
ResBlocks (×9)	Residual conv blocks	(512, 32, 32)
Upsample 1	3×3 Transposed Conv	(256, 64, 64)
Upsample 2	3×3 Transposed Conv	(128, 128, 128)
Upsample 3	3×3 Transposed Conv	(64, 256, 256)
Final conv	7×7 Conv + Tanh	(3, 256, 256)

Key Characteristics
  -Skip Connections: Built into each ResNet block for better gradient flow and identity preservation.
  
  -Normalisation: Instance Normalisation is used throughout, consistent with image synthesis best practices.
  
  -Padding: Reflection padding avoids border artifacts and is used before all 3×3 convolutions.
  
  -This architecture is well-suited for high-quality image generation in cGAN-based pipelines such as Pix2PixHD.


**U-Net++**
The UNetPPGenerator is a deeply nested U-Net++-style generator designed for image-to-image translation tasks. It extends the classic U-Net with dense skip connections and multiple intermediate convolutional paths to improve gradient flow and feature fusion across scales.

Overview
  -The generator operates on a 1-channel 256×256 input (e.g. a pipette segmentation mask or heatmap) and produces a 3-        channel RGB output of the same resolution. It consists of:

Encoder Path

  A series of 5 convolutional blocks with max-pooling to progressively downsample the input and increase feature dimensionality:
  
  (1, 256, 256) -> (64, 256, 256)
  
  -> (128, 128, 128)
  
  -> (256, 64, 64)
  
  -> (512, 32, 32)
  
  -> (1024, 16, 16)

Decoder Path (Nested Dense Skip Connections)

  -Each decoder level builds a series of nested convolutional blocks that merge:

    -Feature maps from the encoder (at the same depth),
    
    -Upsampled features from deeper layers,
    
    -Previous intermediate outputs from earlier decoding stages (nested levels).

-This enables richer contextual blending and multiscale refinement at every level.

Final Output Layer

  -A 1×1 convolution maps the last decoder feature map to 3 channels.
  
  -A Tanh activation maps values to the range [-1, 1] for image generation.

Summary Table
Stage	  Description	Output   Shape
Input	    Raw 1-channel input	  (1, 256, 256)
Encoder	  5 ConvBlocks with MaxPool	  down to (1024, 16, 16)
Decoder	  4 nested levels of upsampling paths	  (64, 256, 256)
Output	  Final 1×1 Conv + Tanh	  (3, 256, 256)

Key Characteristics
  ConvBlock: Each block consists of 2× 3×3 convolutions with InstanceNorm, ReLU, and Dropout (default 0.3).
  
  UpBlock: Transposed convolution that doubles spatial dimensions for upsampling.
  
  Nested Dense Paths: Intermediate nodes like x01, x02, ..., x04 allow for reusing and refining features multiple times.
  
  Deep Supervision Friendly: Structure allows easy extension to deep supervision if needed.
  
  Output Quality: The dense connectivity improves information flow and helps refine spatial detail in high-resolution generation.
  
  This architecture is ideal for applications requiring fine structural fidelity and spatially-aware reconstruction, like synthesising microscopy images from segmentation masks.


**ResNet Refiner**
The ResNetBackgroundRefiner is a post-processing network used to refine GAN-generated images, especially to mitigate checkerboard artifacts, blending errors, and spatial noise in the background regions.

This module is designed to take a concatenation of the generated image and its corresponding mask (or heatmap)—typically a 4-channel tensor—and produce a clean, final RGB image via deep residual refinement.

Architecture Overview
  Input: 4×256×256 (e.g. 3-channel GAN output + 1-channel mask or heatmap)
  
  Output: 3×256×256 (refined RGB image)

The architecture consists of:

  Initial Feature Extraction (head):

    -A 7×7 convolution followed by Instance Normalisation and ReLU.
    
    -Expands the input into a deeper feature representation.

  Residual Refinement Bottleneck:

    -A configurable number (default=6) of ResBlock modules.
  
    -Each ResBlock applies two 3×3 convolutions with InstanceNorm and ReLU.

    -Optional Coordinate Attention is applied to capture long-range spatial dependencies along both height and width axes       without losing positional granularity.

  Output Projection (tail)

    -A final 3×3 convolution maps the feature map to 3 channels.
    
    -Followed by Tanh() to normalize pixel values into the range [-1, 1].

Key Features
  Component	  Description
  CoordAttention	              Captures height-wise and width-wise global context via separate pooling, then modulates                                    features with spatially aware attention maps.
  Residual Blocks	              Maintain spatial integrity while enabling deep nonlinear refinement.
  No Downsampling/Upsampling	  Operates at the original image resolution to preserve fine details and avoid aliasing.
  Lightweight	                  Shallow architecture designed to be fast and effective as a post-GAN cleanup step.

Summary Table
Stage	Layers	Output Shape
Input	GAN image + mask	(4, 256, 256)
Head	7×7 Conv + InstanceNorm + ReLU	(64, 256, 256)
Bottleneck	6× ResBlocks (w/ CoordAttention)	(64, 256, 256)
Tail	3×3 Conv + Tanh	(3, 256, 256)





  **CBAM in U-Net**
  In this version, CBAM modules are applied directly to the concatenated skip connections in the decoder path of a U-Net     generator. After each upsampling operation, the output tensor is concatenated with the corresponding encoder feature       map, and this merged tensor is passed through a CBAM module to enhance it via attention before proceeding to the next      upsampling step.

Specifics:

  -CBAM is used in 7 skip connections, placed between upsampled outputs and encoder feature maps.

  -Each CBAM takes a tensor with 2×F channels (where F is the number of channels from the encoder and decoder branches respectively).

  -Both channel and spatial attention are applied sequentially in each CBAM block.

  -CBAM is not applied within convolutional layers, only after concatenation of skip features.

  **CBAM in ResNet**
  In this version, CBAM is embedded inside each ResNet block in the bottleneck of the generator. Specifically, it is applied after the residual path’s convolutions, just before the residual addition.

Specifics:

  -CBAM is used inside every ResnetCBAMBlock in the ResNet-style bottleneck (typically 9 blocks).
  
  -The ResNet blocks follow a standard residual pattern: conv → norm → ReLU → conv → norm → CBAM → residual addition.
  
  -Channel and spatial attention refine the residual before it is added back to the input (pre-activation residual).
  
  -CBAM receives tensors with fixed channel width equal to the feature map width of the block (typically ngf × 2^n_downsampling).

  **CBAM in U-Net++**
  Implementation summary:
  This version integrates CBAM inside every ConvBlock, meaning attention is applied after each local double-convolution unit across both the encoder and decoder paths. It is thus densely embedded throughout the network, including the core U-Net++ skip pathways.
  
  Specifics:
  
    -CBAM is applied to every node in the U-Net++ grid (i.e. convXY units).
    
    -It follows two convolutional layers (with InstanceNorm, ReLU, and dropout) and is applied before the output is forwarded or concatenated.
    
    -CBAM acts as a post-processing refinement for each feature block, making it more pervasive than in the previous two designs.
    
    -Since UNet++ contains dense skip connections between decoder branches, this results in a large number of CBAM modules enhancing feature fusion granularity.

| Parameter                     | UNetCBAMGenerator (#1)          | CBAMResNetGenerator (#2)        | UNetPPGenerator (#3)                                    |
| ----------------------------- | ------------------------------- | ------------------------------- | ------------------------------------------------------- |
| **Channel reduction ratio**   | `reduction=16`                  | `ratio=16`                      | `reduction=16`                                          |
| **Channel attention pooling** | `avg + max pooled -> shared MLP` | `avg + max pooled → shared MLP` | `avg + max pooled -> separate MLP` summed **before** MLP |
| **Spatial attention kernel**  | `kernel_size=7`                 | `kernel_size=7` (default)       | `kernel_size=7`                                         |
| **Spatial attention input**   | `cat(avg, max)`                 | `cat(avg, max)`                 | `cat(avg, max)`                                         |
| **Channel MLP reuse**         | Shared for avg and max          | Shared                          |  Combined before pass                |
| **Activation**                | `ReLU + Sigmoid`                | `ReLU + Sigmoid`                | `ReLU + Sigmoid`                                        |
| **Attention application**     | `x * CA(x) -> x * SA(x)`         | `x * CA(x) -> x * SA(x)`         | `x * CA(x) -> x * SA(x)`                                 |


**SPADE in U-net**
  Architecture: Traditional U-Net with downsampling path and upsampling decoder with skip connections.
  
  Integration Point: SPADE is only used in the upsampling path (decoder), where each UpBlock replaces standard normalization with SPADE.
  
  Mechanism: Down blocks use InstanceNorm. Up blocks perform a transposed convolution, followed by SPADE normalization and ReLU. The input segmentation map is passed to each SPADE module.
  
  SPADE Usage: Only in decoder (upsampling path).

**SPADE in ResNet**
  Architecture: Encoder–ResNet-style bottleneck–decoder.
  
  Integration Point: SPADE is only used in the bottleneck via SPADEResnetBlock, a residual block where both normalization layers are replaced by SPADE.
  
  Mechanism: Each residual block performs:
  SPADE → ReLU → Conv → SPADE → ReLU → Conv → residual + input.
  
  Downsampling/Upsampling: Uses regular InstanceNorm (not SPADE).
  
  SPADE Usage: Only in bottleneck (residual blocks).

**SPADE in U-Net++**
  Architecture: Deep multi-path decoder with nested skip connections.
  
  Integration Point: SPADE is used throughout all convolutional blocks — including the encoder and all decoder blocks — via SPADEConvBlock, which replaces standard instance normalization.
  
  Mechanism: Each SPADEConvBlock consists of two convolutions, each followed by SPADE normalization (with shared label_nc), dropout, and ReLU.
  
  Input Segmap: Always provided to every block.
  
  SPADE Usage: Fully SPADE-based — both encoder and decoder.



| Parameter / Feature             | **UNet++ SPADE**                     | **SPADE-UNet**                 | **SPADE-ResNet**                |
| ------------------------------- | ------------------------------------ | ------------------------------ | ------------------------------- |
| **Param-free norm type**        | `InstanceNorm2d(affine=False)`       | `InstanceNorm2d(affine=False)` | `InstanceNorm2d(affine=False)`  |
| **Hidden channels (`nhidden`)** | 128                                  | 128                            | 128                             |
| **Shared MLP structure**        | Conv2d(1 → 128) + ReLU               | Conv2d(1 → 128) + ReLU         | Conv2d(1 → 128) + ReLU          |
| **Gamma branch**                | Conv2d(128 → norm\_nc)               | Conv2d(128 → norm\_nc)         | Conv2d(128 → norm\_nc)          |
| **Beta branch**                 | Conv2d(128 → norm\_nc)               | Conv2d(128 → norm\_nc)         | Conv2d(128 → norm\_nc)          |
| **Modulation equation**         | `(x̂ * (1 + γ) + β)`                 | Same                           | Same                            |
| **Segmap resizing**             | `F.interpolate(..., mode='nearest')` | Same                           | Same                            |
| **Used in layers**              | All conv blocks (encoder & decoder)  | Decoder (upsampling only)      | Bottleneck residual blocks only |




  

