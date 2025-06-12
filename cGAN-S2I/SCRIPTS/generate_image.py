import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
#from GENERATOR.resnet_def_gen import ResNetGenerator
from GENERATOR.generator_def_RGB_BIN import UNetGenerator

def generate_fake_image(
    mask_path: str,
    checkpoint_path: str,
    output_path: str,
    in_channels: int = 1,
    out_channels: int = 3,
    image_size: int = 256,
    device: str = "cpu"
):
    """
    Loads a specified checkpoint of a generator (UNetGenerator) and generates
    a fake image from a specified binary mask

    Args:
        mask_path (str): Path to the binary mask image (grayscale)
        checkpoint_path (str): Path to the generator checkpoint (.pth file)
        output_path (str): File path to save the generated image
        in_channels (int): Number of input channels for the generator
                                     Defaults to 1 (grayscale mask)
        out_channels (int): Number of output channels for the generator
                                      Defaults to 3 (RGB)
        image_size (int): The size (width, height) to resize images
                                    Defaults to 256
        device (str): "cpu" or "cuda", Defaults to "cpu"
    """

    #LOAD GENERATOR
    generator = UNetGenerator(in_channels=in_channels, out_channels=out_channels)
    #generator = ResNetGenerator(input_nc=in_channels, output_nc=out_channels)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()  # set generator to evaluation mode

   #PREPROCESSING
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # For 1-channel
    ])

    #MASK LOADING
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found at: {mask_path}")
    
    mask_img = Image.open(mask_path).convert("L")  # Ensure grayscale
    mask_tensor = mask_transform(mask_img).unsqueeze(0).to(device)  # (1, 1, H, W)

    #GENERATE IMAGE
    with torch.no_grad():
        fake_tensor = generator(mask_tensor)  # (1, 3, H, W)
    # fake_tensor in [-1, 1] range due to Tanh

    #POST-PROCESSING
    # Denormalise: [-1, 1] -> [0, 1]
    fake_tensor = (fake_tensor * 0.5) + 0.5  # moves from [-1,1] to [0,1]
    print("Generated tensor shape:", fake_tensor.shape)

    # Remove batch dimension and convert to CPU for saving
    fake_tensor = fake_tensor.squeeze(0).cpu()  # shape: (3, H, W)
    print("Generated tensor shape:", fake_tensor.shape)

    # Convert to numpy and check of type of output
    fake_tensor = fake_tensor.squeeze(0).cpu()
    if out_channels == 3:  # RGB Output
        fake_img_np = np.transpose(fake_tensor, (1, 2, 0))  # Convert (3, H, W) - (H, W, 3)
        fake_img_np = fake_img_np.cpu().numpy()
    elif out_channels == 1:  # Grayscale Output
        fake_img_np = fake_tensor.cpu().numpy() # Convert (1, H, W) - (H, W)
    print("Fake image shape:", fake_img_np.shape)
    fake_img_np = (fake_img_np * 255.0).astype(np.uint8)  # [0,1] -> [0,255]
    print("Fake image shape:", fake_img_np.shape)

    # Create PIL image and save
    fake_img_pil = Image.fromarray(fake_img_np)
    mask_filename = os.path.basename(mask_path)
    output_dir = output_path
    output_path = os.path.join(output_dir, mask_filename)
    fake_img_pil.save(output_path)
    print(f"Generated image saved to: {output_path}")

if __name__ == "__main__":
    generate_fake_image(r"DATA\MASK_DATASET\annotated_Capture_23_08_31_at_21_15_21_combined_labels.png",r"MODELS\TRAINED_TOGETHER\GENERATORS\generator_final.pth",r"DATA\GENERATED_IMAGES")

