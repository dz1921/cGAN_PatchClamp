import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm import tqdm
from GENERATOR.gen_def_UNETPP import UNetPPGenerator

# Denormalisation: [-1, 1] -> [0, 255] uint8
def denorm_to_uint8(t):
    t = (t.clamp(-1, 1) + 1) * 127.5
    return t.to(torch.uint8)

# Mask preprocessing: Resize + Binarise + Normalise [-1, 1]
def preprocess_mask(mask_path, img_size=256):
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask = mask.point(lambda x: 255 if x > 0 else 0)  # Binarise
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # -> [-1, 1]
    ])
    return transform(mask)

# Main generation function
def generate_images_from_masks(model_path, input_folder, output_folder, img_size=256):
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load trained model
    model = UNetPPGenerator(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mask_fnames = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ])

    print(f"[INFO] Found {len(mask_fnames)} mask images.")

    for fname in tqdm(mask_fnames, desc="Generating"):
        mask_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        mask_tensor = preprocess_mask(mask_path, img_size).unsqueeze(0).to(device)  # [1,1,H,W]

        with torch.no_grad():
            output = model(mask_tensor)                    # [1,3,H,W], values in [-1, 1]
            output_uint8 = denorm_to_uint8(output.squeeze(0))  # [3,H,W], uint8 in [0,255]

        to_pil_image(output_uint8).save(out_path)

    print(f"[DONE] All images saved to: {output_folder}")

#PATHS and execution
model_path = r"MODELS\TRAINED_TOGETHER\GENERATORS\LEARNING_RATE_TUNING\UNETPP_LIGHT_LPIPSTV_l5_50_MLP\generator_lr_0_0005.pth"
input_folder = r"DATA\LIGHT_MASKS\VALIDATION_SET"
output_folder = r"DATA\GEN_IMAGES\UNETPP_LIGHT_LPIPSTV_l5_50_MLP_lr_0_0005\VALIDATION_SET"

generate_images_from_masks(model_path, input_folder, output_folder)
