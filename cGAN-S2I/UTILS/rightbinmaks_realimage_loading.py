from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class MaskImageDataset(Dataset):
    """
    Loads paired binary masks (1-channel) and real images (3-channel) for cGANs.
    Applies binarisation to masks before converting to tensor.
    """

    def __init__(self, root_dir="DATA", img_size=256,
                 mask_subfolder="MASKS_DARK_200", image_subfolder="DARK_IMG_200"):

        self.mask_dir = os.path.join(root_dir, mask_subfolder)
        self.image_dir = os.path.join(root_dir, image_subfolder)
        self.img_size = img_size

        valid_ext = [".png", ".jpg", ".jpeg", ".tif"]
        self.mask_paths = sorted([
            f for f in os.listdir(self.mask_dir)
            if os.path.isfile(os.path.join(self.mask_dir, f)) and os.path.splitext(f)[1].lower() in valid_ext
        ])
        self.image_paths = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, f)) and os.path.splitext(f)[1].lower() in valid_ext
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # maps 0 → -1, 1 → +1
        ])

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
        image_path = os.path.join(self.image_dir, self.image_paths[idx])

        # Binarise mask BEFORE ToTensor
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda x: 255 if x > 0 else 0)

        mask = self.mask_transform(mask)
        image = self.image_transform(Image.open(image_path).convert("RGB"))

        return mask, image
