from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MaskImageDataset(Dataset):
    """
    Dataset for Conditional GAN with optional heatmap.
    - Loads paired binary masks (1-channel), heatmaps (1-channel), and real images (3-channel).
    """

    def __init__(self, root_dir="DATA", img_size=256,
                 mask_subfolder="MASKS_DARK_200",
                 heatmap_subfolder="HEATMAPS_200",
                 image_subfolder="DARK_IMG_200"):
        self.root_dir = root_dir
        self.mask_dir = os.path.join(root_dir, mask_subfolder)
        self.heatmap_dir = os.path.join(root_dir, heatmap_subfolder)
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

        self.heatmap_paths = sorted([
            f for f in os.listdir(self.heatmap_dir)
            if os.path.isfile(os.path.join(self.heatmap_dir, f)) and os.path.splitext(f)[1].lower() in valid_ext
        ])

        assert len(self.mask_paths) == len(self.image_paths) == len(self.heatmap_paths), \
            "Mismatch in number of files between mask, image, and heatmap folders."

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.heatmap_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)

        # Load heatmap
        heatmap_path = os.path.join(self.heatmap_dir, self.heatmap_paths[idx])
        heatmap = Image.open(heatmap_path).convert("L")
        heatmap = self.heatmap_transform(heatmap)

        # Load image
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        return mask, heatmap, image
