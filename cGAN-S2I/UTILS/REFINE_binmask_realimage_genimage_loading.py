from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class MaskImageDataset(Dataset):
    """
    Loads triplets of (mask, real image, generated image) for background refiner training.
    Applies binarisation to mask and normalisation to all images.
    """

    def __init__(self, root_dir="DATA", img_size=256,
                 mask_subfolder="MASKS_DARK_200",
                 image_subfolder="DARK_IMG_200",
                 gen_subfolder="DARK_GEN_200"):

        self.mask_dir = os.path.join(root_dir, mask_subfolder)
        self.image_dir = os.path.join(root_dir, image_subfolder)
        self.gen_dir = os.path.join(root_dir, gen_subfolder)
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
        self.gen_paths = sorted([
            f for f in os.listdir(self.gen_dir)
            if os.path.isfile(os.path.join(self.gen_dir, f)) and os.path.splitext(f)[1].lower() in valid_ext
        ])

        assert len(self.mask_paths) == len(self.image_paths) == len(self.gen_paths), \
            "Mismatch in number of files between mask, image, and generated folders"

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 0 → -1, 1 → +1
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
        gen_path = os.path.join(self.gen_dir, self.gen_paths[idx])

        # Mask: binarise then normalise
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda x: 255 if x > 0 else 0)
        mask = self.mask_transform(mask)

        image = self.image_transform(Image.open(image_path).convert("RGB"))
        gen_image = self.image_transform(Image.open(gen_path).convert("RGB"))

        return mask, image, gen_image