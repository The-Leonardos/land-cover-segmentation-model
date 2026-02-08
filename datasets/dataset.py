import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio

class LandCoverDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.image_files = sorted((self.root_dir / 'images').glob('*.tiff'))
        self.mask_files = sorted((self.root_dir / 'masks').glob('*.tiff'))

        assert(len(self.image_files) == len(self.mask_files)), 'Images and masks count mismatch'

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        with rio.open(img_path) as src:
            image = src.read()

        with rio.open(mask_path) as src:
            mask = src.read(1)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask