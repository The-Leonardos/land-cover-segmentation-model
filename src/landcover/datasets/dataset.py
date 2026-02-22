import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio
from src.landcover.utils.data_preprocessing import Preprocessing

class LandCoverDataset(Dataset):
    def __init__(self, root_dir, patch_size=256):
        self.root_dir = Path(root_dir)
        self.image_files = sorted((self.root_dir / 'images').glob('*.tif'))
        self.mask_files = sorted((self.root_dir / 'masks').glob('*.tif'))
        assert(len(self.image_files) == len(self.mask_files)), 'Images and masks count mismatch'

        with rio.open(self.image_files[0]) as src:
            crs = src.crs
            raster_transform = src.transform
        self.preprocess = Preprocessing(raster_transform, crs, patch_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        with rio.open(img_path) as src:
            # image = src.read([1, 2, 3])
            image = src.read()

        with rio.open(mask_path) as src:
            mask = src.read(1)

        image_patch, mask_patch = self.preprocess.run(image, mask)

        image_patch = torch.tensor(image_patch, dtype=torch.float32)
        mask_patch = torch.tensor(mask_patch, dtype=torch.long)

        return image_patch, mask_patch