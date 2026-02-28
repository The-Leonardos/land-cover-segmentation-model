import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from src.landcover.utils.data_preprocessing import Preprocessing


class LandCoverDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, pre_load=True):
        """
        Args:
            root_dir: Root directory (e.g., 'data')
            patch_size: Size of patches
        """
        self.root_dir = Path(root_dir)
        self.pre_load = pre_load

        # get image and mask files
        self.image_files = sorted((self.root_dir / 'images').glob('*.npy'))
        self.mask_files = sorted((self.root_dir / 'masks').glob('*.npy'))
        assert(len(self.image_files) == len(self.mask_files)), 'Images and masks count mismatch'

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                "Run run_data_cleaning.py first or check cleaned files."
            )

        if self.pre_load:
            self.images = [np.load(f) for f in self.image_files]
            self.masks = [np.load(f) for f in self.mask_files]

        self.preprocess = Preprocessing(patch_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.pre_load:
            image = self.images[idx]
            mask = self.masks[idx]
        else:
            image = np.load(self.image_files[idx])
            mask = np.load(self.mask_files[idx])

        image_patch, mask_patch = self.preprocess.run(image, mask)

        image_patch = torch.tensor(image_patch, dtype=torch.float32)
        mask_patch = torch.tensor(mask_patch, dtype=torch.long)

        return image_patch, mask_patch