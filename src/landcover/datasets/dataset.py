import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from src.landcover.utils.data_preprocessing import Preprocessing


class LandCoverDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, train_mode=True, pre_load=True, minority_classes=None, seed=None):
        """
        Args:
            root_dir: Root directory (e.g., "data")
            patch_size: Size of patches
        """
        self.root_dir = Path(root_dir)
        self.pre_load = pre_load
        self.patch_size = patch_size
        self.train_mode = train_mode
        self.minority_classes = minority_classes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # get image and mask files
        self.image_files = sorted((self.root_dir / "images").glob("*.npy"))
        self.mask_files = sorted((self.root_dir / "masks").glob("*.npy"))
        assert(len(self.image_files) == len(self.mask_files)), "Images and masks count mismatch"

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                "Run run_data_cleaning.py first or check cleaned files."
            )

        if self.pre_load:
            self.images = [np.load(f) for f in self.image_files]
            self.masks = [np.load(f) for f in self.mask_files]

        self.preprocess = Preprocessing(
            patch_size=self.patch_size,
            minority_classes=self.minority_classes,
            seed=self.seed
        )

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        self.preprocess.patch_size = patch_size

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.preprocess.seed = seed
        self.preprocess.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.pre_load:
            image = self.images[idx].copy()
            mask = self.masks[idx].copy()
        else:
            image = np.load(self.image_files[idx])
            mask = np.load(self.mask_files[idx])

        if self.train_mode:
            image_patch, mask_patch = self.preprocess.run(image, mask)
        else:
            image_patch, mask_patch = image, mask.copy()

        image_patch = torch.from_numpy(image_patch).float()
        mask_patch = torch.from_numpy(mask_patch).long()

        return image_patch, mask_patch