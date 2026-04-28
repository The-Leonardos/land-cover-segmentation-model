import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from landcover.utils.data_preprocessing import Preprocessing
from landcover import MINORITY_CLASSES


class LandCoverDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, train_mode=True, pre_load=True):
        """
        Args:
            root_dir: Root directory (e.g., "data")
            patch_size: Size of patches
        """
        self.root_dir = Path(root_dir)
        self.pre_load = pre_load
        self.patch_size = patch_size
        self.train_mode = train_mode
        self.minority_classes = MINORITY_CLASSES

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

        self.minority_coords = []
        for i in range(len(self.mask_files)):
            mask = self.masks[i] if self.pre_load else np.load(self.mask_files[i])
            coords = np.argwhere(np.isin(mask, MINORITY_CLASSES))
            self.minority_coords.append(coords)

        self.preprocess = Preprocessing(patch_size=self.patch_size)

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        self.preprocess.patch_size = patch_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        rejected_patches = 0

        if self.pre_load:
            image = self.images[idx]
            mask = self.masks[idx]
        else:
            image = np.load(self.image_files[idx])
            mask = np.load(self.mask_files[idx])

        coords = self.minority_coords[idx]
        if coords is None or len(coords) == 0:
            coords = None

        if self.train_mode:
            image_patch, mask_patch, rejected_patches = self.preprocess.run(image, mask, coords)
        else:
            image_patch, mask_patch = image, mask

        image_patch = torch.from_numpy(image_patch).float()
        mask_patch = torch.from_numpy(mask_patch).long()

        return image_patch, mask_patch, rejected_patches