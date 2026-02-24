import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio
import numpy as np
from src.landcover.utils.data_preprocessing import Preprocessing


class LandCoverDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_size=256, cache_images=True, use_numpy_cache=True):
        """
        Args:
            root_dir: Root directory (e.g., 'data')
            split: 'train', 'validation', or 'test'
            patch_size: Size of patches
            cache_images: Whether to cache in RAM
            use_numpy_cache: Whether to use NumPy file cache
        """
        self.root_dir = Path(root_dir)
        self.split = split

        # Path to the clean dataset based on split
        self.clean_dir = self.root_dir / 'dataset' / 'clean' / split

        # Get image and mask files
        self.image_files = sorted((self.clean_dir / 'images').glob('*.tif'))
        self.mask_files = sorted((self.clean_dir / 'masks').glob('*.tif'))

        # If no files in clean, try raw (fallback)
        if len(self.image_files) == 0:
            self.raw_dir = self.root_dir / 'dataset' / 'raw' / split
            self.image_files = sorted((self.raw_dir / 'images').glob('*.tif'))
            self.mask_files = sorted((self.raw_dir / 'masks').glob('*.tif'))

        assert len(self.image_files) > 0, f'No images found for split: {split}'
        assert len(self.image_files) == len(self.mask_files), 'Images and masks count mismatch'

        self.cache_images = cache_images
        self.use_numpy_cache = use_numpy_cache
        self.image_cache = {}
        self.mask_cache = {}

        # NumPy cache directory (separate for each split)
        self.numpy_cache_dir = Path('data/numpy_cache') / split
        if use_numpy_cache:
            self.numpy_cache_dir.mkdir(parents=True, exist_ok=True)
            self._convert_to_numpy_if_needed()

        # Load first image to get metadata (only once)
        with rio.open(self.image_files[0]) as src:
            crs = src.crs
            raster_transform = src.transform

        # Initialize preprocessing (caches city mask internally now)
        self.preprocess = Preprocessing(raster_transform, crs, patch_size)

        # Optional: Pre-cache all images if RAM allows
        if cache_images and not use_numpy_cache:  # Only if not using numpy cache
            print(f"Caching all {split} images and masks to RAM...")
            for i, (img_path, mask_path) in enumerate(zip(self.image_files, self.mask_files)):
                with rio.open(img_path) as src:
                    self.image_cache[i] = src.read()
                with rio.open(mask_path) as src:
                    self.mask_cache[i] = src.read(1)
            print(f"Caching complete for {split} split!")

    def _convert_to_numpy_if_needed(self):
        """Convert TIFF files to NumPy format for faster loading"""
        print(f"Checking/converting {self.split} TIFF to NumPy cache...")
        converted = 0

        for i, (img_path, mask_path) in enumerate(zip(self.image_files, self.mask_files)):
            # Check image numpy file
            img_npy = self.numpy_cache_dir / f"{img_path.stem}.npy"
            if not img_npy.exists():
                with rio.open(img_path) as src:
                    np.save(img_npy, src.read())
                converted += 1

            # Check mask numpy file
            mask_npy = self.numpy_cache_dir / f"{mask_path.stem}.npy"
            if not mask_npy.exists():
                with rio.open(mask_path) as src:
                    np.save(mask_npy, src.read(1))
                converted += 1

        if converted > 0:
            print(f"Converted {converted} files to NumPy format for {self.split}")
        else:
            print(f"All {self.split} files already in NumPy cache")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Priority: 1. RAM Cache, 2. NumPy Cache, 3. TIFF
        if self.cache_images and idx in self.image_cache:
            # RAM cache (fastest)
            image = self.image_cache[idx]
            mask = self.mask_cache[idx]

        elif self.use_numpy_cache:
            # NumPy file cache (faster than TIFF)
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            img_npy = self.numpy_cache_dir / f"{img_path.stem}.npy"
            mask_npy = self.numpy_cache_dir / f"{mask_path.stem}.npy"

            image = np.load(img_npy)
            mask = np.load(mask_npy)

            # Optionally cache to RAM
            if self.cache_images:
                self.image_cache[idx] = image
                self.mask_cache[idx] = mask

        else:
            # TIFF loading (slowest)
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            with rio.open(img_path) as src:
                image = src.read()

            with rio.open(mask_path) as src:
                mask = src.read(1)

            # Optionally cache this single image
            if self.cache_images:
                self.image_cache[idx] = image
                self.mask_cache[idx] = mask

        image_patch, mask_patch = self.preprocess.run(image, mask)

        image_patch = torch.tensor(image_patch, dtype=torch.float32)
        mask_patch = torch.tensor(mask_patch, dtype=torch.long)

        return image_patch, mask_patch