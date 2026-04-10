import albumentations
import numpy as np
import random
import albumentations as alb
from .. import DATA_PATH


class Preprocessing:
    def __init__(self, patch_size=256, minority_classes=None, seed=None):
        self.patch_size = patch_size
        self.minority_classes = minority_classes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.city_mask = np.load((DATA_PATH / "misc" / "city_mask.npy"))
        self.min_valid_ratio = 0.5

        self.transform = albumentations.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
            alb.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            alb.GaussNoise(p=0.2),
        ])


    def run(self, image, mask):
        # set labels outside the boundary to 255 (i.e. ignored labels)
        mask[~self.city_mask] = 255
        max_attempts = 10
        attempts = 0

        # obtain a random patch within city boundaries
        while True:
            image_patch, mask_patch, city_mask_patch, oversample_class = self._get_random_patch(image, mask)
            attempts += 1

            city_overlap = mask_patch != 255
            city_ratio = np.sum(city_overlap) / (self.patch_size * self.patch_size)
            if city_ratio < self.min_valid_ratio:
                continue

            if oversample_class is not None and random.random() < 0.6:
                minority_ratio = (mask_patch == oversample_class).sum() / (self.patch_size ** 2)
                if minority_ratio < 0.02 and attempts < max_attempts:
                    continue

            break

        if self.rng.integers(0, 2) == 1:
            image_patch, mask_patch = self._augment(image=image_patch, mask=mask_patch)

        # convert nan values to 0.0
        image_patch = np.nan_to_num(image_patch, nan=0.0)

        return image_patch, mask_patch

    def _get_random_patch(self, image, mask):
        _, h, w = image.shape
        p_h = p_w = self.patch_size

        if self.minority_classes is not None:
            oversample_class = self.rng.choice(self.minority_classes)
            coords = np.argwhere(mask == oversample_class)
            if len(coords) > 0:
                y_coord, x_coord = coords[self.rng.integers(0, len(coords) - 1)]

                y_start = min(max(0, y_coord - p_h // 2 + self.rng.integers(-p_h // 4, p_h // 4)), h - p_h)
                x_start = min(max(0, x_coord - p_w // 2 + self.rng.integers(-p_w // 4, p_w // 4)), w - p_w)
            else:
                oversample_class = None
                y_start = self.rng.integers(0, h - p_h)
                x_start = self.rng.integers(0, w - p_w)
        else:
            y_start = self.rng.integers(0, h - p_h)
            x_start = self.rng.integers(0, w - p_w)

        image_patch = image[:, y_start:y_start + p_h, x_start:x_start + p_w]
        mask_patch = mask[y_start:y_start + p_h, x_start:x_start + p_w]
        city_mask_patch = self.city_mask[y_start:y_start + p_h, x_start:x_start + p_w]

        return image_patch, mask_patch, city_mask_patch, oversample_class

    def _augment(self, image, mask):
        # transpose image from (C, H, W) to (H, W, C) for Albumentations
        image = np.transpose(image, (1, 2, 0))
        augmented = self.transform(image=image, mask=mask)

        # transpose image from (H, W, C) to (C, H, W) for PyTorch
        image = np.transpose(augmented["image"], (2, 0, 1))
        mask = augmented["mask"]

        return image, mask
