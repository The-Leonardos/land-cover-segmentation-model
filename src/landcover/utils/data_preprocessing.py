import albumentations
import numpy as np
import random
import albumentations as alb
from .. import DATA_PATH


class Preprocessing:
    def __init__(self, patch_size=256, minority_classes=None):
        self.patch_size = patch_size
        self.city_mask = np.load((DATA_PATH / "misc" / "city_mask.npy"))
        self.transform = albumentations.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
        ])
        self.min_valid_ratio = 0.7
        self.minority_classes = minority_classes

    def run(self, image, mask):
        # set labels outside the boundary to 255 (i.e. ignored labels)
        mask[~self.city_mask] = 255

        max_attempts = 10
        attempts = 0

        # obtain a random patch within city boundaries
        while True:
            image_patch, mask_patch, city_mask_patch, oversample_class = self._get_random_patch(image, mask)
            attempts += 1

            city_overlap = city_mask_patch.sum() / (self.patch_size * self.patch_size)

            if city_overlap < 0.7:
                continue

            if oversample_class is not None:
                minority_ratio = (mask_patch == oversample_class).sum() / (self.patch_size ** 2)

                if minority_ratio < 0.05 and attempts < max_attempts:
                    continue

            break

        # augment images and mask with a 50%% chance of executing
        if random.randint(0, 1) == 1:
            image_patch, mask_patch = self._augment(image_patch, mask_patch)

        # convert nan values to 0.0
        image_patch = np.nan_to_num(image_patch, nan=0.0)

        return image_patch, mask_patch

    def _get_random_patch(self, image, mask):
        _, h, w = image.shape
        p_h = p_w = self.patch_size

        if self.minority_classes is not None:
            oversample_class = random.choice(self.minority_classes)

            coords = np.argwhere(mask == oversample_class)

            if len(coords) > 0:
                x, y = coords[random.randint(0, len(coords)-1)]

                x = min(max(0, x - p_h // 2), h - p_h)
                y = min(max(0, y - p_w // 2), w - p_w)
            else:
                oversample_class = None
                x = random.randint(0, h - p_h)
                y = random.randint(0, w - p_w)
        else:
            x = random.randint(0, h - p_h)
            y = random.randint(0, w - p_w)

        image_patch = image[:, x:x + p_h, y:y + p_w]
        mask_patch = mask[x:x + p_h, y:y + p_w]
        city_mask_patch = self.city_mask[x:x + p_h, y:y + p_w]

        return image_patch, mask_patch, city_mask_patch, oversample_class

    def _augment(self, image, mask):
        # transpose image from (C, H, W) to (H, W, C) for Albumentations
        image = np.transpose(image, (1, 2, 0))
        augmented = self.transform(image=image, mask=mask)

        # transpose image from (H, W, C) to (C, H, W) for PyTorch
        image = np.transpose(augmented["image"], (2, 0, 1))
        mask = augmented["mask"]

        return image, mask
