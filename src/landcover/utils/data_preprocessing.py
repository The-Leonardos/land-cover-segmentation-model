import albumentations
import numpy as np
import random
import albumentations as alb
from .. import DATA_PATH


class Preprocessing:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size
        self.city_mask = np.load((DATA_PATH / 'city_mask.npy'))
        self.transform = albumentations.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
        ])
        self.min_valid_ratio = 0.7

    def run(self, image, mask):
        # set labels outside the boundary to 255 (i.e. ignored labels)
        mask[~self.city_mask] = 255

        # obtain a random patch within city boundaries
        while True:
            image_patch, mask_patch, city_mask_patch = self.get_random_patch(image, mask)

            city_overlap = city_mask_patch.sum() / (self.patch_size * self.patch_size)

            if city_overlap >= 0.7:
                break

        # augment images and mask with a 50%% chance of executing
        if random.randint(0, 1) == 1:
            image_patch, mask_patch = self.augment(image_patch, mask_patch)

        # convert nan values to 0.0
        image_patch = np.nan_to_num(image_patch, nan=0.0)

        return image_patch, mask_patch

    def get_random_patch(self, image, mask):
        _, h, w = image.shape
        p_h = p_w = self.patch_size

        x = random.randint(0, h - p_h)
        y = random.randint(0, w - p_w)

        image_patch = image[:, x:x + p_h, y:y + p_w]
        mask_patch = mask[x:x + p_h, y:y + p_w]
        city_mask_patch = self.city_mask[x:x + p_h, y:y + p_w]

        return image_patch, mask_patch, city_mask_patch

    def augment(self, image, mask):
        # transpose image from (C, H, W) to (H, W, C) for Albumentations
        image = np.transpose(image, (1, 2, 0))
        augmented = self.transform(image=image, mask=mask)

        # transpose image from (H, W, C) to (C, H, W) for PyTorch
        image = np.transpose(augmented['image'], (2, 0, 1))
        mask = augmented['mask']

        return image, mask
