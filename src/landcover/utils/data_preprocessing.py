import albumentations
import numpy as np
import albumentations as alb
from .. import DATA_PATH
from landcover.utils import crop
from landcover import PATCH_SAMPLING_PROBS, AUGMENTATION_PROBS, MINORITY_CLASSES, MIN_VALID_RATIO, IGNORE_INDEX


class Preprocessing:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

        self.city_mask = np.load((DATA_PATH / "misc" / "city_mask.npy"))
        self.transform = albumentations.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
        ])

    def run(self, image, mask):
        mask = mask.copy()
        mask[~self.city_mask] = IGNORE_INDEX

        mode = np.random.choice(
            list(PATCH_SAMPLING_PROBS.keys()),
            p=list(PATCH_SAMPLING_PROBS.values()),
        )

        image_patch, mask_patch = self._get_random_patch(image, mask, mode)

        if np.random.rand() < AUGMENTATION_PROBS:
            image_patch, mask_patch = self._augment(image_patch, mask_patch)

        image_patch = np.nan_to_num(image_patch, nan=0.0)

        return image_patch, mask_patch

    def _get_random_patch(self, image, mask, mode):
        h = image.shape[-2]
        w = image.shape[-1]
        p = self.patch_size

        for _ in range(50):
            if mode == "minority":
                image_patch, mask_patch = self._sample_minority_patch(image, mask, h, w, p)
            else:
                image_patch, mask_patch = self._sample_random_patch(image, mask, h, w, p)

            valid = (mask_patch != IGNORE_INDEX)
            if valid.mean() >= MIN_VALID_RATIO:
                return image_patch, mask_patch

        return image_patch, mask_patch

    def _sample_minority_patch(self, image, mask, h, w, p):
        coords = np.argwhere(np.isin(mask, MINORITY_CLASSES))

        if len(coords) == 0:
            return self._sample_random_patch(image, mask, h, w, p)

        # randomly pick a minority pixel
        y, x = coords[np.random.randint(len(coords))]

        # add jitter
        y_start = np.clip(
            y - p // 2 + np.random.randint(-p // 4, p // 4),
            0, h - p
        )
        x_start = np.clip(
            x - p // 2 + np.random.randint(-p // 4, p // 4),
            0, w - p
        )

        return crop(image, mask, y_start, x_start, p)

    @staticmethod
    def _sample_random_patch(image, mask, h, w, p):
        y = np.random.randint(0, h - p)
        x = np.random.randint(0, w - p)

        return crop(image, mask, y, x, p)

    def _augment(self, image, mask):
        image = np.transpose(image, (1, 2, 0))
        augmented = self.transform(image=image, mask=mask)

        image = np.transpose(augmented["image"], (2, 0, 1))
        mask = augmented["mask"]

        return image, mask
