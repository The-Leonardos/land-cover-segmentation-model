import albumentations
import numpy as np
import random
import albumentations as alb
from rasterio.features import geometry_mask
import geopandas as gpd


class Preprocessing:
    def __init__(self, raster_transform=None, patch_size=256):
        self.raster_transform = raster_transform
        self.patch_size = patch_size
        self.transform = albumentations.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
        ])
        self.boundary = gpd.read_file('../data/bc_boundary/bc_boundary.shp')
        self.min_valid_ratio = 0.7

    def run(self, image, mask):
        # compute city mask
        _, h, w = image.shape
        city_mask = geometry_mask(
            [feature['geometry'] for feature in self.boundary.to_dict('records')],
            transform=self.raster_transform,
            invert=True,
            out_shape=(h, w),
        )

        # set labels outside the boundary to 255 (i.e. ignored labels)
        mask[~city_mask] = 255

        # obtain a random patch within city boundaries
        while True:
            image_patch, mask_patch, city_mask_patch = self.get_random_patch(image, mask, city_mask)

            city_overlap = city_mask_patch.sum() / (self.patch_size * self.patch_size)

            if city_overlap >= 0.7:
                break

        # augment images and mask with a 50%% chance of executing
        if random.randint(0, 1) == 1:
            image_patch, mask_patch = self.augment(image_patch, mask_patch)

        # convert nan values to 0.0
        image_patch = np.nan_to_num(image_patch, nan=0.0)

        return image_patch, mask_patch

    def get_random_patch(self, image, mask, city_mask):
        _, h, w = image.shape
        p_h = p_w = self.patch_size

        x = random.randint(0, h - p_h)
        y = random.randint(0, w - p_w)

        image_patch = image[:, x:x+p_h, y:y+p_w]
        mask_patch = mask[x:x + p_h, y:y + p_w]
        city_mask_patch = city_mask[x:x + p_h, y:y + p_w]

        return image_patch, mask_patch, city_mask_patch

    def augment(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]