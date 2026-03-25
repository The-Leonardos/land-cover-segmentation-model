import numpy as np


class PatchSampler:
    def __init__(self, masks, min_class_pixels=50):
        self.masks = masks
        self.class_locations = {}

        for idx, mask in enumerate(masks):
            for class_id in range(9):
                coords = np.argwhere(mask == class_id)
                if len(coords) >= min_class_pixels:
                    if class_id not in self.class_locations:
                        self.class_locations[class_id] = []
                    for x, y in coords:
                        self.class_locations[class_id].append((idx, x, y))