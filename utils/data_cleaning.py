"""
Data Cleaning Module for Land Cover Segmentation

Implements the DATA CLEANING pipeline from the thesis:
1. Remove Irrelevant Classes (Snow & Ice)
2. Void / Cloud Filtering (Patch-Level)
3. Class Distribution Analysis
4. Boundary Pixel Erosion
5. Minimum Mapping Unit Filtering

"""

import logging
import warnings

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from scipy.ndimage import binary_dilation, binary_erosion, label, sum_labels

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaning:
    """
    Data cleaning pipeline for land cover segmentation masks.
    
    Implements all DATA CLEANING steps from the thesis methodology:
    - Apply City Boundary Mask - Step 1
    - Boundary Pixel Erosion - Step 2
    - Minimum Mapping Unit Filtering - Step 3
    
    Reference: Thesis Section 3.2 (Image Prediction Model Preprocessing)
    """

    def __init__(self,
                 ignore_index: int = 255,
                 boundary_erosion_pixels: int = 2,
                 min_mapping_unit: int = 4,
                 random_seed: int = 42):
        """
        Initialize the DataCleaning pipeline.
        
        Args:
            ignore_index: Value for ignored pixels in loss computation (default: 255)
            boundary_erosion_pixels: Number of pixels to erode from class boundaries (default: 2)
            min_mapping_unit: Minimum connected pixels to keep a class (MMU filter) (default: 4)
            random_seed: Random seed for reproducibility
        """
        self.ignore_index = ignore_index
        self.boundary_erosion_pixels = boundary_erosion_pixels
        self.min_mapping_unit = min_mapping_unit
        self.random_seed = random_seed
        self.boundary = gpd.read_file('../data/bc_boundary/bc_boundary.shp')

        # Set random seed
        np.random.seed(random_seed)

        # Valid classes for Baguio (all except snow_and_ice)
        self.valid_classes = [i for i in range(8)]  # 0-7 only, exclude 8 (snow/ice)

        logger.info(f"DataCleaning initialized: "
                    f"ignore_index={ignore_index}, "
                    f"erosion={boundary_erosion_pixels}px, "
                    f"MMU={min_mapping_unit}px, ")

    # ==================== COMPLETE DATA CLEANING PIPELINE ====================

    def clean(self, mask: np.ndarray, crs, transform) -> np.ndarray:
        """
        Apply the complete data cleaning pipeline to a mask.

        Pipeline order:
        1. Fix outside boundary labels
        2. Apply boundary pixel erosion
        3. Apply minimum mapping unit filtering

        Args:
            mask: Ground truth mask to clean
            crs: Coordinate reference system of the raster
            transform: affine transform of the raster

        Returns:
            Cleaned mask
        """
        # Make a copy to avoid modifying original
        cleaned_mask = mask.copy()

        # Step 1: Fix outside boundary labels
        cleaned_mask = self._apply_boundary_mask(cleaned_mask, crs, transform)

        # Step 2: Apply boundary pixel erosion
        cleaned_mask = self._erode_boundaries(cleaned_mask)

        # Step 3: Apply minimum mapping unit filtering
        cleaned_mask = self._apply_mmp_filter(cleaned_mask)

        # Log final stats
        n_ignore = np.sum(cleaned_mask == self.ignore_index)
        n_total = cleaned_mask.size
        ignore_percent = (n_ignore / n_total) * 100
        logger.info(f"Cleaning complete: {n_ignore}/{n_total} pixels ({ignore_percent:.2f}%) set to ignore_index")

        return cleaned_mask

    # ==================== DATA CLEANING STEP 1: FIX BOUNDARY LABELS ====================

    def _apply_boundary_mask(self, mask, crs, transform) -> np.ndarray:
        """
        Sets all labels outside the boundary to 255.

        Fixes an issue where labels outside the boundary are labeled as water.

        Returns:
            Mask with outside boundary pixels set to ignore_index
        """

        relabeled_mask = mask.copy()

        boundary = self.boundary.to_crs(crs)
        h, w = mask.shape

        city_mask = geometry_mask(
            [feature['geometry'] for feature in boundary.to_dict('records')],
            transform=transform,
            invert=True,
            out_shape=(h, w),
        )

        relabeled_mask[~city_mask] = 255

        return relabeled_mask

    # ==================== DATA CLEANING STEP 2: BOUNDARY PIXEL EROSION ====================

    def _erode_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply boundary pixel erosion.
        
        Reference: DATA CLEANING Step 4
        Paper: Foody (2002) - Status of land cover classification accuracy assessment
        
        Identifies pixels at class boundaries and sets them to ignore_index.
        This prevents the model from learning from uncertain mixed pixels.
        
        Args:
            mask: Ground truth mask
            
        Returns:
            Mask with boundary class pixels set to ignore_index
        """
        if self.boundary_erosion_pixels <= 0:
            return mask

        eroded_mask = mask.copy()
        valid_pixels = (eroded_mask != self.ignore_index)

        if not np.any(valid_pixels):
            logger.warning("Step 4 - No valid pixels found for boundary erosion")
            return eroded_mask

        # Get unique classes (excluding ignore_index)
        unique_classes = np.unique(eroded_mask[valid_pixels])

        # Create boundary mask
        boundary_mask = np.zeros_like(mask, dtype=bool)

        # Find boundaries for each class
        for class_val in unique_classes:
            class_mask = (eroded_mask == class_val)

            if not np.any(class_mask):
                continue

            # Dilate and erode to find boundaries
            dilated = binary_dilation(
                class_mask,
                iterations=self.boundary_erosion_pixels
            )
            eroded_class = binary_erosion(
                class_mask,
                iterations=self.boundary_erosion_pixels
            )

            # Boundary = dilated but not eroded
            class_boundary = dilated & ~eroded_class
            boundary_mask |= class_boundary

        # Also mark pixels adjacent to ignore_index as boundaries
        ignore_adjacent = self._find_ignore_adjacent(eroded_mask)
        boundary_mask |= ignore_adjacent

        # Set boundary pixels to ignore_index
        eroded_mask[boundary_mask] = self.ignore_index

        n_boundary = np.sum(boundary_mask)
        if n_boundary > 0:
            logger.info(f"Step 4 - Eroded {n_boundary} boundary pixels ({self.boundary_erosion_pixels}px erosion)")

        return eroded_mask

    def _find_ignore_adjacent(self, mask: np.ndarray) -> np.ndarray:
        """Find pixels adjacent to ignore_index regions."""
        ignore_mask = (mask == self.ignore_index)

        if not np.any(ignore_mask):
            return np.zeros_like(mask, dtype=bool)

        # Dilate ignore regions
        dilated = binary_dilation(
            ignore_mask,
            iterations=self.boundary_erosion_pixels
        )

        # Adjacent = dilated ignore but not original ignore
        adjacent = dilated & ~ignore_mask

        return adjacent

    # ==================== DATA CLEANING STEP 3: MINIMUM MAPPING UNIT FILTERING ====================

    def _apply_mmp_filter(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Minimum Mapping Unit (MMU) filtering.
        
        Reference: DATA CLEANING Step 5
        Paper: Teillet, Guindon & Goodenough (1982) - On the slope-aspect correction
        
        Removes isolated pixel groups smaller than min_mapping_unit by
        reclassifying them to the surrounding majority class.
        
        Args:
            mask: Ground truth mask
            
        Returns:
            Mask with small isolated groups removed
        """
        if self.min_mapping_unit <= 1:
            return mask

        filtered_mask = mask.copy()
        unique_classes = np.unique(filtered_mask[filtered_mask != self.ignore_index])

        total_changed = 0

        for class_val in unique_classes:
            class_mask = (filtered_mask == class_val)

            if not np.any(class_mask):
                continue

            # Label connected components
            labeled, num_features = label(class_mask)

            if num_features == 0:
                continue

            # Get sizes of each component
            component_sizes = sum_labels(
                class_mask.astype(int),
                labeled,
                range(1, num_features + 1)
            )

            # Find small components
            for comp_id in range(1, num_features + 1):
                if component_sizes[comp_id - 1] < self.min_mapping_unit:
                    component_pixels = (labeled == comp_id)

                    # Find neighboring pixels (not ignore_index and not part of this component)
                    dilated = binary_dilation(component_pixels)
                    neighbors = dilated & ~component_pixels & (filtered_mask != self.ignore_index)

                    if np.any(neighbors):
                        # Reclassify to majority neighbor class
                        neighbor_classes = filtered_mask[neighbors]
                        majority_class = np.bincount(neighbor_classes.astype(int)).argmax()
                        filtered_mask[component_pixels] = majority_class
                        total_changed += np.sum(component_pixels)
                    else:
                        # No valid neighbors, set to ignore_index
                        filtered_mask[component_pixels] = self.ignore_index
                        total_changed += np.sum(component_pixels)

        if total_changed > 0:
            logger.info(f"Step 5 - MMU filter changed {total_changed} pixels (< {self.min_mapping_unit}px)")

        return filtered_mask