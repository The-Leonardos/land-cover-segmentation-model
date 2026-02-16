"""
Data Cleaning Module for Land Cover Segmentation

Implements the DATA CLEANING pipeline from the thesis:
1. Remove Irrelevant Classes (Snow & Ice)
2. Void / Cloud Filtering (Patch-Level)
3. Class Distribution Analysis
4. Boundary Pixel Erosion
5. Minimum Mapping Unit Filtering

"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, label, sum_labels
import logging
from typing import Tuple, List, Optional, Dict, Union
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaning:
    """
    Data cleaning pipeline for land cover segmentation masks.
    
    Implements all DATA CLEANING steps from the thesis methodology:
    - Remove Irrelevant Classes (Snow & Ice) - Step 1
    - Void / Cloud Filtering (Patch-Level) - Step 2
    - Class Distribution Analysis - Step 3
    - Boundary Pixel Erosion - Step 4
    - Minimum Mapping Unit Filtering - Step 5
    
    Reference: Thesis Section 3.2 (Image Prediction Model Preprocessing)
    """
    
    # Dynamic World class mapping (9 classes)
    CLASS_MAPPING = {
        0: 'water',
        1: 'trees', 
        2: 'grass',
        3: 'flooded_vegetation',
        4: 'crops',
        5: 'shrub_and_scrub',
        6: 'built',
        7: 'bare',
        8: 'snow_and_ice'  # To be removed for Baguio
    }
    
    def __init__(self,
                 ignore_index: int = 255,
                 boundary_erosion_pixels: int = 2,
                 min_mapping_unit: int = 4,
                 void_threshold: float = 0.5,
                 confidence_threshold: float = 0.65,
                 random_seed: int = 42):
        """
        Initialize the DataCleaning pipeline.
        
        Args:
            ignore_index: Value for ignored pixels in loss computation (default: 255)
            boundary_erosion_pixels: Number of pixels to erode from class boundaries (default: 2)
            min_mapping_unit: Minimum connected pixels to keep a class (MMU filter) (default: 4)
            void_threshold: Maximum ratio of void pixels allowed in a patch (default: 0.5)
            confidence_threshold: Minimum probability to keep a pixel (for confidence masking) (default: 0.65)
            random_seed: Random seed for reproducibility
        """
        self.ignore_index = ignore_index
        self.boundary_erosion_pixels = boundary_erosion_pixels
        self.min_mapping_unit = min_mapping_unit
        self.void_threshold = void_threshold
        self.confidence_threshold = confidence_threshold
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Valid classes for Baguio (all except snow_and_ice)
        self.valid_classes = [i for i in range(8)]  # 0-7 only, exclude 8 (snow/ice)
        
        logger.info(f"DataCleaning initialized: "
                   f"ignore_index={ignore_index}, "
                   f"erosion={boundary_erosion_pixels}px, "
                   f"MMU={min_mapping_unit}px, "
                   f"void_threshold={void_threshold}, "
                   f"conf_threshold={confidence_threshold}")
    
    # ==================== DATA CLEANING STEP 1: REMOVE IRRELEVANT CLASSES ====================
    
    def remove_irrelevant_classes(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove irrelevant classes (snow/ice) for Baguio.
        
        Reference: DATA CLEANING Step 1
        Paper: Congalton & Green (2019) - Assessing Accuracy of Remotely Sensed Data
        
        Args:
            mask: Ground truth mask (class labels)
            
        Returns:
            Mask with snow/ice pixels set to ignore_index
        """
        # Create mask of valid classes (0-7 only)
        valid_mask = np.isin(mask, self.valid_classes)
        
        # Set invalid classes (snow/ice = 8) to ignore_index
        cleaned_mask = np.where(valid_mask, mask, self.ignore_index)
        
        # Count removed pixels
        n_removed = np.sum(~valid_mask)
        if n_removed > 0:
            logger.info(f"Step 1 - Removed {n_removed} irrelevant class pixels (snow/ice)")
        
        return cleaned_mask
    
    # ==================== DATA CLEANING STEP 2: VOID / CLOUD FILTERING ====================
    
    def filter_void_patches(self,
                           image_patch: np.ndarray,
                           mask_patch: np.ndarray) -> bool:
        """
        Filter patches with too many void/black pixels.
        
        Reference: DATA CLEANING Step 2
        Paper: Zhu & Woodcock (2012) - Object-based cloud detection in Landsat imagery
        
        Args:
            image_patch: Image patch to check (RGB/NIR)
            mask_patch: Corresponding mask patch
            
        Returns:
            True if patch should be kept, False if discarded
        """
        # Check for void pixels in image (black or near-black)
        if len(image_patch.shape) == 3:  # RGB image
            # Black pixels (all channels near 0)
            void_img = np.all(image_patch < 10, axis=-1)
        else:  # Grayscale
            void_img = image_patch < 10
        
        # Check for ignore pixels in mask
        void_mask = (mask_patch == self.ignore_index)
        
        # Combine void pixels
        void_pixels = void_img | void_mask
        void_ratio = np.sum(void_pixels) / void_pixels.size
        
        keep = void_ratio <= self.void_threshold
        
        if not keep:
            logger.debug(f"Step 2 - Discarding patch: {void_ratio:.2%} void pixels > {self.void_threshold:.2%}")
        
        return keep
    
    # ==================== DATA CLEANING STEP 3: CLASS DISTRIBUTION ANALYSIS ====================
    
    def class_distribution_analysis(self, mask: np.ndarray) -> Dict[int, float]:
        """
        Analyze class distribution in mask.
        
        Reference: DATA CLEANING Step 3
        Paper: Buda et al. (2018) - A systematic study of the class imbalance problem in CNNs
        
        Args:
            mask: Ground truth mask
            
        Returns:
            Dictionary of class -> percentage
        """
        # Get only valid pixels (not ignore_index)
        valid_pixels = mask[mask != self.ignore_index]
        
        if len(valid_pixels) == 0:
            logger.warning("Step 3 - No valid pixels found in mask")
            return {}
        
        # Calculate class distribution
        unique, counts = np.unique(valid_pixels, return_counts=True)
        percentages = (counts / len(valid_pixels)) * 100
        
        distribution = {int(cls): float(pct) for cls, pct in zip(unique, percentages)}
        
        # Log distribution
        logger.info("Step 3 - Class distribution analysis:")
        for cls, pct in sorted(distribution.items()):
            class_name = self.CLASS_MAPPING.get(cls, f"class_{cls}")
            logger.info(f"  {class_name} (class {cls}): {pct:.2f}%")
        
        # Check for severe imbalance
        if len(distribution) > 0:
            max_class = max(distribution, key=distribution.get)
            min_class = min(distribution, key=distribution.get)
            imbalance_ratio = distribution[max_class] / distribution[min_class] if distribution[min_class] > 0 else float('inf')
            
            if imbalance_ratio > 10:
                logger.warning(f"Step 3 - Severe class imbalance detected! "
                              f"Ratio {max_class}:{min_class} = {imbalance_ratio:.2f}:1")
        
        return distribution
    
    # ==================== DATA CLEANING STEP 4: BOUNDARY PIXEL EROSION ====================
    
    def erode_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply boundary pixel erosion.
        
        Reference: DATA CLEANING Step 4
        Paper: Foody (2002) - Status of land cover classification accuracy assessment
        
        Identifies pixels at class boundaries and sets them to ignore_index.
        This prevents the model from learning from uncertain mixed pixels.
        
        Args:
            mask: Ground truth mask
            
        Returns:
            Mask with boundary pixels set to ignore_index
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
    
    # ==================== DATA CLEANING STEP 5: MINIMUM MAPPING UNIT FILTERING ====================
    
    def apply_mmp_filter(self, mask: np.ndarray) -> np.ndarray:
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
    
    # ==================== COMPLETE DATA CLEANING PIPELINE ====================
    
    def clean(self, 
              mask: np.ndarray,
              image_patch: Optional[np.ndarray] = None,
              confidence_map: Optional[np.ndarray] = None,
              perform_analysis: bool = True) -> np.ndarray:
        """
        Apply the complete data cleaning pipeline to a mask.
        
        Pipeline order:
        1. Remove irrelevant classes (snow/ice)
        2. Apply confidence threshold (if confidence_map provided)
        3. Apply boundary pixel erosion
        4. Apply minimum mapping unit filtering
        
        Note: Void filtering (Step 2) is applied at patch level in dataset class
        
        Args:
            mask: Ground truth mask to clean
            image_patch: Optional image patch for void filtering (used by dataset)
            confidence_map: Optional confidence scores for mask pixels (Dynamic World probabilities)
            perform_analysis: Whether to run class distribution analysis
            
        Returns:
            Cleaned mask
        """
        if mask is None:
            return None
        
        # Make a copy to avoid modifying original
        cleaned_mask = mask.copy()
        
        # Step 1: Remove irrelevant classes (snow/ice)
        cleaned_mask = self.remove_irrelevant_classes(cleaned_mask)
        
        # Optional: Apply confidence threshold if confidence map provided
        if confidence_map is not None:
            cleaned_mask = self._apply_confidence_threshold(cleaned_mask, confidence_map)
        
        # Step 4: Apply boundary pixel erosion
        cleaned_mask = self.erode_boundaries(cleaned_mask)
        
        # Step 5: Apply minimum mapping unit filtering
        cleaned_mask = self.apply_mmp_filter(cleaned_mask)
        
        # Optional: Perform class distribution analysis
        if perform_analysis:
            self.class_distribution_analysis(cleaned_mask)
        
        # Log final stats
        n_ignore = np.sum(cleaned_mask == self.ignore_index)
        n_total = cleaned_mask.size
        ignore_percent = (n_ignore / n_total) * 100
        logger.info(f"Cleaning complete: {n_ignore}/{n_total} pixels ({ignore_percent:.2f}%) set to ignore_index")
        
        return cleaned_mask
    
    def _apply_confidence_threshold(self, mask: np.ndarray, confidence_map: np.ndarray) -> np.ndarray:
        """
        Apply confidence threshold masking (from Data Collection step).
        
        This is included here as it's closely related to mask cleaning.
        """
        if confidence_map is None:
            return mask
            
        assert mask.shape == confidence_map.shape, \
            f"Shape mismatch: mask {mask.shape} != confidence {confidence_map.shape}"
        
        # Create low-confidence mask
        low_conf_mask = confidence_map < self.confidence_threshold
        
        # Set low-confidence pixels to ignore_index
        cleaned_mask = mask.copy()
        cleaned_mask[low_conf_mask] = self.ignore_index
        
        n_low_conf = np.sum(low_conf_mask)
        if n_low_conf > 0:
            logger.info(f"Applied confidence threshold: masked {n_low_conf} low-confidence pixels")
        
        return cleaned_mask
    
    # ==================== UTILITY METHODS ====================
    
    def visualize_cleaning_steps(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Visualize each step of the cleaning process (for debugging/analysis).
        
        Args:
            mask: Original mask
            
        Returns:
            Dictionary with results from each step
        """
        results = {
            'original': mask.copy(),
            'step1_removed_classes': self.remove_irrelevant_classes(mask),
            'step4_eroded': None,
            'step5_mmp_filtered': None,
            'final': None
        }
        
        # Step 1
        step1 = results['step1_removed_classes']
        
        # Step 4
        step4 = self.erode_boundaries(step1)
        results['step4_eroded'] = step4
        
        # Step 5
        step5 = self.apply_mmp_filter(step4)
        results['step5_mmp_filtered'] = step5
        results['final'] = step5
        
        return results


# ==================== SCRIPT USAGE EXAMPLE ====================

if __name__ == "__main__":
    """
    Example usage of the DataCleaning class.
    """
    
    # Create a sample mask for testing
    sample_mask = np.array([
        [8, 8, 1, 1, 1, 2, 2, 2],  # snow at top
        [8, 1, 1, 1, 1, 2, 2, 2],
        [1, 1, 1, 1, 1, 2, 2, 2],
        [1, 1, 1, 1, 1, 6, 6, 6],  # built at bottom
        [1, 1, 1, 1, 1, 6, 6, 6],
        [1, 1, 1, 6, 6, 6, 6, 6],
        [1, 1, 6, 6, 6, 6, 6, 6],
        [1, 1, 6, 6, 6, 6, 6, 6]
    ], dtype=np.uint8)
    
    # Create dummy confidence map
    confidence = np.ones_like(sample_mask, dtype=float)
    confidence[2:4, 2:4] = 0.5  # low confidence area
    
    # Initialize cleaner
    cleaner = DataCleaning(
        ignore_index=255,
        boundary_erosion_pixels=1,
        min_mapping_unit=4,
        void_threshold=0.5,
        confidence_threshold=0.65
    )
    
    # Run cleaning
    cleaned_mask = cleaner.clean(
        mask=sample_mask,
        confidence_map=confidence,
        perform_analysis=True
    )
    
    print("\nOriginal mask unique values:", np.unique(sample_mask))
    print("Cleaned mask unique values:", np.unique(cleaned_mask))
    print(f"Number of ignore pixels: {np.sum(cleaned_mask == 255)}")
    
    # Visualize steps
    steps = cleaner.visualize_cleaning_steps(sample_mask)
    print("\nCleaning steps completed:")
    for step_name, step_result in steps.items():
        if step_result is not None:
            n_ignore = np.sum(step_result == 255)
            print(f"  {step_name}: {n_ignore} ignore pixels")