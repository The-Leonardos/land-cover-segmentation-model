# scripts/run_data_cleaning.py

import argparse
from pathlib import Path
import rasterio as rio
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.landcover.utils.data_cleaning import DataCleaning

def main():
    parser = argparse.ArgumentParser(description="Run data cleaning on mask files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing mask .tiff files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save cleaned masks")
    parser.add_argument("--erosion", type=int, default=2,
                        help="Boundary erosion pixels (default: 2)")
    parser.add_argument("--mmu", type=int, default=4,
                        help="Minimum mapping unit size (default: 4)")
    
    args = parser.parse_args()
    
    # Create output directory
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize cleaner
    cleaner = DataCleaning(
        ignore_index=255,
        boundary_erosion_pixels=args.erosion,
        min_mapping_unit=args.mmu
    )
    
    # Get all mask files
    mask_files = list(input_path.glob("*.tiff")) + list(input_path.glob("*.tif"))
    print(f"Found {len(mask_files)} mask files to process")
    
    # Process each mask
    for mask_file in tqdm(mask_files):
        # Read mask
        with rio.open(mask_file) as src:
            mask = src.read(1)
            profile = src.profile
            crs = src.crs
            transform = src.transform
        
        # Clean the mask
        cleaned_mask = cleaner.clean(
            mask=mask,
            crs=crs,
            transform=transform
        )
        
        # Save cleaned mask
        out_file = output_path / mask_file.name
        profile.update(dtype=rio.uint8, nodata=255)
        
        with rio.open(out_file, "w", **profile) as dst:
            dst.write(cleaned_mask.astype(rio.uint8), 1)
    
    print(f"Done! Cleaned masks saved to {args.output_dir}")

if __name__ == "__main__":
    main()