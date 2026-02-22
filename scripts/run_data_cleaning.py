from pathlib import Path
import rasterio as rio
from tqdm import tqdm
from landcover.utils import DataCleaning

def main():
    base_path = Path(__file__).resolve().parent.parent / 'data' / 'dataset'

    input_path = base_path / 'raw'
    output_path = base_path / 'clean'

    # create output path if missing
    output_path.mkdir(parents=True, exist_ok=True)

    # get train, test, and validation splits
    splits = [p.name for p in input_path.iterdir() if p.is_dir()]

    # initialize cleaner
    cleaner = DataCleaning(
        ignore_index=255,
        boundary_erosion_pixels=2,
        min_mapping_unit=4
    )

    # clean data per split
    for split in splits:
        raw_masks_path = input_path / split / 'masks'
        clean_masks_path = output_path / split / 'masks'

        # create clean masks path if missing
        clean_masks_path.mkdir(parents=True, exist_ok=True)

        # Get all mask files
        mask_files = (list(raw_masks_path.glob('*.tif')) + list(raw_masks_path.glob('*.tiff')))
        print(f'[{split}] Found {len(mask_files)} mask files to process')

        # clean all masks in the split
        for mask_file in tqdm(mask_files, desc=f'Cleaning {split} split', leave=False):
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
            out_file = clean_masks_path / mask_file.name
            profile.update(dtype=rio.uint8, nodata=255)

            with rio.open(out_file, "w", **profile) as dst:
                dst.write(cleaned_mask.astype(rio.uint8), 1)

        print(f'\nDone! Cleaned masks saved to {clean_masks_path}')

    print(f"\nCleaning Complete!")

if __name__ == "__main__":
    main()