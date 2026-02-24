import sys
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.landcover.datasets.dataset import LandCoverDataset
from src.landcover.utils.data_preprocessing import Preprocessing
import rasterio as rio


def test_city_mask_caching():
    """Test if city mask is being cached properly"""
    print("\n" + "=" * 60)
    print("TEST 1: City Mask Caching")
    print("=" * 60)

    # Use training split
    try:
        dataset = LandCoverDataset('data/', split='train', cache_images=False, use_numpy_cache=False)
    except AssertionError as e:
        print(f" No training data found! Error: {e}")
        return False

    if len(dataset) == 0:
        print(" No data found in dataset/clean/train/")
        return False

    # Get transform and CRS from first image
    with rio.open(dataset.image_files[0]) as src:
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

    # Initialize preprocessing
    preprocess = Preprocessing(transform, crs, patch_size=256)

    # First call - should compute
    print("\n1. First call (computing city mask):")
    start = time.time()
    mask1 = preprocess._get_city_mask(h, w)
    t1 = time.time() - start
    print(f"   Time: {t1:.4f} seconds")

    # Second call - should use cache
    print("\n2. Second call (should use cache):")
    start = time.time()
    mask2 = preprocess._get_city_mask(h, w)
    t2 = time.time() - start
    print(f"   Time: {t2:.4f} seconds")

    # Verify
    if np.array_equal(mask1, mask2):
        print(f"\n City mask caching WORKING! {t1 / t2:.1f}x faster")
        return True
    else:
        print("\n City mask caching NOT working")
        return False


def test_numpy_cache_conversion():
    """Test if NumPy cache conversion works"""
    print("\n" + "=" * 60)
    print("TEST 2: NumPy Cache Conversion")
    print("=" * 60)

    # Delete existing numpy cache to test conversion
    numpy_cache_dir = Path('data/numpy_cache')
    if numpy_cache_dir.exists():
        import shutil
        shutil.rmtree(numpy_cache_dir)
        print(" Cleared existing numpy cache")

    # Create dataset with numpy cache
    print("\n1. Creating dataset with use_numpy_cache=True...")
    try:
        start = time.time()
        dataset = LandCoverDataset('data/', split='train', cache_images=False, use_numpy_cache=True)
        creation_time = time.time() - start
        print(f"   Dataset creation time: {creation_time:.4f} seconds")
    except Exception as e:
        print(f" Failed to create dataset: {e}")
        return False

    # Check if numpy files were created
    npy_files = list((numpy_cache_dir / 'train').glob('*.npy'))
    print(f"\n2. NumPy files created: {len(npy_files)}")
    for f in list(npy_files)[:3]:  # Show first 3
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name}: {size_kb:.2f} KB")

    return len(npy_files) > 0


def test_loading_speed_comparison():
    """Compare loading speeds of different caching methods"""
    print("\n" + "=" * 60)
    print("TEST 3: Loading Speed Comparison")
    print("=" * 60)

    results = {}

    try:
        # Test 1: TIFF only (no cache)
        print("\n1. TIFF only (no cache):")
        start = time.time()
        d1 = LandCoverDataset('data/', split='train', cache_images=False, use_numpy_cache=False)
        num_samples = min(5, len(d1))
        for i in range(num_samples):
            img, mask = d1[i]
        t1 = time.time() - start
        print(f"   Time: {t1:.4f} seconds")
        results['TIFF only'] = t1

        # Test 2: NumPy cache only
        print("\n2. NumPy cache only:")
        start = time.time()
        d2 = LandCoverDataset('data/', split='train', cache_images=False, use_numpy_cache=True)
        for i in range(num_samples):
            img, mask = d2[i]
        t2 = time.time() - start
        print(f"   Time: {t2:.4f} seconds")
        results['NumPy cache'] = t2

        # Test 3: RAM cache only
        print("\n3. RAM cache only:")
        start = time.time()
        d3 = LandCoverDataset('data/', split='train', cache_images=True, use_numpy_cache=False)
        for i in range(num_samples):
            img, mask = d3[i]
        t3 = time.time() - start
        print(f"   Time: {t3:.4f} seconds")
        results['RAM cache'] = t3

        # Test 4: Both caches (RAM + NumPy)
        print("\n4. Both caches (RAM + NumPy):")
        start = time.time()
        d4 = LandCoverDataset('data/', split='train', cache_images=True, use_numpy_cache=True)
        for i in range(num_samples):
            img, mask = d4[i]
        t4 = time.time() - start
        print(f"   Time: {t4:.4f} seconds")
        results['Both'] = t4

        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        baseline = results['TIFF only']
        for method, time_taken in results.items():
            speedup = baseline / time_taken
            print(f"{method:15}: {time_taken:.4f}s ({speedup:.1f}x faster)")

        return results
    except Exception as e:
        print(f" Test failed: {e}")
        return None


def test_dataloader_with_cache():
    """Test if DataLoader works with cached dataset"""
    print("\n" + "=" * 60)
    print("TEST 4: DataLoader with Cached Dataset")
    print("=" * 60)

    try:
        # Create dataset with best cache settings
        dataset = LandCoverDataset('data/', split='train', cache_images=True, use_numpy_cache=True)

        # Create dataloader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        print(f" Dataset size: {len(dataset)} samples")
        print(f" Batch size: 4")
        print(f" Testing 2 batches...")

        start = time.time()
        for i, (images, masks) in enumerate(loader):
            if i >= 2:  # Test 2 batches only
                break
            print(f"   Batch {i + 1}:")
            print(f"      Images shape: {images.shape}")
            print(f"      Masks shape: {masks.shape}")
            print(f"      Images dtype: {images.dtype}")
            print(f"      Masks dtype: {masks.dtype}")
        t = time.time() - start
        print(f"\n DataLoader working! Time for 2 batches: {t:.4f}s")
        return True
    except Exception as e:
        print(f" DataLoader failed: {e}")
        return False


def verify_data_exists():
    """Verify that data exists in the expected locations"""
    print("\n" + "=" * 60)
    print("VERIFYING DATA LOCATIONS")
    print("=" * 60)

    # Check various possible locations
    possible_paths = [
        Path('data/dataset/clean/train/images'),
        Path('data/dataset/clean/train/masks'),
        Path('data/dataset/raw/train/images'),
        Path('data/dataset/raw/train/masks'),
    ]

    all_good = True
    for path in possible_paths[:2]:  # Check images and masks for train
        if path.exists():
            files = list(path.glob('*.tif'))
            print(f" {path}: {len(files)} .tif files")
            if len(files) == 0:
                all_good = False
        else:
            print(f" {path}: Does not exist")
            all_good = False

    return all_good


if __name__ == "__main__":
    print("\n" + "*" * 20)
    print("FINAL OPTIMIZATION TESTS")
    print("*" * 20)

    # First verify data exists
    if not verify_data_exists():
        print("\n Data verification failed!")
        print("\nYour data should be in:")
        print("  data/dataset/clean/train/images/")
        print("  data/dataset/clean/train/masks/")
        print("\nOr in:")
        print("  data/dataset/raw/train/images/")
        print("  data/dataset/raw/train/masks/")
        print("\nCurrent structure:")
        import subprocess

        subprocess.run("dir data\\dataset /s /b | findstr .tif", shell=True)
        sys.exit(1)

    # Run all tests
    tests = [
        ("City Mask Caching", test_city_mask_caching),
        ("NumPy Cache Conversion", test_numpy_cache_conversion),
        ("Loading Speed Comparison", test_loading_speed_comparison),
        ("DataLoader Test", test_dataloader_with_cache),
    ]

    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = " PASSED" if result else " ISSUES"
        except Exception as e:
            print(f" Test failed with error: {e}")
            results[name] = " FAILED"

    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"{name:30}: {status}")