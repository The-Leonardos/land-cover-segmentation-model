# Land Cover Segmentation Model

This project is a deep learning pipeline for land cover segmentation using satellite imagery. It trains a **DeepLabV3+** semantic segmentation model (with a ResNet-50 encoder) on 4-channel multispectral GeoTIFF patches to classify each pixel into one of 7 land cover classes derived from the [Dynamic World v1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) dataset. Training metrics are tracked via [Weights & Biases (wandb)](https://wandb.ai/).

## Land Cover Classes

The model predicts 7 land cover classes (mapped from Dynamic World v1):

| Class ID | Label              |
|----------|--------------------|
| 0        | Water              |
| 1        | Trees              |
| 2        | Grass              |
| 3        | Crops              |
| 4        | Shrub & Scrub      |
| 5        | Built              |
| 6        | Bare               |

> **Note:** Dynamic World's `flooded_vegetation` (class 3) and `snow_and_ice` (class 8) are excluded.

## Project Structure

```
land-cover-segmentation-model/
├── config.env                   # Environment variables (wandb API key)
├── requirements.txt             # Python dependencies
├── data/
│   ├── bc_boundary/             # Shapefile of the study area boundary
│   │   └── bc_boundary.shp
│   └── dataset/
│       ├── raw/                 # Raw GeoTIFF patches (input)
│       │   ├── train/
│       │   │   ├── images/      # 4-channel .tif image patches
│       │   │   └── masks/       # Single-band .tif label masks
│       │   ├── test/
│       │   │   ├── images/
│       │   │   └── masks/
│       │   └── validation/
│       │       ├── images/
│       │       └── masks/
│       └── clean/               # Cleaned .npy patches (generated)
│           ├── train/
│           ├── test/
│           └── validation/
├── notebooks/
│   └── dataset_exploration.ipynb
├── scripts/
│   ├── run_data_cleaning.py     # Step 1: Clean raw data
│   ├── run_hyperparameter_tuning.py  # Step 2 (optional): Tune hyperparameters
│   └── run_model_training.py    # Step 3: Train the model
└── src/
    └── landcover/
        ├── datasets/            # PyTorch Dataset class
        ├── models/              # DeepLabV3+ model wrapper
        ├── training/            # Training loop & hyperparameter tuning
        ├── evaluation/          # Evaluation/test loop
        └── utils/               # Data cleaning, preprocessing, loss, metrics
```

## Setup Instructions

Follow these steps to set up the project locally.

### 1. Prerequisites

Ensure you have the following installed:

- [Python](https://www.python.org/downloads/) 3.9 or higher
- [pip](https://pip.pypa.io/en/stable/) (comes with Python)
- [CUDA-enabled GPU](https://developer.nvidia.com/cuda-downloads) (strongly recommended for training; CPU fallback is supported but very slow)
- A [Weights & Biases](https://wandb.ai/) account and API key (for experiment tracking)

### 2. Clone the Repository

```bash
git clone <repository-url>
cd land-cover-segmentation-model
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following key packages:

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `segmentation-models-pytorch` | DeepLabV3+ model and loss functions |
| `rasterio` | Reading GeoTIFF satellite images |
| `geopandas` | Geospatial boundary masking |
| `albumentations` | Data augmentation |
| `optuna` | Hyperparameter optimization |
| `wandb` | Experiment tracking |
| `numpy`, `scipy`, `tqdm` | Numerical computing and progress bars |
| `python-dotenv` | Loading environment variables |

### 5. Install the `landcover` Package

Install the `src` package in editable mode so the scripts can import `landcover`:

```bash
pip install -e src/
```

> If there is no `setup.py` or `pyproject.toml` yet, add the `src/` folder to your `PYTHONPATH` instead:
>
> ```bash
> # On Windows (PowerShell):
> $env:PYTHONPATH = "src"
>
> # On macOS/Linux:
> export PYTHONPATH=src
> ```

### 6. Configure Weights & Biases

Open `config.env` and paste your wandb API key:

```env
WANDB_API_KEY=your_actual_api_key_here
```

You can find your key at [https://wandb.ai/authorize](https://wandb.ai/authorize).

### 7. Prepare Your Raw Dataset

Place your raw GeoTIFF patches in the following structure under `data/dataset/raw/`:

```
data/dataset/raw/
├── train/
│   ├── images/    <- 4-channel multispectral .tif files
│   └── masks/     <- single-band land cover label .tif files
├── test/
│   ├── images/
│   └── masks/
└── validation/
    ├── images/
    └── masks/
```

Also ensure the study area boundary shapefile is present at:

```
data/bc_boundary/bc_boundary.shp
```

## Running the Pipeline

All scripts must be run from inside the `scripts/` directory:

```bash
cd scripts
```

---

### Step 1: Clean the Raw Data

This script reads the raw GeoTIFF patches, applies cleaning operations (boundary masking, minimum mapping unit filtering, and nodata handling), and saves the results as NumPy `.npy` files in `data/dataset/clean/`.

```bash
python run_data_cleaning.py
```

**What it does:**
- Applies boundary erosion (1 pixel) to remove edge artifacts.
- Applies a minimum mapping unit (MMU) filter (4 pixels) to remove noise.
- Marks invalid or out-of-boundary pixels with `ignore_index=255`.
- Saves a `city_mask.npy` in `data/misc/` for the study area boundary.

**Expected output:**
```
[train] Found N mask files to process
[train] Found N images to process
[Images] Cleaned images saved to data/dataset/clean/train/images
[Masks]  Cleaned masks saved to data/dataset/clean/train/masks
[City Mask] Done! City mask saved to data/misc/city_mask.npy
[Data Cleaning] All images and masks successfully cleaned...
```

---

### Step 2 (Optional): Hyperparameter Tuning

Run this step if you want to search for optimal hyperparameters before full training. It uses [Optuna](https://optuna.org/) with median pruning and logs all trials to wandb.

```bash
python run_hyperparameter_tuning.py
```

**Tuning configuration (editable in the script):**

| Parameter | Default |
|---|---|
| Encoder | `efficientnet-b0` |
| Number of trials | 30 |
| Epochs per trial | 50 |
| Search space | lr, weight_decay, batch_size, decoder_channels, atrous rates, patch_size, ASPP dropout, dice weight |

**Output:**
- Prints the best hyperparameters to the console.
- Saves a CSV of all trial results to `data/hyperparameter_tuning_logs/<timestamp>_deeplabv3plus_tuning_results.csv`.
- Logs all runs to the `land-cover-mapping` wandb project.

---

### Step 3: Train the Model

Run the full training loop with the configured hyperparameters. By default, this uses the best-known hyperparameters from a prior tuning run.

```bash
python run_model_training.py
```

**Key training configuration (editable in the script):**

| Parameter | Default Value |
|---|---|
| Encoder | `resnet50` |
| Encoder depth | 4 |
| Encoder output stride | 8 |
| Decoder channels | 256 |
| Decoder atrous rates | `(12, 24, 36)` |
| ASPP dropout | 0.3616 |
| Batch size | 32 |
| Patch size | 512 |
| Learning rate | 0.003920 |
| Weight decay | 9.62e-05 |
| Dice loss weight | 0.5344 |
| Total epochs | 220 |
| LR scheduler | Cosine Annealing |
| Optimizer | AdamW |

**What it does:**
1. Loads cleaned `.npy` patches from `data/dataset/clean/train/` and `data/dataset/clean/test/`.
2. Initializes a `DeepLabV3+` model with a ResNet-50 encoder pretrained on ImageNet.
3. Trains for 220 epochs using a combined Dice + Cross-Entropy loss (weighted by class frequency).
4. Saves the best model checkpoint (by validation mIoU) to `data/models/resnet50/`.
5. Stops early if the loss becomes `NaN`.
6. Logs all metrics per epoch to wandb: train/val loss, mIoU, learning rate, best epoch.

**Expected console output (per epoch):**
```
[device]: CUDA
resnet50 Training:  45%|##########     | 99/220 [...]
```

**Saved model checkpoints:**
```
data/models/resnet50/
├── deeplabv3plus_resnet50_best_<epoch>.pth   <- best validation mIoU checkpoint
└── deeplabv3plus_resnet50_best_220.pth       <- final epoch checkpoint
```

---

## Monitoring Training

All training runs and hyperparameter trials are logged to your [Weights & Biases](https://wandb.ai/) project `land-cover-mapping`.

Metrics tracked per epoch:

| Metric | Description |
|---|---|
| `Train Loss` | Average training loss |
| `Validation Loss` | Average validation loss |
| `Train IoU` | Mean IoU on the training set |
| `Validation IoU` | Mean IoU on the validation set |
| `Best IoU` | Best validation mIoU seen so far |
| `Best Epoch` | Epoch at which best IoU was achieved |
| `learning rate` | Current LR (follows cosine annealing) |

---

## Available Scripts

| Script | Description |
|---|---|
| `scripts/run_data_cleaning.py` | Cleans raw GeoTIFF patches and saves `.npy` files |
| `scripts/run_hyperparameter_tuning.py` | Runs Optuna hyperparameter search (30 trials x 50 epochs) |
| `scripts/run_model_training.py` | Trains the full DeepLabV3+ model (220 epochs) |
| `notebooks/dataset_exploration.ipynb` | Exploratory data analysis notebook |