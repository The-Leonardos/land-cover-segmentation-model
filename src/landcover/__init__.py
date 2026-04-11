from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'

# Classes
NUM_CLASSES = 7
MINORITY_CLASSES = [0, 6, 3, 4]
MIN_VALID_RATIO = 0.6
IGNORE_INDEX = 255

# Probability of Patch Types and Augmentation
PATCH_SAMPLING_PROBS = {"minority": 0.7, "random": 0.3}
AUGMENTATION_PROBS = 0.5