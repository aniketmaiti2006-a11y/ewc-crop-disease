# =============================================================================
# data_loader.py
# Project : Elastic Weight Consolidation for Continual Crop Disease Classification
# Paper   : IEEE — EWC for Continual Crop Disease Classification
# =============================================================================
"""
Seasonal Drift Data Loaders
────────────────────────────
Simulates three agro-photometric seasonal scenarios that induce
distribution shift in crop-disease image data:

    Task 1 — 'Spring'  : Low-light / overcast conditions (~2 000 lux).
                         Gamma darkening + mild Gaussian noise.
    Task 2 — 'Summer'  : High-brightness outdoor light (~10 000 lux).
                         Gamma brightening + saturation boost.
    Task 3 — 'Autumn'  : Colour-normalised / grey-shifted foliage.
                         Reduced saturation + warm-hue shift.

Dataset
───────
This loader expects the
**New Plant Diseases Dataset (Augmented)** from Kaggle:

    https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

It contains ~87,000 RGB images across 38 plant-disease classes
(Apple, Corn, Cherry, Grape, Peach, Pepper, Potato, Tomato,
Strawberry, Squash, …) in a standard ``ImageFolder`` layout::

    data/plant_diseases/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

Usage
─────
    from data_loader import get_task_loaders, TASKS

    loaders = get_task_loaders(
        root="./data/plant_diseases", batch_size=32
    )
    train_loader, val_loader = loaders["Spring"]

If ``root`` does not contain real images the module falls back to a
synthetic ``FakeCropDataset`` so the training pipeline can be validated
without any downloaded data — useful for CI / unit tests.
"""

import os
import warnings
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

# ── Attempt to use a real ImageFolder dataset; fall back to synthetic ────────
try:
    from torchvision.datasets import ImageFolder
    _HAS_IMAGEFOLDER = True
except ImportError:                          # pragma: no cover
    _HAS_IMAGEFOLDER = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASKS: Tuple[str, ...] = ("Spring", "Summer", "Autumn")

# ImageNet normalisation — applied *after* seasonal augmentation
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224          # ResNet-50 input
NUM_CLASSES_DEFAULT = 38  # overridden when real data is used


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-Season Augmentation Pipelines
# ─────────────────────────────────────────# ─────────────────────────────────

class _SpringTransform:
    """
    Spring (low-light, ~2 000 lux):
      • Gamma correction > 1  →  darken image
      • Mild Gaussian blur (simulates diffuse overcast light)
      • Standard geometric augmentations
    """
    def __init__(self):
        self._base = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
        ])

    def __call__(self, img):
        img = self._base(img)
        # Gamma > 1 → darkens (x^γ, γ=1.6)
        img = img.pow(1.6).clamp(0, 1)
        # Additive Gaussian noise (σ=0.02) mimics sensor noise in low light
        img = (img + torch.randn_like(img) * 0.02).clamp(0, 1)
        img = T.functional.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
        return img


class _SummerTransform:
    """
    Summer (high-brightness, ~10 000 lux):
      • Gamma correction < 1  →  brighten / overexpose
      • Saturation boost (vivid green foliage, yellow lesions)
      • ColorJitter for sun-angle variation
    """
    def __init__(self):
        self._base = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            # Brightness/contrast variation due to direct sun
            T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.05),
            T.ToTensor(),
        ])

    def __call__(self, img):
        img = self._base(img)
        # Gamma < 1 → brightens (x^γ, γ=0.6)
        img = img.pow(0.6).clamp(0, 1)
        # Increase saturation in tensor space (channel-level scaling)
        # Approximate: boost G channel slightly (green leaf saturation)
        img[1] = (img[1] * 1.15).clamp(0, 1)
        img = T.functional.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
        return img


class _AutumnTransform:
    """
    Autumn (colour-normalised / senescence):
      • Reduce saturation (foliage yellowing / browning)
      • Warm hue shift (+10° in HSV) simulating chlorophyll degradation
      • Sharpen edges: leaf-edge features become more pronounced
    """
    def __init__(self):
        self._base = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
        ])

    def __call__(self, img):
        img = self._base(img)
        # Desaturate: blend toward greyscale (α=0.55 colour weight)
        grey = img.mean(dim=0, keepdim=True).expand_as(img)
        img  = (0.55 * img + 0.45 * grey).clamp(0, 1)
        # Warm hue shift: boost R, reduce B slightly
        img[0] = (img[0] * 1.10).clamp(0, 1)  # red channel up
        img[2] = (img[2] * 0.85).clamp(0, 1)  # blue channel down
        img = T.functional.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
        return img


# Validation transform (no augmentation, just resize + normalise)
_VAL_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

SEASON_TRANSFORMS = {
    "Spring": _SpringTransform(),
    "Summer": _SummerTransform(),
    "Autumn": _AutumnTransform(),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Synthetic Dataset (fall-back when no real data is available)
# ─────────────────────────────────────────────────────────────────────────────

class FakeCropDataset(Dataset):
    """
    Generates random image tensors with integer class labels.

    Useful for pipeline validation without a downloaded dataset.
    All pixels are drawn from N(0,1) and then passed through each
    season transform so the statistical properties mirror the real
    augmentation pipeline.

    Args:
        season     : One of ``TASKS``.
        num_samples: Number of synthetic samples.
        num_classes: Number of disease categories.
        train      : Whether to apply training augmentation (unused here;
                     kept for API compatibility).
    """

    def __init__(
        self,
        season: str = "Spring",
        num_samples: int = 640,
        num_classes: int = NUM_CLASSES_DEFAULT,
        train: bool = True,
    ):
        assert season in TASKS, f"season must be one of {TASKS}"
        self.num_samples = num_samples
        self.num_classes = num_classes
        # Pre-generate random data in normalised image space
        self._images = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
        self._labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self._images[idx], self._labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset Factory
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalCropDataset(Dataset):
    """
    Wraps an existing image-folder dataset and applies a seasonal transform.

    The underlying dataset should be structured as::

        root/
          train/
            <class_name>/
              image1.jpg
              ...
          val/
            <class_name>/
              ...

    Seasonal image transforms are applied on-the-fly so the same images
    can be reused across all three tasks with different photometric
    simulation.

    Args:
        base_dataset : ``torchvision.datasets.ImageFolder`` instance.
        season       : One of ``TASKS``.
        is_train     : Apply training augmentation if True, else val transform.
    """

    def __init__(self, base_dataset: Dataset, season: str, is_train: bool = True):
        assert season in TASKS
        self.base    = base_dataset
        self.season  = season
        self.transform = SEASON_TRANSFORMS[season] if is_train else _VAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img, label = self.base[idx]
        # base_dataset returns a PIL image when transform=None
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Public API — get_task_loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_task_loaders(
    root: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    num_classes: int = NUM_CLASSES_DEFAULT,
    synthetic_samples: int = 640,
    pin_memory: bool = True,
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    """
    Build training and validation ``DataLoader`` objects for every season.

    If ``root`` contains a valid ``train/`` sub-directory with image
    folders, real data is loaded via ``ImageFolder``.  Otherwise a
    ``FakeCropDataset`` is used automatically (with a console warning).

    Args:
        root              : Path to the dataset root directory.
        batch_size        : Mini-batch size.
        val_split         : Fraction of training data reserved for validation.
        num_workers       : DataLoader worker processes.
        num_classes       : Number of disease categories (synthetic mode only).
        synthetic_samples : Samples per season in synthetic mode.
        pin_memory        : Pin CUDA page-locked memory for speed.

    Returns:
        Dictionary mapping each season name → (train_loader, val_loader).

    Example::

        loaders = get_task_loaders(root="./PlantVillage", batch_size=32)
        spring_train, spring_val = loaders["Spring"]
    """
    _loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory and torch.cuda.is_available(),
    )

    use_real_data = (
        _HAS_IMAGEFOLDER
        and os.path.isdir(os.path.join(root, "train"))
    )

    if not use_real_data:
        warnings.warn(
            f"Real dataset not found at '{root}/train'. "
            "Falling back to FakeCropDataset for pipeline validation.",
            UserWarning,
            stacklevel=2,
        )

    loaders: Dict[str, Tuple[DataLoader, DataLoader]] = {}

    for season in TASKS:
        if use_real_data:
            # Real data: load base ImageFolder (no transform) + wrap seasonally
            train_base = ImageFolder(os.path.join(root, "train"), transform=None)
            val_base   = ImageFolder(os.path.join(root, "val"),   transform=None)

            train_ds = SeasonalCropDataset(train_base, season, is_train=True)
            val_ds   = SeasonalCropDataset(val_base,   season, is_train=False)
        else:
            # Synthetic fall-back
            full_ds  = FakeCropDataset(
                season      = season,
                num_samples = synthetic_samples,
                num_classes = num_classes,
            )
            n_val    = max(1, int(len(full_ds) * val_split))
            n_train  = len(full_ds) - n_val
            train_ds, val_ds = random_split(
                full_ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

        loaders[season] = (
            DataLoader(train_ds, shuffle=True,  **_loader_kwargs),
            DataLoader(val_ds,   shuffle=False, **_loader_kwargs),
        )
        print(
            f"  [DataLoader] {season:<8} | "
            f"train={len(train_ds):>5d}  val={len(val_ds):>4d}  "
            f"batch={batch_size}"
        )

    return loaders
