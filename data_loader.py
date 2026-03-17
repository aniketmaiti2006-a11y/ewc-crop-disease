import os
import warnings
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from torchvision.datasets import ImageFolder
    _HAS_IMAGEFOLDER = True
except ImportError:
    _HAS_IMAGEFOLDER = False


TASKS: Tuple[str, ...] = ("Spring", "Summer", "Autumn")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224
NUM_CLASSES_DEFAULT = 10

SEASON_NOISE_PROFILES = {
    "Spring": {"gamma": 1.6, "noise_std": 0.02, "color_shift": (1.0, 1.0, 1.0)},
    "Summer": {"gamma": 0.6, "noise_std": 0.01, "color_shift": (1.0, 1.15, 1.0)},
    "Autumn": {"gamma": 1.0, "noise_std": 0.015, "color_shift": (1.10, 0.9, 0.85)},
}


class _SpringTransform:
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
        img = img.pow(1.6).clamp(0, 1)
        img = (img + torch.randn_like(img) * 0.02).clamp(0, 1)
        img = T.functional.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
        return img


class _SummerTransform:
    def __init__(self):
        self._base = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.05),
            T.ToTensor(),
        ])

    def __call__(self, img):
        img = self._base(img)
        img = img.pow(0.6).clamp(0, 1)
        img[1] = (img[1] * 1.15).clamp(0, 1)
        img = T.functional.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
        return img


class _AutumnTransform:
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
        grey = img.mean(dim=0, keepdim=True).expand_as(img)
        img  = (0.55 * img + 0.45 * grey).clamp(0, 1)
        img[0] = (img[0] * 1.10).clamp(0, 1)
        img[2] = (img[2] * 0.85).clamp(0, 1)
        img = T.functional.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
        return img


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


def _generate_class_prototypes(num_classes: int, channels: int = 3,
                                size: int = IMAGE_SIZE, seed: int = 12345):
    pi = 3.14159265
    rows = torch.arange(size, dtype=torch.float32).unsqueeze(1).expand(size, size)
    cols = torch.arange(size, dtype=torch.float32).unsqueeze(0).expand(size, size)

    prototypes = []
    for c in range(num_classes):
        freq_x = float((c % 5) + 1)
        freq_y = float((c // 5) + 1)
        diag_freq = float(c + 1)

        proto = torch.zeros(channels, size, size)
        for ch in range(channels):
            g = torch.Generator().manual_seed(seed + c * 100 + ch)
            channel_offset = torch.rand(1, generator=g).item() * 0.3
            pattern = (
                0.5
                + 0.2 * torch.sin(2.0 * pi * freq_x * rows / size)
                + 0.2 * torch.cos(2.0 * pi * freq_y * cols / size)
                + 0.1 * torch.sin(2.0 * pi * diag_freq * (rows + cols) / (size * 2))
                + channel_offset
            )
            proto[ch] = pattern.clamp(0, 1)
        prototypes.append(proto)
    return prototypes


class FakeCropDataset(Dataset):

    def __init__(
        self,
        season: str = "Spring",
        num_samples: int = 2000,
        num_classes: int = NUM_CLASSES_DEFAULT,
        train: bool = True,
    ):
        assert season in TASKS, f"season must be one of {TASKS}"
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.season = season

        prototypes = _generate_class_prototypes(num_classes)
        profile = SEASON_NOISE_PROFILES[season]

        noise_std = 0.08 if train else 0.02

        self._images = torch.zeros(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
        self._labels = torch.zeros(num_samples, dtype=torch.long)

        gen = torch.Generator()
        gen.manual_seed(hash((season, train)) % (2**31))

        for i in range(num_samples):
            cls = i % num_classes
            self._labels[i] = cls
            img = prototypes[cls].clone()
            img = (img + torch.randn_like(img) * noise_std).clamp(0, 1)
            img = img.pow(profile["gamma"]).clamp(0, 1)
            r_s, g_s, b_s = profile["color_shift"]
            img[0] = (img[0] * r_s).clamp(0, 1)
            img[1] = (img[1] * g_s).clamp(0, 1)
            img[2] = (img[2] * b_s).clamp(0, 1)
            self._images[i] = img

        perm = torch.randperm(num_samples, generator=gen)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        self._images = (self._images - mean) / std

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self._images[idx], self._labels[idx]


class SeasonalCropDataset(Dataset):

    def __init__(self, base_dataset: Dataset, season: str, is_train: bool = True):
        assert season in TASKS
        self.base    = base_dataset
        self.season  = season
        self.transform = SEASON_TRANSFORMS[season] if is_train else _VAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img, label = self.base[idx]
        return self.transform(img), label


def get_task_loaders(
    root: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    num_classes: int = NUM_CLASSES_DEFAULT,
    synthetic_samples: int = 2000,
    pin_memory: bool = True,
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
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
            train_base = ImageFolder(os.path.join(root, "train"), transform=None)
            val_base   = ImageFolder(os.path.join(root, "val"),   transform=None)

            train_ds = SeasonalCropDataset(train_base, season, is_train=True)
            val_ds   = SeasonalCropDataset(val_base,   season, is_train=False)
        else:
            n_train_samples = int(synthetic_samples * (1 - val_split))
            n_val_samples = synthetic_samples - n_train_samples

            train_ds = FakeCropDataset(
                season=season,
                num_samples=n_train_samples,
                num_classes=num_classes,
                train=True,
            )
            val_ds = FakeCropDataset(
                season=season,
                num_samples=n_val_samples,
                num_classes=num_classes,
                train=False,
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
