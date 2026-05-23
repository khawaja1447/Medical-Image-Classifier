import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


class ChestXRayDataset(Dataset):
    """Kaggle Chest X-Ray Pneumonia dataset.

    Folder structure expected:
        data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
    """

    CLASSES = ["NORMAL", "PNEUMONIA"]
    CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []

        split_dir = self.root_dir / split
        for class_name in self.CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            label = self.CLASS_TO_IDX[class_name]
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((str(img_path), label))
                    self.targets.append(label)

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {split_dir}. "
                "Run `python scripts/download_data.py` first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.FloatTensor:
        counts = np.bincount(self.targets)
        total = len(self.targets)
        weights = total / (len(self.CLASSES) * counts.astype(float))
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> list[float]:
        class_weights = self.get_class_weights()
        return [float(class_weights[t]) for t in self.targets]

    def class_distribution(self) -> dict:
        counts = np.bincount(self.targets)
        return {cls: int(counts[i]) for i, cls in enumerate(self.CLASSES)}


def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
) -> tuple[dict, dict]:
    datasets = {
        split: ChestXRayDataset(
            root_dir=data_dir,
            split=split,
            transform=get_transforms(split, img_size),
        )
        for split in ("train", "val", "test")
    }

    sample_weights = datasets["train"].get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders, datasets
