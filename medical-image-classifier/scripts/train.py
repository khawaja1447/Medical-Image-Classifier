"""Main training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import ChestXRayDataset, get_transforms
from src.evaluate import plot_training_history
from src.model import ResNet50Classifier
from src.train import Trainer
from src.utils import get_device, set_seed, setup_logging


def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(log_file=config["paths"]["log_file"])
    logger = logging.getLogger(__name__)

    set_seed(config["seed"])
    device = get_device()

    # ------------------------------------------------------------------
    # Data — carve 15% of train as validation (avoids the tiny 16-img val set)
    # ------------------------------------------------------------------
    logger.info("Loading dataset …")
    bs   = config["data"]["batch_size"]
    nw   = config["data"]["num_workers"]
    size = config["data"]["img_size"]
    root = config["data"]["data_dir"]

    full_train = ChestXRayDataset(root, split="train", transform=get_transforms("train", size))
    test_ds    = ChestXRayDataset(root, split="test",  transform=get_transforms("val",   size))

    val_len   = int(0.15 * len(full_train))
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    # Weighted sampler on the training subset
    train_targets = [full_train.targets[i] for i in train_ds.indices]
    import numpy as np, torch
    counts  = np.bincount(train_targets)
    weights = len(train_targets) / (2 * counts.astype(float))
    sample_w = [float(weights[t]) for t in train_targets]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    loaders = {
        "train": DataLoader(train_ds, batch_size=bs, sampler=sampler,  num_workers=nw, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=bs, shuffle=False,    num_workers=nw, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=bs, shuffle=False,    num_workers=nw, pin_memory=True),
    }
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

    logger.info(f"  train: {train_len:,} samples")
    logger.info(f"  val  : {val_len:,} samples  (15% split from train)")
    logger.info(f"  test : {len(test_ds):,} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Building ResNet-50 model …")
    model = ResNet50Classifier(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=config["model"]["pretrained"],
    )
    logger.info(f"  Total params:     {model.count_parameters(False):,}")
    logger.info(f"  Trainable params: {model.count_parameters(True):,}  (before unfreeze)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    class_weights = full_train.get_class_weights()
    logger.info(f"  Class weights:    {class_weights.tolist()}")

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        save_dir=config["paths"]["save_dir"],
    )
    history = trainer.fit(loaders["train"], loaders["val"], class_weights=class_weights)

    # ------------------------------------------------------------------
    # Save training curves
    # ------------------------------------------------------------------
    plot_training_history(history, save_dir=config["paths"]["output_dir"])
    logger.info(
        "\nTraining finished!\n"
        "  Best checkpoint : models/best_model.pth\n"
        "  Training curves : outputs/training_history.png\n"
        "  Next step       : python scripts/evaluate.py"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
