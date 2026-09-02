"""Main training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --save-dir models/experiment

The Kaggle `test/` folder is never touched here — not for training, not for
validation, not for model selection. Validation is carved out of `train/` at
the patient level (see `patient_id`).
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import ChestXRayDataset, get_transforms
from src.evaluate import plot_training_history
from src.model import ResNet50Classifier
from src.train import Trainer
from src.utils import get_device, set_seed, setup_logging

VAL_FRACTION = 0.15


def patient_id(path: str) -> str:
    """Group key identifying the patient a radiograph belongs to.

    The Kaggle set stores multiple images per patient, so an image-level split
    puts different radiographs of the same person on both sides of the
    train/val boundary and inflates validation accuracy.

    Two naming families encode the patient:
      * PNEUMONIA — ``person1_bacteria_1.jpeg``, ``person1_virus_6.jpeg``
      * NORMAL    — ``IM-0629-0001.jpeg``, ``NORMAL2-IM-1412-0001.jpeg``

    NORMAL files are *not* one-per-patient: 130 of the 1,341 training NORMAL
    images share an ``IM-####`` study id with at least one other image. The
    ``NORMAL2-`` prefix marks a separate acquisition batch, so it is part of the
    key — ``IM-0115`` and ``NORMAL2-IM-0115`` are different patients.
    """
    name = Path(path).name

    m = re.search(r"person(\d+)", name)
    if m:
        return f"person{m.group(1)}"

    m = re.match(r"(NORMAL2-)?IM-(\d+)", name)
    if m:
        return f"{m.group(1) or ''}IM-{m.group(2)}"

    # Unrecognised filename — treat it as its own patient rather than silently
    # merging it into a group it may not belong to.
    return Path(path).stem


def main(config_path: str = "configs/config.yaml", save_dir: str | None = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    save_dir = save_dir or config["paths"]["save_dir"]

    setup_logging(log_file=config["paths"]["log_file"])
    logger = logging.getLogger(__name__)

    set_seed(config["seed"])
    device = get_device()

    # ------------------------------------------------------------------
    # Data — carve 15% of train/ as validation, split by PATIENT not by image.
    # The Kaggle val/ folder (16 images) is too small to select a model against.
    # ------------------------------------------------------------------
    logger.info("Loading dataset …")
    bs   = config["data"]["batch_size"]
    nw   = config["data"]["num_workers"]
    size = config["data"]["img_size"]
    root = config["data"]["data_dir"]

    # Same underlying images, two transform pipelines: the validation subset must
    # not be randomly cropped/flipped/rotated or its accuracy is just noise.
    full_train = ChestXRayDataset(root, split="train", transform=get_transforms("train", size))
    full_eval  = ChestXRayDataset(root, split="train", transform=get_transforms("val",   size))

    groups = np.array([patient_id(p) for p, _ in full_train.samples])
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=config["seed"]
    )
    tr_idx, va_idx = next(splitter.split(np.arange(len(full_train)), groups=groups))

    # A patient must never appear on both sides. Cheap to check, fatal if wrong.
    overlap = set(groups[tr_idx]) & set(groups[va_idx])
    assert not overlap, f"Patient leak across split: {sorted(overlap)[:5]}"

    train_ds = Subset(full_train, tr_idx)
    val_ds   = Subset(full_eval,  va_idx)
    test_ds  = ChestXRayDataset(root, split="test", transform=get_transforms("val", size))

    # Weighted sampler over the training subset only — this is the single
    # correction for the ~2.9:1 class imbalance (the loss stays unweighted).
    train_targets = [full_train.targets[i] for i in tr_idx]
    counts   = np.bincount(train_targets, minlength=2)
    weights  = len(train_targets) / (2 * counts.astype(float))
    sample_w = [float(weights[t]) for t in train_targets]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    loaders = {
        "train": DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=bs, shuffle=False,   num_workers=nw, pin_memory=True),
    }

    val_targets = [full_train.targets[i] for i in va_idx]
    logger.info(f"  patients      : {len(set(groups)):,} across {len(full_train):,} images")
    logger.info(
        f"  train         : {len(tr_idx):,} images, {len(set(groups[tr_idx])):,} patients  "
        f"{dict(zip(ChestXRayDataset.CLASSES, np.bincount(train_targets, minlength=2).tolist(), strict=True))}"
    )
    logger.info(
        f"  val           : {len(va_idx):,} images, {len(set(groups[va_idx])):,} patients  "
        f"{dict(zip(ChestXRayDataset.CLASSES, np.bincount(val_targets, minlength=2).tolist(), strict=True))}"
    )
    logger.info(f"  test (unused) : {len(test_ds):,} images — held out entirely")
    logger.info(f"  sampler weights: {weights.tolist()}")

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
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        save_dir=save_dir,
    )
    history = trainer.fit(loaders["train"], loaders["val"])

    # ------------------------------------------------------------------
    # Save training curves
    # ------------------------------------------------------------------
    plot_training_history(history, save_dir=config["paths"]["output_dir"])
    logger.info(
        "\nTraining finished!\n"
        f"  Best checkpoint : {save_dir}/best_model.pth\n"
        "  Training curves : outputs/training_history.png\n"
        "  Next step       : python scripts/evaluate.py"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--save-dir", default=None,
        help="Checkpoint/history directory (default: paths.save_dir from the config)",
    )
    args = parser.parse_args()
    main(args.config, args.save_dir)
