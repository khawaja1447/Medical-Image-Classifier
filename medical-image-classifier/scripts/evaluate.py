"""Run full test-set evaluation on the best saved checkpoint.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint models/best_model.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.evaluate import evaluate_model
from src.model import ResNet50Classifier
from src.utils import get_device, load_checkpoint, set_seed, setup_logging


def main(config_path: str = "configs/config.yaml", checkpoint: str = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(config["seed"])
    device = get_device()

    loaders, _ = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        img_size=config["data"]["img_size"],
        num_workers=config["data"]["num_workers"],
    )

    model = ResNet50Classifier(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=False,
    )

    ckpt_path = checkpoint or (Path(config["paths"]["save_dir"]) / "best_model.pth")
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Train first: python scripts/train.py")
        sys.exit(1)

    val_acc, epoch = load_checkpoint(model, ckpt_path, device)
    logger.info(f"Loaded checkpoint from epoch {epoch}  (val_acc={val_acc:.2f}%)")
    model = model.to(device)

    metrics = evaluate_model(
        model, loaders["test"], device,
        save_dir=config["paths"]["output_dir"],
    )

    print("\n" + "=" * 45)
    print("  FINAL TEST RESULTS")
    print("=" * 45)
    print(f"  Accuracy       : {metrics['accuracy']:.2f}%")
    print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print("=" * 45)
    print(f"\nPlots saved to: {config['paths']['output_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
