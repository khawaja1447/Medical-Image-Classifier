"""Run full test-set evaluation on the best saved checkpoint.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint models/best_model.pth

Writes the full metrics dict to models/test_results.json so the headline
numbers in the README have a committed artifact behind them.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_test_loader
from src.evaluate import evaluate_model
from src.model import ResNet50Classifier
from src.utils import get_device, load_checkpoint, set_seed, setup_logging


def main(
    config_path: str = "configs/config.yaml",
    checkpoint: str | None = None,
    results_path: str | None = None,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(config["seed"])
    device = get_device()

    # Only the held-out test split — evaluation must not depend on train/ being
    # present, and must never build a sampler over the training set.
    test_loader, test_ds = get_test_loader(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        img_size=config["data"]["img_size"],
        num_workers=config["data"]["num_workers"],
    )
    logger.info(f"Test set: {len(test_ds):,} images  {test_ds.class_distribution()}")

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
        model, test_loader, device,
        save_dir=config["paths"]["output_dir"],
    )

    cm = metrics["confusion_matrix"]
    op = metrics["screening_operating_point"]

    print("\n" + "=" * 58)
    print("  FINAL TEST RESULTS")
    print("=" * 58)
    print(f"  Test images    : {metrics['n_test']:,}")
    print(f"  Accuracy       : {metrics['accuracy']:.2f}%")
    print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print("-" * 58)
    print(
        f"  Sensitivity    : {metrics['sensitivity']:.4f}"
        f"   ({cm['fn']} of {cm['fn'] + cm['tp']} pneumonia cases missed)"
    )
    print(
        f"  Specificity    : {metrics['specificity']:.4f}"
        f"   ({cm['fp']} of {cm['fp'] + cm['tn']} normals flagged)"
    )
    if op["achievable"]:
        print("-" * 58)
        print(f"  Screening operating point (recall >= {op['target_recall']:.2f}):")
        print(f"    threshold    : {op['threshold']:.4f}   (vs 0.5 for argmax)")
        print(f"    sensitivity  : {op['sensitivity']:.4f}")
        print(f"    precision    : {op['precision']:.4f}")
        print(f"    specificity  : {op['specificity']:.4f}")
        print(f"    accuracy     : {op['accuracy']:.2f}%")
        print(
            f"    trade-off    : {op['missed_pneumonia']} missed, "
            f"{op['false_alarms']} false alarms"
        )
    print("=" * 58)

    out = Path(results_path or (Path(config["paths"]["save_dir"]) / "test_results.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": Path(ckpt_path).as_posix(),
        "checkpoint_epoch": epoch,
        "checkpoint_val_acc": val_acc,
        **metrics,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nMetrics saved to: {out}")
    print(f"Plots saved to  : {config['paths']['output_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--results", default=None,
        help="Where to write the metrics JSON (default: models/test_results.json)",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.results)
