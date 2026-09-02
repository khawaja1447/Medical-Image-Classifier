"""Evaluation utilities: metrics, plots, full test-set evaluation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)
CLASSES = ["NORMAL", "PNEUMONIA"]


# ------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------

def high_sensitivity_operating_point(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_recall: float = 0.98,
) -> dict:
    """Pick the highest-precision threshold that still reaches ``target_recall``.

    ``argmax`` at an implicit 0.5 threshold optimises accuracy. For a screening
    tool that is the wrong objective: a missed pneumonia costs far more than a
    false alarm, so the operating point is deliberately moved down the
    precision-recall curve until recall clears ``target_recall``.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve appends a (recall=0, precision=1) endpoint that has
    # no threshold behind it; drop it so the arrays line up with `thresholds`.
    precision, recall = precision[:-1], recall[:-1]

    eligible = recall >= target_recall
    if not eligible.any():
        return {
            "target_recall": target_recall,
            "achievable": False,
            "note": f"No threshold reaches recall >= {target_recall:.2f}.",
        }

    # Among thresholds that clear the recall floor, take the most precise one.
    idx = int(np.argmax(np.where(eligible, precision, -np.inf)))
    threshold = float(thresholds[idx])

    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "target_recall": target_recall,
        "achievable": True,
        "threshold": threshold,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "accuracy": 100.0 * float((y_pred == y_true).mean()),
        "missed_pneumonia": int(fn),
        "false_alarms": int(fp),
    }


# ------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    save_dir: str = "outputs",
    target_recall: float = 0.98,
) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = 100.0 * (y_pred == y_true).mean()
    auc = roc_auc_score(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])

    # Sensitivity / specificity — the first two numbers a clinician asks for.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    report = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    operating_point = high_sensitivity_operating_point(
        y_true, y_prob[:, 1], target_recall=target_recall
    )

    logger.info(f"Test Accuracy : {acc:.2f}%")
    logger.info(f"AUC-ROC       : {auc:.4f}")
    logger.info(f"Avg Precision : {ap:.4f}")
    logger.info(
        f"Sensitivity   : {sensitivity:.4f}  "
        f"({fn} of {tp + fn} pneumonia cases missed)"
    )
    logger.info(
        f"Specificity   : {specificity:.4f}  "
        f"({fp} of {tn + fp} normals flagged)"
    )
    logger.info("\n" + classification_report(y_true, y_pred, target_names=CLASSES))

    if operating_point["achievable"]:
        logger.info(
            f"Screening operating point (recall >= {target_recall:.2f}): "
            f"threshold={operating_point['threshold']:.4f}  "
            f"sensitivity={operating_point['sensitivity']:.4f}  "
            f"precision={operating_point['precision']:.4f}  "
            f"specificity={operating_point['specificity']:.4f}"
        )

    plot_confusion_matrix(y_true, y_pred, save_dir)
    plot_roc_curve(y_true, y_prob[:, 1], auc, save_dir)
    plot_precision_recall(y_true, y_prob[:, 1], ap, save_dir, operating_point)

    return {
        "n_test": len(y_true),
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "avg_precision": float(ap),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "per_class": {
            cls: {k: float(v) for k, v in report[cls].items()} for cls in CLASSES
        },
        "screening_operating_point": operating_point,
    }


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrix", fontsize=16, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Raw Counts", "Normalised"],
        strict=True,
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            ax=ax, square=True, linewidths=0.5, cbar=True,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: confusion_matrix.png")


def plot_roc_curve(y_true, y_scores, auc_score: float, save_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "b-", lw=2.5, label=f"ResNet-50  (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="blue")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Chest X-Ray Classifier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: roc_curve.png")


def plot_precision_recall(
    y_true, y_scores, avg_prec: float, save_dir: Path, operating_point: dict | None = None
):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, "r-", lw=2.5, label=f"ResNet-50  (AP = {avg_prec:.4f})")
    ax.axhline(baseline, color="k", lw=1, linestyle="--", label=f"Random (AP = {baseline:.2f})")
    if operating_point and operating_point.get("achievable"):
        ax.plot(
            operating_point["sensitivity"], operating_point["precision"],
            "o", ms=10, mfc="none", mec="black", mew=2,
            label=(
                f"Screening point  (recall {operating_point['sensitivity']:.3f}, "
                f"precision {operating_point['precision']:.3f})"
            ),
        )
    ax.fill_between(recall, precision, alpha=0.08, color="red")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: precision_recall_curve.png")


def plot_training_history(history: dict, save_dir: str = "outputs"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-o", ms=4, label="Validation")
    axes[0].set_title("Loss per Epoch", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", ms=4, label="Train")
    axes[1].plot(epochs, history["val_acc"], "r-o", ms=4, label="Validation")
    axes[1].set_title("Accuracy per Epoch", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Training History", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: training_history.png")
