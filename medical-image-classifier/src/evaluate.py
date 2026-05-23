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

def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    save_dir: str = "outputs",
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

    logger.info(f"Test Accuracy : {acc:.2f}%")
    logger.info(f"AUC-ROC       : {auc:.4f}")
    logger.info(f"Avg Precision : {ap:.4f}")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=CLASSES))

    plot_confusion_matrix(y_true, y_pred, save_dir)
    plot_roc_curve(y_true, y_prob[:, 1], auc, save_dir)
    plot_precision_recall(y_true, y_prob[:, 1], ap, save_dir)

    return {"accuracy": acc, "auc_roc": auc, "avg_precision": ap}


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


def plot_precision_recall(y_true, y_scores, avg_prec: float, save_dir: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, "r-", lw=2.5, label=f"ResNet-50  (AP = {avg_prec:.4f})")
    ax.axhline(baseline, color="k", lw=1, linestyle="--", label=f"Random (AP = {baseline:.2f})")
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
