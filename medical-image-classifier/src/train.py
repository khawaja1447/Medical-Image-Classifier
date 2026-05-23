"""Training engine: two-phase strategy with mixed precision and early stopping."""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from .utils import save_checkpoint

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = -float("inf")
        self.triggered = False

    def __call__(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ------------------------------------------------------------------
# Training loop helpers
# ------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    scheduler=None,
) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / total,
        100.0 * correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class Trainer:
    """Two-phase transfer-learning trainer for ResNet-50.

    Phase 1 (warmup_epochs): backbone frozen, head only.
    Phase 2 (remaining):     full fine-tuning with differential LRs.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        save_dir: str = "models",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = GradScaler()
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
        }

    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader, class_weights=None):
        cfg = self.config["training"]
        num_epochs = cfg["num_epochs"]
        warmup = cfg["warmup_epochs"]

        criterion = (
            nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )

        # ---- Phase 1: head only ----
        self.model.freeze_backbone()
        head_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg["head_lr"],
            weight_decay=cfg["weight_decay"],
        )

        logger.info(f"=== Phase 1: head-only training ({warmup} epochs) ===")
        for ep in range(1, warmup + 1):
            self._run_epoch(ep, warmup, train_loader, val_loader,
                            head_optimizer, criterion, phase=1)

        # ---- Phase 2: full fine-tune ----
        self.model.unfreeze_backbone()
        fine_tune_epochs = num_epochs - warmup

        optimizer = optim.AdamW(
            [
                {"params": self.model.features.parameters(), "lr": cfg["backbone_lr"]},
                {"params": self.model.classifier.parameters(), "lr": cfg["head_lr"]},
            ],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-7)
        early_stop = EarlyStopping(patience=cfg["early_stopping_patience"])
        best_val_acc = 0.0

        logger.info(f"=== Phase 2: full fine-tuning ({fine_tune_epochs} epochs) ===")
        for ep in range(1, fine_tune_epochs + 1):
            val_acc = self._run_epoch(
                ep, fine_tune_epochs, train_loader, val_loader,
                optimizer, criterion, scheduler=scheduler, phase=2,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    self.model, ep + warmup, val_acc,
                    self.config, self.save_dir / "best_model.pth"
                )
                logger.info(f"  -> Checkpoint saved (val_acc={val_acc:.2f}%)")

            if early_stop(val_acc):
                logger.info("Early stopping triggered.")
                break

        save_checkpoint(
            self.model, num_epochs, best_val_acc,
            self.config, self.save_dir / "last_model.pth"
        )
        self._dump_history()
        logger.info(f"Done. Best val accuracy: {best_val_acc:.2f}%")
        return self.history

    # ------------------------------------------------------------------

    def _run_epoch(
        self, ep, total_ep, train_loader, val_loader,
        optimizer, criterion, scheduler=None, phase=2,
    ) -> float:
        t0 = time.perf_counter()
        train_loss, train_acc = _train_one_epoch(
            self.model, train_loader, optimizer, criterion,
            self.device, self.scaler,
        )
        val_loss, val_acc, _, _, _ = _evaluate(
            self.model, val_loader, criterion, self.device
        )
        if scheduler is not None:
            scheduler.step()

        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[P{phase}] Epoch {ep:03d}/{total_ep:03d} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
            f"{elapsed:.1f}s"
        )
        return val_acc

    def _dump_history(self):
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
