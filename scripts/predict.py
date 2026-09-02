"""Single-image prediction with Grad-CAM visualisation.

Usage:
    python scripts/predict.py path/to/xray.jpg
    python scripts/predict.py path/to/xray.jpg --checkpoint models/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_transforms
from src.gradcam import GradCAM, explain
from src.model import ResNet50Classifier
from src.utils import get_device, load_checkpoint

CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_COLORS = {"NORMAL": "#28a745", "PNEUMONIA": "#dc3545"}


def predict_and_explain(
    image_path: str,
    checkpoint_path: str = "models/best_model.pth",
    output_dir: str = "outputs",
    show: bool = True,
):
    device = get_device()

    # --- Model ---
    model = ResNet50Classifier(num_classes=2, pretrained=False)
    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("        Train first: python scripts/train.py")
        sys.exit(1)
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device).eval()

    # --- Image ---
    image_pil = Image.open(image_path).convert("RGB")
    transform = get_transforms("val", img_size=224)
    tensor = transform(image_pil)

    # --- Inference ---
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # --- Grad-CAM ---
    cam, img_np, _ = explain(model, tensor, device)
    overlay = GradCAM.overlay(cam, img_np, alpha=0.45)

    # --- Figure ---
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    color = CLASS_COLORS[pred_class]

    fig.suptitle(
        f"Prediction: {pred_class}  ({confidence:.1f}% confidence)",
        fontsize=18, fontweight="bold", color=color, y=1.01,
    )

    panels = [
        (image_pil.resize((224, 224)), "Original X-Ray", "gray"),
        (cam, "Grad-CAM Heatmap", "hot"),
        (overlay, "Overlay", None),
    ]

    for i, (img, title, cmap) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, i)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    # Probability bar chart
    ax_bar = fig.add_subplot(1, 4, 4)
    bars = ax_bar.barh(
        CLASSES, probs * 100,
        color=[CLASS_COLORS[c] for c in CLASSES],
        edgecolor="white", linewidth=0.5,
    )
    for bar, prob in zip(bars, probs, strict=True):
        ax_bar.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%", va="center", color="white", fontsize=11,
        )
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Confidence (%)", color="white")
    ax_bar.set_title("Class Probabilities", color="white", fontsize=12, pad=8)
    ax_bar.set_facecolor("#0d1117")
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"prediction_{Path(image_path).stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")

    if show:
        plt.show()
    plt.close(fig)

    print(f"\nPrediction : {pred_class}")
    print(f"Confidence : {confidence:.1f}%")
    print(f"Saved to   : {out_path}")
    return {"class": pred_class, "confidence": confidence, "probs": probs.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict chest X-ray diagnosis")
    parser.add_argument("image", help="Path to chest X-ray image (JPG/PNG)")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    args = parser.parse_args()
    predict_and_explain(args.image, args.checkpoint, args.output_dir, show=not args.no_show)
