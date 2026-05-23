"""Streamlit web application for the Medical Image Classifier.

Run:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_transforms
from src.gradcam import GradCAM, explain
from src.model import ResNet50Classifier
from src.utils import get_device, load_checkpoint

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CLASSES = ["NORMAL", "PNEUMONIA"]
CHECKPOINT_PATH = Path("models/best_model.pth")
CLASS_COLORS = {"NORMAL": "#28a745", "PNEUMONIA": "#dc3545"}
CLASS_ICONS = {"NORMAL": "✅", "PNEUMONIA": "⚠️"}

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Image Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — dark card theme matching portfolio screenshot
# ------------------------------------------------------------------

st.markdown("""
<style>
    /* Gradient header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    /* Prediction cards */
    .pred-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 0.8rem;
    }
    .pred-normal  { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .pred-pneumo  { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    /* Tag chips */
    .chip {
        display: inline-block;
        background: #1e2130;
        border: 1px solid #333;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        color: #ccc;
        margin: 2px 3px;
    }
    /* Section divider */
    hr { border-color: #2a2d3a; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model loader (cached)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    device = get_device()
    model = ResNet50Classifier(num_classes=2, pretrained=False)
    if not CHECKPOINT_PATH.exists():
        return None, device
    load_checkpoint(model, str(CHECKPOINT_PATH), device)
    model = model.to(device).eval()
    return model, device

# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def run_inference(model, device, image_pil: Image.Image):
    transform = get_transforms("val", img_size=224)
    tensor = transform(image_pil)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs, tensor


def build_figure(image_pil, cam, img_np, overlay, probs):
    """Return a dark-themed matplotlib figure for display."""
    BG = "#0d1117"
    fig = plt.figure(figsize=(16, 5), facecolor=BG)

    panels = [
        (np.array(image_pil.resize((224, 224))), "Original X-Ray", "gray"),
        (cam, "Grad-CAM", "hot"),
        (overlay, "Overlay", None),
    ]
    for i, (img, title, cmap) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, i)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")
        ax.set_facecolor(BG)

    ax_bar = fig.add_subplot(1, 4, 4)
    colors = [CLASS_COLORS[c] for c in CLASSES]
    bars = ax_bar.barh(CLASSES, probs * 100, color=colors, edgecolor="#555", lw=0.6)
    for bar, p in zip(bars, probs):
        ax_bar.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%", va="center", color="white", fontsize=10,
        )
    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Confidence (%)", color="#aaa", fontsize=9)
    ax_bar.set_title("Probabilities", color="white", fontsize=11, pad=6)
    ax_bar.set_facecolor(BG)
    ax_bar.tick_params(colors="white")
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#333")

    fig.tight_layout(pad=1.0)
    return fig

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

def sidebar():
    with st.sidebar:
        st.markdown("## 🫁 About")
        st.info(
            "**Model:** ResNet-50 (transfer learning)\n\n"
            "**Dataset:** Kaggle Chest X-Ray (5,863 images)\n\n"
            "**Classes:** Normal · Pneumonia\n\n"
            "**Accuracy:** ~94% on test set\n\n"
            "**Explainability:** Grad-CAM heatmaps"
        )
        st.markdown("## 🏷️ Tech Stack")
        tags = ["PyTorch", "ResNet-50", "Grad-CAM", "Streamlit", "OpenCV", "scikit-learn"]
        st.markdown(" ".join(f'<span class="chip">{t}</span>' for t in tags),
                    unsafe_allow_html=True)
        st.markdown("## 📋 How to use")
        st.markdown(
            "1. Upload a frontal chest X-ray (JPG / PNG)\n"
            "2. The model classifies it as **Normal** or **Pneumonia**\n"
            "3. Grad-CAM highlights the regions driving the decision"
        )
        st.markdown("---")
        st.caption("Built by **Abeer Ashraf** · ResNet-50 + Grad-CAM")

# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------

def main():
    # Hero
    st.markdown('<h1 class="hero-title">🫁 Medical Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">ResNet-50 Transfer Learning · Grad-CAM Explainability · 94%+ Accuracy</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    sidebar()

    model, device = load_model()
    model_ready = model is not None

    if not model_ready:
        st.error(
            "**Model checkpoint not found.**  \n"
            "Train the model first:  \n"
            "```bash\npython scripts/train.py\n```"
        )

    uploaded = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Frontal chest X-ray — JPEG or PNG",
    )

    if uploaded is None:
        # Demo placeholder
        st.info("Upload a chest X-ray above to get a diagnosis.")
        return

    if not model_ready:
        st.warning("Model not loaded — cannot analyse. See error above.")
        return

    import tempfile, os
    suffix = os.path.splitext(uploaded.name)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    image_pil = Image.open(tmp_path).convert("RGB").resize((224, 224))
    os.unlink(tmp_path)

    with st.spinner("Analysing …"):
        probs, tensor = run_inference(model, device, image_pil)
        cam, img_np, pred_idx = explain(model, tensor, device)
        overlay = GradCAM.overlay(cam, img_np, alpha=0.45)

    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # ---- Top result strip ----
    col_img, col_pred = st.columns([1, 2], gap="large")
    with col_img:
        st.image(image_pil, caption="Uploaded X-Ray", use_container_width=True)
    with col_pred:
        css_cls = "pred-normal" if pred_class == "NORMAL" else "pred-pneumo"
        icon = CLASS_ICONS[pred_class]
        st.markdown(
            f'<div class="pred-card {css_cls}">{icon} {pred_class}</div>',
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("**Class probabilities**")
        for cls, prob in zip(CLASSES, probs):
            c1, c2 = st.columns([3, 1])
            c1.progress(float(prob), text=cls)
            c2.caption(f"{prob * 100:.1f}%")

    # ---- Grad-CAM visualisation ----
    st.divider()
    st.subheader("Grad-CAM Explainability")
    st.caption(
        "The heatmap highlights the image regions that most influenced the prediction. "
        "Warm colours (red/yellow) = high activation."
    )
    fig = build_figure(image_pil, cam, img_np, overlay, probs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ---- Clinical disclaimer ----
    st.divider()
    st.warning(
        "**Clinical Disclaimer** — This tool is for educational and research purposes only. "
        "It is **not** a medical device and must not be used for clinical diagnosis. "
        "Always consult a qualified radiologist and physician."
    )


if __name__ == "__main__":
    main()
