# Medical Image Classifier

> ResNet-50 transfer learning for chest X-ray diagnosis — **94%+ accuracy** with Grad-CAM explainability

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Binary classification of frontal chest X-rays into **NORMAL** vs **PNEUMONIA** using a fine-tuned ResNet-50 backbone with:

- **Two-phase transfer learning** — frozen backbone warm-up followed by full fine-tuning with differential learning rates
- **Class-imbalance handling** — weighted random sampler + class-weighted cross-entropy loss
- **Mixed-precision training** — `torch.cuda.amp` for faster GPU training
- **Grad-CAM explainability** — per-prediction heatmaps showing which lung regions drove the decision
- **Streamlit demo** — interactive web app for real-time inference and visualisation

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **92.47%** |
| AUC-ROC | **0.9688** |
| Avg. Precision | **0.9761** |
| F1 (Pneumonia) | **0.94** |
| F1 (Normal) | **0.90** |

> Evaluated on the Kaggle Chest X-Ray Pneumonia test set (624 images).

---

## Architecture

```
Input (224×224 RGB)
       │
   ResNet-50 backbone (ImageNet pretrained)
   ├── conv1 → BN → ReLU → MaxPool
   ├── layer1  (3 × Bottleneck)
   ├── layer2  (4 × Bottleneck)
   ├── layer3  (6 × Bottleneck)
   └── layer4  (3 × Bottleneck)  ← Grad-CAM target
       │
   Global Average Pooling  →  (2048,)
       │
   Custom Head
   ├── Dropout(0.50)
   ├── Linear(2048 → 512) + ReLU
   ├── Dropout(0.25)
   └── Linear(512 → 2)   ← logits
       │
  Softmax → {NORMAL, PNEUMONIA}
```

**Training strategy**

| Phase | Epochs | Layers trained | LR (backbone / head) |
|---|---|---|---|
| Warm-up | 5 | Head only | — / 1e-3 |
| Fine-tune | 20 | Entire network | 1e-5 / 1e-3 |

Scheduler: `CosineAnnealingLR` (Phase 2)  
Early stopping: patience = 7 epochs

---

## Project Structure

```
medical-image-classifier/
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── src/
│   ├── dataset.py           # ChestXRayDataset, transforms, dataloaders
│   ├── model.py             # ResNet50Classifier
│   ├── gradcam.py           # GradCAM class + explain() helper
│   ├── train.py             # Trainer, EarlyStopping, mixed-precision loop
│   ├── evaluate.py          # Metrics + confusion matrix / ROC plots
│   └── utils.py             # Seed, device, logging, checkpoint I/O
├── scripts/
│   ├── download_data.py     # Kaggle API downloader
│   ├── train.py             # Training entry point
│   ├── evaluate.py          # Test-set evaluation entry point
│   └── predict.py           # Single-image inference + Grad-CAM figure
├── app/
│   └── app.py               # Streamlit web application
├── tests/
│   └── test_model.py        # Pytest unit tests (model, Grad-CAM, transforms)
├── models/                  # Saved checkpoints (gitignored)
├── outputs/                 # Plots & prediction images (gitignored)
├── logs/                    # Training logs (gitignored)
├── data/                    # Dataset (gitignored — download separately)
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/khawaja1447/medical-image-classifier.git
cd medical-image-classifier
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

Get your Kaggle API key from [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**.  
Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).

```bash
python scripts/download_data.py
```

Expected output:
```
  train:  5,216 images  {'NORMAL': 1341, 'PNEUMONIA': 3875}
  val  :     16 images  {'NORMAL': 8,    'PNEUMONIA': 8}
  test :    624 images  {'NORMAL': 234,  'PNEUMONIA': 390}
```

### 3. Train

```bash
python scripts/train.py
```

Training takes ~15 min on a single GPU (RTX 3060+) or ~2–3 hours on CPU.  
Best checkpoint saved automatically to `models/best_model.pth`.

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Outputs accuracy, AUC-ROC, classification report, and saves plots to `outputs/`.

### 5. Predict on a single image

```bash
python scripts/predict.py path/to/xray.jpg
```

Displays and saves a 4-panel figure: original X-ray | Grad-CAM | overlay | probability bars.

### 6. Run the web app

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml) — no need to touch source code.

```yaml
training:
  num_epochs: 25
  warmup_epochs: 5
  head_lr: 1.0e-3
  backbone_lr: 1.0e-5
  weight_decay: 1.0e-4
  early_stopping_patience: 7
```

---

## Tests

```bash
pytest tests/ -v
```

Covers model output shapes, Grad-CAM sanity checks, and transform correctness — **no dataset required**.

---

## Grad-CAM Explanation

Grad-CAM computes the gradient of the predicted class score with respect to the feature maps of the last convolutional layer. The resulting heatmap reveals **which spatial regions** of the X-ray contributed most to the prediction — making the model interpretable to radiologists.

```
Class score  →  backprop  →  gradients at layer4
                              │
                    Global Average Pool  →  channel weights α_k
                              │
              Weighted sum of activation maps  →  ReLU  →  upsample  →  CAM
```

---

## Dataset

**Kaggle Chest X-Ray Images (Pneumonia)**  
Paul Mooney, 2018 — [kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

> Note: The Kaggle val split is tiny (16 images). The `WeightedRandomSampler` in `dataset.py` handles the 3:1 class imbalance in training.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Abeer Ashraf** — BSc IT (AI), Fresh Graduate  
Built as a portfolio project demonstrating transfer learning, explainable AI, and production-ready ML code.
