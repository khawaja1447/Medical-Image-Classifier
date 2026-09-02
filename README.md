# Medical Image Classifier

> ResNet-50 transfer learning for chest X-ray classification — **92.47% accuracy** on a held-out test set, with Grad-CAM explainability

![CI](https://github.com/khawaja1447/Medical-Image-Classifier/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Binary classification of frontal chest X-rays into **NORMAL** vs **PNEUMONIA** using a fine-tuned ResNet-50 backbone with:

- **Two-phase transfer learning** — frozen backbone warm-up followed by full fine-tuning with differential learning rates
- **Class-imbalance handling** — a `WeightedRandomSampler` over the training subset (applied once; see [Methodology](#methodology))
- **Patient-level validation split** — no patient appears on both sides of the train/val boundary
- **Mixed-precision training** — enabled on CUDA, cleanly disabled on CPU
- **Grad-CAM explainability** — per-prediction heatmaps showing which lung regions drove the decision
- **Streamlit demo** — interactive web app for real-time inference and visualisation

---

## Methodology

**The Kaggle `test/` folder was never used for training, validation, early stopping, or model selection.**

This is the claim the headline number rests on, so it is worth stating plainly:

- Training and validation are both carved out of the Kaggle `train/` folder (5,216 images).
- The 624-image `test/` folder is loaded exactly once, by `scripts/evaluate.py`, after training is finished.
- `scripts/train.py` never builds a test loader for optimisation, and `src/dataset.py:get_test_loader()`
  constructs the test split alone — it does not touch `train/` at all.

Every number in the Results table below is reproducible with `python scripts/evaluate.py`, which writes
[`models/test_results.json`](models/test_results.json) — committed, so the claims have an artifact behind them.

---

## Results

Measured on the Kaggle test folder (624 images: 234 NORMAL, 390 PNEUMONIA).

| Metric | Value |
|---|---|
| Test Accuracy | **92.47%** |
| AUC-ROC | **0.9703** |
| Avg. Precision | **0.9775** |
| **Sensitivity** (pneumonia recall) | **0.9308** — 27 of 390 pneumonia cases missed |
| **Specificity** (normal recall) | **0.9145** — 20 of 234 normals flagged |
| F1 (Pneumonia) | 0.94 |
| F1 (Normal) | 0.90 |

Confusion matrix at the default 0.5 threshold:

|  | pred NORMAL | pred PNEUMONIA |
|---|---|---|
| **true NORMAL** | 214 | 20 |
| **true PNEUMONIA** | 27 | 363 |

### Choosing an operating point

Accuracy is the wrong objective for a screening tool. `argmax` implicitly cuts at 0.5, which optimises
overall accuracy; but a missed pneumonia costs far more than a false alarm, so the operating point should
be moved deliberately down the precision-recall curve.

`evaluate_model` reports the most precise threshold that still reaches **recall ≥ 0.98**:

| | default (`argmax`, 0.5) | screening point (0.030) |
|---|---|---|
| Sensitivity | 0.9308 | **0.9821** |
| Specificity | 0.9145 | 0.7735 |
| Precision | 0.9478 | 0.8784 |
| Accuracy | 92.47% | 90.38% |
| Missed pneumonia | 27 | **7** |
| False alarms | 20 | 53 |

Three points of accuracy buys back 20 of the 27 missed cases. That trade is the whole decision, and it is
one a clinician should be making explicitly rather than inheriting from a default `argmax`.

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

Grad-CAM hooks `layer4` as a whole, so the residual addition and the output ReLU are inside the captured
activation — the conventional target for ResNet, and what `model.grad_cam_target_layer` returns.

**Training strategy**

| Phase | Epochs | Layers trained | LR (backbone / head) |
|---|---|---|---|
| Warm-up | 5 | Head only | — / 1e-3 |
| Fine-tune | 20 | Entire network | 1e-5 / 1e-3 |

Scheduler: `CosineAnnealingLR` (Phase 2)
Early stopping: patience = 7 epochs
Both phases checkpoint against the same running best, so a strong warm-up epoch is not discarded.

---

## Project Structure

```
Medical-Image-Classifier/
├── .github/workflows/
│   └── ci.yml               # ruff + pytest on 3.10 and 3.12
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── src/
│   ├── dataset.py           # ChestXRayDataset, transforms, dataloaders
│   ├── model.py             # ResNet50Classifier
│   ├── gradcam.py           # GradCAM class + explain() helper
│   ├── train.py             # Trainer, EarlyStopping, mixed-precision loop
│   ├── evaluate.py          # Metrics, threshold selection, plots
│   └── utils.py             # Seed, device, logging, checkpoint I/O
├── scripts/
│   ├── download_data.py     # Kaggle API downloader
│   ├── train.py             # Training entry point + patient_id() split key
│   ├── evaluate.py          # Test-set evaluation entry point
│   └── predict.py           # Single-image inference + Grad-CAM figure
├── app/
│   └── app.py               # Streamlit web application
├── tests/
│   ├── test_model.py        # Model, Grad-CAM, transforms
│   ├── test_training.py     # EarlyStopping, class weights, patient_id
│   └── test_evaluate.py     # evaluate_model, threshold selection
├── models/
│   ├── training_history.json  # Per-epoch curves (committed)
│   └── test_results.json      # Full test metrics (committed)
├── outputs/                 # Plots & prediction images (gitignored)
├── logs/                    # Training logs (gitignored)
├── data/                    # Dataset (gitignored — download separately)
├── ruff.toml
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/khawaja1447/Medical-Image-Classifier.git
cd Medical-Image-Classifier
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` pulls the **CPU** build of PyTorch by default. On a CUDA machine, drop the
`--extra-index-url` line at the top and install torch from [pytorch.org](https://pytorch.org).

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

Training takes ~15 min on a single GPU (RTX 3060+) or ~3 hours on CPU.
Best checkpoint saved automatically to `models/best_model.pth`.

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Prints accuracy, AUC-ROC, sensitivity, specificity and the screening operating point; writes
`models/test_results.json` and plots to `outputs/`.

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
python -m pytest tests/ -v
ruff check .
```

48 tests covering model output shapes, Grad-CAM behaviour (including the frozen-backbone and `no_grad`
paths), transform correctness, early stopping, class weighting, the patient-ID split key, and the metric
and threshold-selection maths — **no dataset and no network access required**. Both commands run in CI on
Python 3.10 and 3.12.

---

## Grad-CAM Explanation

Grad-CAM computes the gradient of the predicted class score with respect to the feature maps of the last
convolutional stage. The resulting heatmap reveals **which spatial regions** of the X-ray contributed most
to the prediction.

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

The dataset ships with three folders:

| Kaggle folder | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| `train/` | 1,341 | 3,875 | 5,216 |
| `val/` | 8 | 8 | 16 |
| `test/` | 234 | 390 | 624 |

### The split actually used

The shipped `val/` folder holds 16 images — far too few to select a model against (every accuracy on it is
a multiple of 6.25%). `scripts/train.py` therefore **ignores `val/` entirely** and carves its own 15%
validation split out of `train/`:

| Split | Source | Images | Patients | NORMAL | PNEUMONIA |
|---|---|---|---|---|---|
| Train | 85% of Kaggle `train/` | 4,450 | 2,419 | 1,122 | 3,328 |
| Val | 15% of Kaggle `train/` | 766 | 427 | 219 | 547 |
| Test | Kaggle `test/`, untouched | 624 | — | 234 | 390 |

Class imbalance (~2.9:1) is corrected **once**, by a `WeightedRandomSampler` built from the training
subset only. The loss stays unweighted — applying both would correct the same imbalance twice.

### Why the split is by patient, not by image

The Kaggle set contains multiple radiographs per patient, and the filenames say so:
`person1_bacteria_1.jpeg`, `person1_bacteria_2.jpeg`, `person1_virus_6.jpeg`. A `random_split` over the
image list puts different radiographs of the *same* patient on both sides of the boundary, and validation
accuracy then measures partly-memorised patients rather than generalisation.

`scripts/train.py` uses `GroupShuffleSplit` over a `patient_id()` group key, and asserts that no patient
crosses the boundary. The key handles both naming families — note that NORMAL files are **not**
one-per-patient: 130 of the 1,341 training NORMAL images share an `IM-####` study id with at least one
other image, so keying on the filename stem alone would leak them.

This does not affect the reported test number, which was never measured on a validation split of any kind.
It affects how much the validation curve should be *believed*.

---

## Preprocessing note

The model is trained on aspect-squashed images (`Resize((256,256))` → `RandomCrop(224)`), and 604 of the
624 test images are more than 10% off square. Inference must therefore squash the same way — the app and
`predict.py` both hand the raw upload to `get_transforms("val")` and nothing else.

Letterboxing or centre-cropping the upload "to preserve anatomy" is intuitive but wrong here, because it
feeds the model a distribution it never saw. Measured on the same test set and checkpoint:

| Inference preprocessing | Accuracy | AUC | Sensitivity |
|---|---|---|---|
| Squash to 224×224 (as trained) | **92.47%** | 0.9703 | **0.9308** |
| Letterbox to square, then resize | 66.35% | 0.9619 | 0.4615 |
| Centre-crop to square, then resize | 85.42% | 0.9643 | 0.7974 |

Letterboxing would miss more than half of all pneumonia cases. Matching train-time preprocessing wins.

---

## Limitations

- Single dataset, single source. Pneumonia and normal images in this collection differ in acquisition as
  well as pathology, so some of the measured performance is plausibly dataset artifact rather than clinical
  signal. Nothing here is validated against an external cohort.
- 92.47% accuracy is one run with one seed. No confidence interval, no repeated-seed variance.
- Not a medical device. Educational and research use only.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Abeer Ashraf** — BSc IT (AI), Fresh Graduate
Built as a portfolio project demonstrating transfer learning, explainable AI, and production-ready ML code.
