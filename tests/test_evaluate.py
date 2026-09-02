"""Unit tests for evaluate_model and the screening-threshold selection.

Uses a scripted stub model so the expected metrics are known exactly.

Run:
    python -m pytest tests/ -v
"""

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
import torch.nn as nn

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_model, high_sensitivity_operating_point


class _ScriptedModel(nn.Module):
    """Returns logits whose softmax P(pneumonia) equals a pre-set score list."""

    def __init__(self, scores):
        super().__init__()
        self.scores = list(scores)
        self.pos = 0

    def forward(self, x):
        n = x.shape[0]
        chunk = self.scores[self.pos:self.pos + n]
        self.pos += n
        rows = []
        for p in chunk:
            p = min(max(p, 1e-6), 1 - 1e-6)
            rows.append([0.0, math.log(p / (1 - p))])  # softmax -> [1-p, p]
        return torch.tensor(rows, dtype=torch.float32)


# 4 NORMAL then 4 PNEUMONIA. At the 0.5 argmax threshold this gives
# tn=3, fp=1, fn=1, tp=3  ->  accuracy 75%, sensitivity 0.75, specificity 0.75.
LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
SCORES = [0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.95]


@pytest.fixture
def loader():
    """Two batches of 4, in the (images, labels) shape a DataLoader yields."""
    return [
        (torch.zeros(4, 3, 8, 8), torch.tensor(LABELS[:4])),
        (torch.zeros(4, 3, 8, 8), torch.tensor(LABELS[4:])),
    ]


class TestEvaluateModel:

    def test_core_metrics(self, loader, tmp_path):
        m = evaluate_model(_ScriptedModel(SCORES), loader, torch.device("cpu"), save_dir=tmp_path)
        assert m["n_test"] == 8
        assert m["accuracy"] == pytest.approx(75.0)
        assert m["sensitivity"] == pytest.approx(0.75)
        assert m["specificity"] == pytest.approx(0.75)

    def test_confusion_matrix_counts(self, loader, tmp_path):
        m = evaluate_model(_ScriptedModel(SCORES), loader, torch.device("cpu"), save_dir=tmp_path)
        assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}

    def test_sensitivity_matches_pneumonia_recall(self, loader, tmp_path):
        """The two must agree — they are the same quantity under two names."""
        m = evaluate_model(_ScriptedModel(SCORES), loader, torch.device("cpu"), save_dir=tmp_path)
        assert m["sensitivity"] == pytest.approx(m["per_class"]["PNEUMONIA"]["recall"])
        assert m["specificity"] == pytest.approx(m["per_class"]["NORMAL"]["recall"])

    def test_plots_are_written(self, loader, tmp_path):
        evaluate_model(_ScriptedModel(SCORES), loader, torch.device("cpu"), save_dir=tmp_path)
        for name in ("confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"):
            assert (tmp_path / name).exists(), f"missing {name}"

    def test_result_is_json_serialisable(self, loader, tmp_path):
        """The dict is committed as models/test_results.json — no numpy types."""
        import json
        m = evaluate_model(_ScriptedModel(SCORES), loader, torch.device("cpu"), save_dir=tmp_path)
        json.dumps(m)  # raises TypeError on np.float64 / np.int64


class TestScreeningThreshold:

    def test_reaches_the_requested_recall(self):
        op = high_sensitivity_operating_point(
            np.array(LABELS), np.array(SCORES), target_recall=1.0
        )
        assert op["achievable"]
        assert op["sensitivity"] >= 1.0
        assert op["missed_pneumonia"] == 0

    def test_trades_specificity_for_sensitivity(self):
        """The whole point: more recall than argmax, at a cost in precision."""
        op = high_sensitivity_operating_point(
            np.array(LABELS), np.array(SCORES), target_recall=1.0
        )
        assert op["threshold"] <= 0.4
        assert op["sensitivity"] > 0.75, "must beat the 0.5-threshold sensitivity"
        assert op["specificity"] <= 0.75, "and give something up for it"

    def test_reports_unachievable_rather_than_guessing(self):
        """A perfect-recall demand on a model that cannot deliver must say so."""
        op = high_sensitivity_operating_point(
            np.array([0, 1]), np.array([0.9, 0.1]), target_recall=1.01
        )
        assert op["achievable"] is False
        assert "threshold" not in op

    def test_picks_the_most_precise_eligible_threshold(self):
        op = high_sensitivity_operating_point(
            np.array(LABELS), np.array(SCORES), target_recall=0.75
        )
        # Several thresholds reach recall 0.75; the chosen one must be at least
        # as precise as the default 0.5 cut (which scores 3/4 = 0.75).
        assert op["precision"] >= 0.75
