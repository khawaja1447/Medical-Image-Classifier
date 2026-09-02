"""Unit tests for EarlyStopping, the dataset weighting helpers, and the
patient-level split key. All pure functions — no dataset, no network.

Run:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import patient_id
from src.dataset import ChestXRayDataset
from src.train import EarlyStopping

# ------------------------------------------------------------------
# EarlyStopping
# ------------------------------------------------------------------

class TestEarlyStopping:

    def test_does_not_trigger_while_improving(self):
        es = EarlyStopping(patience=2)
        for score in (80.0, 85.0, 90.0, 95.0):
            assert not es(score)
        assert es.counter == 0

    def test_triggers_after_patience_stalled_epochs(self):
        es = EarlyStopping(patience=3)
        assert not es(90.0)   # sets the best
        assert not es(89.0)   # stall 1
        assert not es(89.5)   # stall 2
        assert es(89.9)       # stall 3 -> stop
        assert es.triggered

    def test_improvement_resets_the_counter(self):
        es = EarlyStopping(patience=3)
        es(80.0)
        es(79.0)
        assert es.counter == 1
        es(95.0)
        assert es.counter == 0
        assert not es.triggered

    def test_improvement_smaller_than_min_delta_does_not_count(self):
        es = EarlyStopping(patience=1, min_delta=0.5)
        es(90.0)
        assert es(90.4), "A +0.4 gain is below min_delta and must count as a stall"

    def test_best_tracks_the_maximum(self):
        es = EarlyStopping(patience=10)
        for score in (70.0, 91.0, 85.0, 88.0):
            es(score)
        assert es.best == pytest.approx(91.0)


# ------------------------------------------------------------------
# Class / sample weighting
# ------------------------------------------------------------------

def _make_dataset(tmp_path, n_normal: int, n_pneumonia: int) -> ChestXRayDataset:
    """Build a real ChestXRayDataset over tiny synthetic JPEGs."""
    for cls, n in (("NORMAL", n_normal), ("PNEUMONIA", n_pneumonia)):
        d = tmp_path / "train" / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray(
                np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            ).save(d / f"{cls.lower()}_{i}.jpeg")
    return ChestXRayDataset(str(tmp_path), split="train")


class TestDatasetWeighting:

    def test_class_weights_are_inverse_to_frequency(self, tmp_path):
        ds = _make_dataset(tmp_path, n_normal=3, n_pneumonia=9)
        w = ds.get_class_weights()
        assert len(w) == 2
        # total / (n_classes * count)  ->  12/(2*3)=2.0, 12/(2*9)=0.667
        assert float(w[0]) == pytest.approx(2.0)
        assert float(w[1]) == pytest.approx(12 / 18)
        assert w[0] > w[1], "The rarer class must get the larger weight"

    def test_class_weights_balance_the_two_classes(self, tmp_path):
        """weight * count must be equal across classes — that is the point."""
        ds = _make_dataset(tmp_path, n_normal=3, n_pneumonia=9)
        w = ds.get_class_weights()
        assert float(w[0]) * 3 == pytest.approx(float(w[1]) * 9)

    def test_class_weights_are_unity_when_balanced(self, tmp_path):
        ds = _make_dataset(tmp_path, n_normal=5, n_pneumonia=5)
        w = ds.get_class_weights()
        assert float(w[0]) == pytest.approx(1.0)
        assert float(w[1]) == pytest.approx(1.0)

    def test_sample_weights_align_with_targets(self, tmp_path):
        ds = _make_dataset(tmp_path, n_normal=3, n_pneumonia=9)
        sw = ds.get_sample_weights()
        cw = ds.get_class_weights()
        assert len(sw) == len(ds) == 12
        for weight, target in zip(sw, ds.targets, strict=True):
            assert weight == pytest.approx(float(cw[target]))

    def test_class_distribution(self, tmp_path):
        ds = _make_dataset(tmp_path, n_normal=3, n_pneumonia=9)
        assert ds.class_distribution() == {"NORMAL": 3, "PNEUMONIA": 9}

    def test_missing_split_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ChestXRayDataset(str(tmp_path), split="nope")


# ------------------------------------------------------------------
# Patient-level split key
# ------------------------------------------------------------------

class TestPatientId:

    def test_pneumonia_images_of_one_patient_share_a_group(self):
        ids = {
            patient_id("data/train/PNEUMONIA/person1_bacteria_1.jpeg"),
            patient_id("data/train/PNEUMONIA/person1_bacteria_2.jpeg"),
            patient_id("data/train/PNEUMONIA/person1_virus_6.jpeg"),
        }
        assert len(ids) == 1, f"Same patient must map to one group, got {ids}"

    def test_different_patients_do_not_collide(self):
        assert patient_id("person1_bacteria_1.jpeg") != patient_id("person11_bacteria_1.jpeg")
        assert patient_id("person2_virus_1.jpeg") != patient_id("person20_virus_1.jpeg")

    def test_normal_images_of_one_study_share_a_group(self):
        """NORMAL files are NOT one-per-patient — IM-0629 appears 4 times."""
        ids = {
            patient_id("IM-0629-0001.jpeg"),
            patient_id("IM-0629-0001-0001.jpeg"),
            patient_id("IM-0629-0001-0002.jpeg"),
        }
        assert len(ids) == 1, f"Same NORMAL study must map to one group, got {ids}"

    def test_normal2_prefix_is_a_distinct_patient(self):
        assert patient_id("IM-0115-0001.jpeg") != patient_id("NORMAL2-IM-0115-0001.jpeg")

    def test_normal_and_pneumonia_never_collide(self):
        assert patient_id("person1_bacteria_1.jpeg") != patient_id("IM-0001-0001.jpeg")

    def test_unrecognised_name_falls_back_to_its_own_group(self):
        assert patient_id("weird_name.jpeg") != patient_id("other_name.jpeg")
