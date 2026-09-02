"""Unit tests for model architecture, Grad-CAM, and transforms.

Run:
    python -m pytest tests/ -v

No dataset and no network access required: the backbone is built with random
weights (`pretrained=False`), so these run offline and in CI.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_transforms
from src.gradcam import GradCAM, explain
from src.model import ResNet50Classifier

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def model_cpu():
    """A randomly-initialised classifier on CPU.

    Function-scoped on purpose. These tests mutate the model (freeze/unfreeze),
    and a module-scoped fixture leaked that state into whichever test ran next
    — which is exactly why the frozen-backbone Grad-CAM crash went unnoticed.
    """
    return ResNet50Classifier(num_classes=2, dropout=0.5, pretrained=False)


@pytest.fixture
def dummy_batch():
    """A batch of 4 random 224x224 images."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def dummy_image_tensor():
    return torch.randn(3, 224, 224)


# ------------------------------------------------------------------
# Model architecture tests
# ------------------------------------------------------------------

class TestResNet50Classifier:

    def test_output_shape(self, model_cpu, dummy_batch):
        model_cpu.eval()
        with torch.no_grad():
            out = model_cpu(dummy_batch)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_output_shape_single(self, model_cpu, dummy_image_tensor):
        model_cpu.eval()
        with torch.no_grad():
            out = model_cpu(dummy_image_tensor.unsqueeze(0))
        assert out.shape == (1, 2)

    def test_softmax_sums_to_one(self, model_cpu, dummy_batch):
        model_cpu.eval()
        with torch.no_grad():
            logits = model_cpu(dummy_batch)
            probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_freeze_backbone(self, model_cpu):
        model_cpu.freeze_backbone()
        trainable = model_cpu.count_parameters(trainable_only=True)
        total = model_cpu.count_parameters(trainable_only=False)
        assert trainable < total, "Freezing backbone should reduce trainable count"

    def test_unfreeze_backbone(self, model_cpu):
        model_cpu.freeze_backbone()
        model_cpu.unfreeze_backbone()
        trainable = model_cpu.count_parameters(trainable_only=True)
        total = model_cpu.count_parameters(trainable_only=False)
        assert trainable == total, "After unfreezing, all params should be trainable"

    def test_fixture_is_not_shared_between_tests(self, model_cpu):
        """Guards the fixture scope: every test must get an unfrozen model."""
        trainable = model_cpu.count_parameters(trainable_only=True)
        total = model_cpu.count_parameters(trainable_only=False)
        assert trainable == total, "Fixture leaked frozen state from another test"

    def test_unfreeze_last_n_layers(self, model_cpu):
        model_cpu.freeze_backbone()
        model_cpu.unfreeze_last_n_layers(n=1)
        stages = list(model_cpu.features.children())
        assert all(p.requires_grad for p in stages[-1].parameters())
        assert not any(p.requires_grad for p in stages[0].parameters())

    def test_grad_cam_target_layer_is_layer4(self, model_cpu):
        """The README documents layer4 as the Grad-CAM target — hold it to that.

        Hooking the stage keeps the residual add and output ReLU inside the
        captured activation; hooking `layer4[-1].conv3` would not.
        """
        layer = model_cpu.grad_cam_target_layer
        assert isinstance(layer, torch.nn.Module)
        assert layer is model_cpu.features[-1]
        assert isinstance(layer, torch.nn.Sequential)
        assert len(layer) == 3, "ResNet-50 layer4 has 3 bottleneck blocks"

    def test_count_parameters_reasonable(self, model_cpu):
        total = model_cpu.count_parameters(trainable_only=False)
        # ResNet-50 ~23M + custom head ~1M
        assert 20_000_000 < total < 30_000_000, f"Unexpected param count: {total:,}"


# ------------------------------------------------------------------
# Grad-CAM tests
# ------------------------------------------------------------------

class TestGradCAM:

    def test_cam_shape_matches_input(self, model_cpu, dummy_image_tensor):
        model_cpu.eval()
        gradcam = GradCAM(model_cpu, model_cpu.grad_cam_target_layer)
        cam, _ = gradcam.generate(dummy_image_tensor)
        gradcam.remove_hooks()
        assert cam.shape == (224, 224), f"Expected (224, 224), got {cam.shape}"

    def test_cam_values_in_range(self, model_cpu, dummy_image_tensor):
        model_cpu.eval()
        gradcam = GradCAM(model_cpu, model_cpu.grad_cam_target_layer)
        cam, _ = gradcam.generate(dummy_image_tensor)
        gradcam.remove_hooks()
        assert cam.min() >= 0.0 and cam.max() <= 1.0, "CAM must be normalised to [0, 1]"

    def test_cam_pred_class_valid(self, model_cpu, dummy_image_tensor):
        model_cpu.eval()
        gradcam = GradCAM(model_cpu, model_cpu.grad_cam_target_layer)
        _, pred = gradcam.generate(dummy_image_tensor)
        gradcam.remove_hooks()
        assert pred in (0, 1), f"Predicted class must be 0 or 1, got {pred}"

    def test_cam_works_with_frozen_backbone(self, model_cpu, dummy_image_tensor):
        """Regression: this used to raise AttributeError on None.

        With the backbone frozen and an input that does not require grad, the
        backward hook never fired and `self._gradients` stayed None.
        """
        model_cpu.freeze_backbone()
        model_cpu.eval()
        gradcam = GradCAM(model_cpu, model_cpu.grad_cam_target_layer)
        cam, pred = gradcam.generate(dummy_image_tensor)
        gradcam.remove_hooks()
        assert cam.shape == (224, 224)
        assert pred in (0, 1)

    def test_cam_works_inside_no_grad(self, model_cpu, dummy_image_tensor):
        """Callers routinely wrap inference in torch.no_grad()."""
        model_cpu.eval()
        with torch.no_grad():
            gradcam = GradCAM(model_cpu, model_cpu.grad_cam_target_layer)
            cam, _ = gradcam.generate(dummy_image_tensor)
            gradcam.remove_hooks()
        assert cam.shape == (224, 224)

    def test_cam_does_not_mutate_input(self, model_cpu, dummy_image_tensor):
        original = dummy_image_tensor.clone()
        model_cpu.eval()
        gradcam = GradCAM(model_cpu, model_cpu.grad_cam_target_layer)
        gradcam.generate(dummy_image_tensor)
        gradcam.remove_hooks()
        assert torch.equal(dummy_image_tensor, original)
        assert not dummy_image_tensor.requires_grad

    def test_explain_helper(self, model_cpu, dummy_image_tensor):
        model_cpu.eval()
        cam, img_np, pred = explain(model_cpu, dummy_image_tensor, torch.device("cpu"))
        assert cam.shape == (224, 224)
        assert img_np.shape == (224, 224, 3)
        assert 0.0 <= img_np.min() and img_np.max() <= 1.0
        assert pred in (0, 1)

    def test_overlay_shape_and_dtype(self, model_cpu, dummy_image_tensor):
        model_cpu.eval()
        gradcam = GradCAM(model_cpu, model_cpu.grad_cam_target_layer)
        cam, _ = gradcam.generate(dummy_image_tensor)
        gradcam.remove_hooks()

        img = dummy_image_tensor.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        overlay = GradCAM.overlay(cam, img, alpha=0.4)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8


# ------------------------------------------------------------------
# Transform / augmentation tests
# ------------------------------------------------------------------

class TestTransforms:

    def test_train_output_shape(self):
        transform = get_transforms("train", img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_output_shape(self):
        transform = get_transforms("val", img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_non_square_input_still_gives_224(self):
        """Inference must accept the wide X-rays the app actually receives."""
        transform = get_transforms("val", img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (1000, 1800, 3), dtype=np.uint8))
        assert transform(img).shape == (3, 224, 224)

    def test_transforms_return_float_tensor(self):
        transform = get_transforms("test", img_size=224)
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.dtype == torch.float32

    def test_normalisation_range(self):
        """Verify normalised tensors are roughly centred around 0."""
        transform = get_transforms("val")
        img = Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )
        tensor = transform(img)
        # After ImageNet normalisation pixel values are typically in [-2.5, 2.5]
        assert tensor.min() < 0, "Normalised tensor should have negative values"
        assert tensor.max() > 0, "Normalised tensor should have positive values"
