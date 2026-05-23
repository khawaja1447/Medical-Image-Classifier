"""Gradient-weighted Class Activation Mapping (Grad-CAM).

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization", ICCV 2017.
"""

from typing import Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class GradCAM:
    """Attach forward/backward hooks to a target layer and compute Grad-CAM maps."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _fwd_hook(self, _module, _input, output):
        self._activations = output.detach()

    def _bwd_hook(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    # ------------------------------------------------------------------
    # Grad-CAM computation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """Return a (H, W) float32 CAM array in [0, 1] and the predicted class index."""
        self.model.eval()

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1))

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # α_k = global-average-pool of gradients for channel k
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32), class_idx

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def overlay(
        cam: np.ndarray,
        image_rgb: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Blend a Grad-CAM heatmap with the original image.

        Args:
            cam:       Normalised (H, W) array in [0, 1].
            image_rgb: (H, W, 3) uint8 or float32 in [0, 1].
            alpha:     Heatmap opacity.

        Returns:
            (H, W, 3) uint8 blended image.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if image_rgb.max() > 1.0:
            image_rgb = image_rgb.astype(np.float32) / 255.0

        blended = np.clip(heatmap * alpha + image_rgb * (1 - alpha), 0, 1)
        return (blended * 255).astype(np.uint8)

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def explain(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run Grad-CAM and return (cam, denormed_image_float, pred_class_idx).

    Args:
        model:        ResNet50Classifier (must have .grad_cam_target_layer).
        image_tensor: (C, H, W) normalised tensor.
        device:       Target device.

    Returns:
        cam:        (H, W) float32 in [0, 1].
        image_np:   (H, W, 3) float32 in [0, 1] — denormalised for display.
        pred_class: Predicted class index.
    """
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    gradcam = GradCAM(model, model.grad_cam_target_layer)
    tensor = image_tensor.to(device)
    cam, pred_class = gradcam.generate(tensor)
    gradcam.remove_hooks()

    img_np = image_tensor.cpu() * _STD + _MEAN
    img_np = img_np.permute(1, 2, 0).numpy().clip(0, 1)
    return cam, img_np, pred_class
