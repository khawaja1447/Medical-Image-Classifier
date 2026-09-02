import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """ResNet-50 with a custom two-layer classification head.

    Training strategy:
        Phase 1 — freeze backbone, train head only (fast convergence).
        Phase 2 — unfreeze all layers, fine-tune with differential LRs.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Strip the original FC; keep everything up to (and including) layer4
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = backbone.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )
        self._init_classifier()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_classifier(self):
        linears = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]

        # Hidden layers are followed by a ReLU, so Kaiming's relu gain applies.
        for m in linears[:-1]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

        # The output layer feeds a softmax, not a ReLU. Kaiming's sqrt(2) relu
        # gain inflates the initial logits and starts training over-confident;
        # Xavier is the right choice for a linear output.
        nn.init.xavier_normal_(linears[-1].weight)
        nn.init.constant_(linears[-1].bias, 0)

    # ------------------------------------------------------------------
    # Backbone freezing helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def unfreeze_last_n_layers(self, n: int = 2):
        """Unfreeze only the last *n* ResNet stages (e.g. layer3+layer4)."""
        layers = list(self.features.children())
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        """``layer4`` — the conventional Grad-CAM target for ResNet.

        Hooking the stage as a whole, rather than the final 1x1 conv inside its
        last bottleneck, keeps the residual addition and the output ReLU inside
        the captured activation. That is what Grad-CAM assumes, and it is what
        the architecture diagram in the README describes.
        """
        return self.features[-1]
