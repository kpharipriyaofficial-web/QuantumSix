"""Model architecture used by AquaMind training and inference."""

from torch import nn
from torchvision import models


def create_model(num_classes: int) -> nn.Module:
    """Create the ResNet-18 classifier matching the saved checkpoint."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
