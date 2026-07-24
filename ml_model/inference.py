"""Shared inference utilities for AquaMind."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_plankton_model.pth"


class PlanktonClassifier:
    """Load the trained ResNet-18 checkpoint and classify PIL images."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path.resolve()}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:  # Supports older PyTorch releases.
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        required = {"model_state_dict", "class_names", "image_size"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint is missing metadata: {', '.join(sorted(missing))}")
        self.class_names = checkpoint["class_names"]
        normalization = checkpoint.get("normalization", {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)})
        size = int(checkpoint["image_size"])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(normalization["mean"], normalization["std"]),
        ])
        self.model: nn.Module = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, object]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu()
        confidence, index = torch.max(probabilities, dim=0)
        return {
            "predicted_class": self.class_names[index.item()],
            "confidence": round(float(confidence), 6),
            "probabilities": {name: round(float(score), 6) for name, score in zip(self.class_names, probabilities)},
        }
