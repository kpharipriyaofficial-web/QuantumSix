"""Train AquaMind's plankton image classifier.

The dataset must contain one directory per class, for example::

    dataset/Aquamind/
        Corethron/*.png
        Ditylum/*.png

Run from the repository root:

    py train.py --epochs 15 --batch-size 32

The best checkpoint is written to ``models/best_plankton_model.pth``.  It
contains both model weights and the class-name order required for inference.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    """A small ImageFolder-like dataset that converts every image to RGB."""

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                return self.transform(image), label
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Could not read image: {image_path}") from error


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a ResNet-18 plankton classifier.")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset" / "Aquamind")
    parser.add_argument("--output-dir", type=Path, default=root / "models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 if Windows worker processes fail.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet weights.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_classes(data_dir: Path) -> dict[str, list[Path]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir.resolve()}")

    classes: dict[str, list[Path]] = {}
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        images = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            classes[folder.name] = images
        else:
            print(f"Warning: ignoring empty class folder '{folder.name}'.")

    if len(classes) < 2:
        raise ValueError("At least two non-empty class folders are required.")
    return classes


def stratified_split(
    classes: dict[str, list[Path]], val_split: float, seed: int
) -> tuple[list[str], list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be greater than 0 and less than 1.")

    rng = random.Random(seed)
    class_names = list(classes)
    train, validation = [], []
    for label, class_name in enumerate(class_names):
        paths = classes[class_name].copy()
        if len(paths) < 2:
            raise ValueError(f"Class '{class_name}' needs at least two valid images.")
        rng.shuffle(paths)
        validation_count = min(max(1, round(len(paths) * val_split)), len(paths) - 1)
        validation.extend((path, label) for path in paths[:validation_count])
        train.extend((path, label) for path in paths[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return class_names, train, validation


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = correct = total = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("--epochs, --batch-size, and --image-size must be positive.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes = discover_classes(args.data_dir)
    class_names, train_samples, validation_samples = stratified_split(classes, args.val_split, args.seed)
    print("Classes:", ", ".join(class_names))
    print(f"Training images: {len(train_samples)} | Validation images: {len(validation_samples)}")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Sample inverse to class frequency so the large Corethron/Ditylum folders
    # do not dominate every optimization step.
    counts = torch.bincount(torch.tensor([label for _, label in train_samples]), minlength=len(class_names))
    weights = torch.tensor([1.0 / counts[label].item() for _, label in train_samples], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(ImageDataset(train_samples, train_transform), batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.workers, pin_memory=pin_memory)
    validation_loader = DataLoader(ImageDataset(validation_samples, validation_transform), batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.workers, pin_memory=pin_memory)

    pretrained_weights = None if args.no_pretrained else models.ResNet18_Weights.DEFAULT
    try:
        model = models.resnet18(weights=pretrained_weights)
    except Exception as error:
        print(f"Pretrained weights unavailable ({error}); training from scratch.")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    checkpoint_path = args.output_dir / "best_plankton_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * labels.size(0)

        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        scheduler.step(validation_accuracy)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss: {training_loss / len(train_samples):.4f} "
            f"| val loss: {validation_loss:.4f} | val accuracy: {validation_accuracy:.2%}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({
                "architecture": "resnet18",
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "normalization": {"mean": MEAN, "std": STD},
                "validation_accuracy": best_accuracy,
            }, checkpoint_path)

    with (args.output_dir / "classes.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Saved checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
