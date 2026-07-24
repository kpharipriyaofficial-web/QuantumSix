"""Train an image classifier from class-named folders in ./dataset.

Example:
    py train.py --epochs 12 --batch-size 16
"""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class PlanktonDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plankton image classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0, help="Keep 0 on Windows if loading fails.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet weights.")
    return parser.parse_args()


def collect_samples(data_dir):
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir.resolve()}")

    class_files = {}
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        files = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if files:
            class_files[folder.name] = sorted(files)
        else:
            print(f"Warning: skipping empty class folder: {folder.name}")

    if len(class_files) < 2:
        raise ValueError("At least two non-empty class folders are required to train a classifier.")
    return class_files


def split_samples(class_files, val_split, seed):
    if not 0 < val_split < 1:
        raise ValueError("--val-split must be between 0 and 1.")
    rng = random.Random(seed)
    class_names = list(class_files)
    train_samples, val_samples = [], []

    for label, class_name in enumerate(class_names):
        files = class_files[class_name].copy()
        rng.shuffle(files)
        if len(files) < 2:
            raise ValueError(f"Class '{class_name}' needs at least two images.")
        val_count = max(1, round(len(files) * val_split))
        val_count = min(val_count, len(files) - 1)
        val_samples.extend((path, label) for path in files[:val_count])
        train_samples.extend((path, label) for path in files[val_count:])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return class_names, train_samples, val_samples


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_files = collect_samples(args.data_dir)
    class_names, train_samples, val_samples = split_samples(class_files, args.val_split, args.seed)
    print("Classes:", ", ".join(class_names))
    print(f"Training images: {len(train_samples)} | Validation images: {len(val_samples)}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(PlanktonDataset(train_samples, train_transform), args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(PlanktonDataset(val_samples, validation_transform), args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    weights = None if args.no_pretrained else models.ResNet18_Weights.DEFAULT
    try:
        model = models.resnet18(weights=weights)
    except Exception as error:
        print(f"Could not load pretrained weights ({error}). Training from scratch instead.")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        val_accuracy = accuracy(model, val_loader, device)
        train_loss = running_loss / len(train_samples)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss: {train_loss:.4f} | val accuracy: {val_accuracy:.2%}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": 224,
                "architecture": "resnet18",
                "validation_accuracy": best_accuracy,
            }
            torch.save(checkpoint, args.output_dir / "best_plankton_model.pth")

    with open(args.output_dir / "classes.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    print(f"Done. Best model saved to: {(args.output_dir / 'best_plankton_model.pth').resolve()}")
    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
