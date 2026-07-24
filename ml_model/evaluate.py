"""Evaluate the trained model on class-named image folders."""

import argparse
from pathlib import Path

from PIL import Image

from ml_model.inference import PlanktonClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure accuracy on labelled images.")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    classifier = PlanktonClassifier()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    total = correct = 0
    for class_dir in sorted(path for path in args.data_dir.iterdir() if path.is_dir()):
        for image_path in class_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                continue
            try:
                with Image.open(image_path) as image:
                    correct += classifier.predict(image)["predicted_class"] == class_dir.name
                total += 1
            except OSError:
                print(f"Skipping unreadable image: {image_path}")
    if not total:
        raise ValueError("No readable images found.")
    print(f"Accuracy: {correct / total:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
