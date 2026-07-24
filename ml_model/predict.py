"""Classify a single image with the trained AquaMind model."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_model.inference import DEFAULT_CHECKPOINT, PlanktonClassifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a plankton image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    try:
        with Image.open(args.image) as image:
            result = PlanktonClassifier(args.model).predict(image)
    except (OSError, UnidentifiedImageError) as error:
        parser.error(f"Cannot open image: {error}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
