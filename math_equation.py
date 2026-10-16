import argparse
import json

import torch

from image_processing import find_chars, open_image, save_preview
from models import load_checkpoint
from recognition import recognize


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recognize a one-line arithmetic expression.")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="models/ocr_model.pth")
    parser.add_argument("--handwritten", action="store_true")
    parser.add_argument("--segment-only", action="store_true")
    parser.add_argument("--preview", help="Save an image with symbol bounding boxes.")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    try:
        image = open_image(args.image)
        if args.segment_only:
            result = {"boxes": find_chars(image, args.handwritten)}
        else:
            model, labels = load_checkpoint(args.checkpoint)
            result = recognize(image, model, labels, handwritten=args.handwritten)
        if args.preview:
            save_preview(image, result["boxes"], args.preview)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result.get("error") else 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Recognition failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
