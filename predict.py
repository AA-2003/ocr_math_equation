import argparse
import csv
import os
import tempfile
from pathlib import Path

import torch

from image_processing import handwriting_tensor, open_image
from models import load_checkpoint
from recognition import recognize


def predict_folder(folder, output, ocr_checkpoint, handwriting_checkpoint=None, handwritten=False):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    images = sorted(folder.glob("*.png"), key=lambda path: (
        0, int(path.stem)) if path.stem.isdigit() else (1, path.name))
    if not images:
        raise ValueError(f"No PNG images found in {folder}")
    model, labels = load_checkpoint(ocr_checkpoint)
    handwriting = load_checkpoint(handwriting_checkpoint, "handwriting")[0] if handwriting_checkpoint else None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            fields = ("path", "type", "expression", "answer", "error")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for path in images:
                image = open_image(path)
                flag = int(handwritten)
                if handwriting is not None:
                    with torch.inference_mode():
                        flag = int(handwriting(handwriting_tensor(image).unsqueeze(0)).item() >= 0)
                try:
                    result = recognize(image, model, labels, handwritten=bool(flag))
                    row = {key: result[key] for key in ("expression", "answer", "error")}
                except ValueError as error:
                    row = {"expression": "", "answer": None, "error": str(error)}
                writer.writerow({"path": path.name, "type": flag, **row})
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(images)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict equations in a PNG folder and write a new CSV.")
    parser.add_argument("folder")
    parser.add_argument("--output", default="reports/predictions.csv")
    parser.add_argument("--ocr-checkpoint", default="models/ocr_model.pth")
    parser.add_argument("--handwriting-checkpoint")
    parser.add_argument("--handwritten", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    try:
        count = predict_folder(args.folder, args.output, args.ocr_checkpoint,
                               args.handwriting_checkpoint, args.handwritten)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(f"Saved {count} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
