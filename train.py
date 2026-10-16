import argparse

import torch

from training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and save an OCR or handwriting classifier.")
    parser.add_argument("task", choices=("symbols", "handwriting"))
    parser.add_argument("--data", help="Symbol class folders or equation image folder.")
    parser.add_argument("--metadata", default="train_info.csv")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Limit each split for a smoke test; not an accuracy benchmark.")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu.")
    data = args.data or ("train_ocr" if args.task == "symbols" else "train")
    output = args.output or ("models/ocr_model.pth" if args.task == "symbols" else "models/handwriting_model.pth")
    try:
        train(args.task, data, args.metadata, output, args.epochs, args.batch_size,
              args.seed, args.device, args.limit)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(f"Saved best validation checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
