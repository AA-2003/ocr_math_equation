import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from image_processing import handwriting_tensor, open_image, png_paths, symbol_tensor
from models import ComplexNet, OCR_CNN, SYMBOLS, save_checkpoint


def symbol_records(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Symbol directory not found: {folder}")
    classes = sorted(path.name for path in folder.iterdir() if path.is_dir())
    if classes != list(SYMBOLS):
        raise ValueError(f"Expected these symbol folders: {', '.join(SYMBOLS)}")
    records = []
    for index, label in enumerate(SYMBOLS):
        paths = png_paths(folder / label)
        if len(paths) < 2:
            raise ValueError(f"Class {label} needs at least two PNG images.")
        records.extend((path, index) for path in paths)
    return records


def handwriting_records(folder, metadata):
    folder = Path(folder)
    records = []
    with open(metadata, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            path = folder / row["path"]
            if path.parent.resolve() != folder.resolve() or not path.is_file():
                raise ValueError(f"Invalid training image: {row['path']}")
            label = int(row["type"])
            if label not in (0, 1):
                raise ValueError("Handwriting labels must be 0 or 1.")
            records.append((path, label))
    return records


def split_records(records, seed=42, validation_fraction=0.2):
    groups = defaultdict(list)
    for record in records:
        groups[record[1]].append(record)
    if not groups:
        raise ValueError("No training images were found.")
    rng = random.Random(seed)
    training, validation = [], []
    for label in sorted(groups):
        group = groups[label].copy()
        if len(group) < 2:
            raise ValueError(f"Class {label} needs at least two images.")
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * validation_fraction)))
        validation.extend(group[:count])
        training.extend(group[count:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


class ImageDataset(Dataset):
    def __init__(self, records, task, augment=False, seed=42):
        self.records = records
        self.task = task
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        image = open_image(path)
        if self.augment:
            image = ImageEnhance.Contrast(image.convert("RGBA")).enhance(self.rng.uniform(0.8, 1.2))
            if self.rng.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(self.rng.uniform(0.1, 0.6)))
        transform = symbol_tensor if self.task == "symbols" else handwriting_tensor
        return transform(image), label


def run_epoch(model, loader, criterion, device, optimizer=None, binary=False):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), targets.float()) if binary else criterion(outputs, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            predictions = (outputs.squeeze(1) >= 0).long() if binary else outputs.argmax(1)
            total_loss += loss.item() * len(targets)
            correct += (predictions == targets).sum().item()
            count += len(targets)
    return {"loss": total_loss / count, "accuracy": correct / count}


def train(task, data, metadata, output, epochs=5, batch_size=16, seed=42, device="cpu", limit=None):
    if task not in ("symbols", "handwriting"):
        raise ValueError("Unknown training task.")
    if epochs < 1 or batch_size < 1 or (limit is not None and limit < 2):
        raise ValueError("Epochs and batch size must be positive; limit must be at least 2.")
    random.seed(seed)
    torch.manual_seed(seed)
    records = symbol_records(data) if task == "symbols" else handwriting_records(data, metadata)
    training, validation = split_records(records, seed)
    if limit is not None:
        training, validation = training[:limit], validation[:limit]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ImageDataset(training, task, True, seed), batch_size=batch_size,
                              shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(ImageDataset(validation, task), batch_size=batch_size, num_workers=0)
    binary = task == "handwriting"
    model = (ComplexNet() if binary else OCR_CNN()).to(device)
    criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, binary)
        val_metrics = run_epoch(model, val_loader, criterion, device, binary=binary)
        metrics = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(metrics)
        print(f"Epoch {epoch}/{epochs}: train loss={train_metrics['loss']:.4f}, "
              f"validation accuracy={val_metrics['accuracy']:.1%}", flush=True)
        if val_metrics["accuracy"] > best:
            best = val_metrics["accuracy"]
            save_checkpoint(output, model, task, ("printed", "handwritten") if binary else SYMBOLS,
                            epoch=epoch, seed=seed, validation=val_metrics,
                            train_size=len(training), validation_size=len(validation))
    return history
