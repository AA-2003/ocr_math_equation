import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from image_processing import find_chars, grayscale, handwriting_tensor, symbol_tensor
from models import ComplexNet, OCR_CNN, SYMBOLS, load_checkpoint, save_checkpoint
from predict import predict_folder
from recognition import evaluate_expression, recognize
from training import ImageDataset, run_epoch, split_records, symbol_records

torch.set_num_threads(2)


def equation_image():
    image = Image.new("RGBA", (100, 50), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 8, 8, 40), fill="black")
    draw.rectangle((40, 22, 55, 25), fill="black")
    draw.rectangle((46, 10, 49, 13), fill="black")
    draw.rectangle((46, 34, 49, 37), fill="black")
    draw.rectangle((80, 8, 90, 40), fill="black")
    return image


class ImageTests(unittest.TestCase):
    def test_background_is_excluded_and_division_parts_are_grouped(self):
        boxes = find_chars(equation_image())
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[0][0], 0)
        self.assertLess(boxes[1][1], 11)
        for x, y, w, h in boxes:
            self.assertTrue(0 <= x < x + w <= 100)
            self.assertTrue(0 <= y < y + h <= 50)

    def test_blank_image_has_no_symbols(self):
        self.assertEqual(find_chars(Image.new("L", (80, 50), 255)), [])

    def test_alpha_is_composited_on_white(self):
        image = Image.new("RGBA", (30, 30), (0, 0, 0, 0))
        self.assertEqual(np.asarray(grayscale(image)).min(), 255)

    def test_preprocessing_matches_both_polarities(self):
        image = equation_image().convert("L")
        inverted = Image.fromarray(255 - np.asarray(image))
        self.assertTrue(torch.equal(symbol_tensor(image), symbol_tensor(inverted)))
        tensor = symbol_tensor(image)
        self.assertEqual(tuple(tensor.shape), (1, 64, 64))
        self.assertGreaterEqual(tensor.min().item(), -1)
        self.assertLessEqual(tensor.max().item(), 1)
        self.assertEqual(tuple(handwriting_tensor(image).shape), (4, 128, 128))


class ExpressionTests(unittest.TestCase):
    def test_arithmetic_and_parentheses(self):
        self.assertEqual(evaluate_expression("-(12+3)*2/4"), -7.5)
        self.assertEqual(evaluate_expression("1/3"), 0.33)

    def test_invalid_or_unsupported_expressions(self):
        for text in ("", "1/0", "1+", "2**3", "__import__('os')", "True", "1e100", "a+1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evaluate_expression(text)

    def test_invalid_prediction_is_not_silently_zero(self):
        class FixedModel(nn.Module):
            def forward(self, images):
                scores = torch.zeros(len(images), 16)
                scores[:, SYMBOLS.index("(")] = 10
                return scores
        result = recognize(equation_image(), FixedModel(), SYMBOLS)
        self.assertIsNone(result["answer"])
        self.assertIsNotNone(result["error"])


class TrainingTests(unittest.TestCase):
    def test_split_is_seeded_stratified_and_disjoint(self):
        records = [(Path(f"{label}/{i}.png"), label) for label in range(16) for i in range(8)]
        training, validation = split_records(records)
        self.assertEqual((training, validation), split_records(records))
        self.assertFalse(set(training) & set(validation))
        self.assertEqual({label for _, label in training}, set(range(16)))
        self.assertEqual({label for _, label in validation}, set(range(16)))

    def test_validation_has_no_augmentation_and_no_file_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label in ("0", "1"):
                (root / label).mkdir()
                equation_image().save(root / label / "same.png")
            records = [(root / label / "same.png", int(label)) for label in ("0", "1")]
            dataset = ImageDataset(records, "symbols", augment=False)
            self.assertTrue(torch.equal(dataset[0][0], dataset[0][0]))
            ImageDataset(records, "symbols", augment=True)[0]
            self.assertEqual(len(list(root.rglob("*.png"))), 2)

    def test_missing_or_incomplete_classes_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                symbol_records(folder)

    def test_ocr_training_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            image_path = path / "symbol.png"
            equation_image().save(image_path)
            loader = DataLoader(ImageDataset([(image_path, 4), (image_path, 5)], "symbols"), batch_size=2)
            model = OCR_CNN()
            before = model.fc2.weight.detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu", optimizer)
            self.assertFalse(torch.equal(before, model.fc2.weight))
            run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu")
            self.assertFalse(model.training)
            checkpoint = path / "ocr.pth"
            save_checkpoint(checkpoint, model, "symbols", SYMBOLS)
            restored, labels = load_checkpoint(checkpoint)
            sample = symbol_tensor(equation_image()).unsqueeze(0)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(sample), restored(sample)))
            self.assertEqual(labels, list(SYMBOLS))
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, "handwriting")

    def test_handwriting_training_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "image.png"
            equation_image().save(image_path)
            loader = DataLoader(ImageDataset([(image_path, 0), (image_path, 1)], "handwriting"), batch_size=2)
            model = ComplexNet()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            metrics = run_epoch(model, loader, nn.BCEWithLogitsLoss(), "cpu", optimizer, binary=True)
            self.assertTrue(np.isfinite(metrics["loss"]))
            run_epoch(model, loader, nn.BCEWithLogitsLoss(), "cpu", binary=True)
            self.assertFalse(model.training)
            checkpoint = Path(folder) / "handwriting.pth"
            save_checkpoint(checkpoint, model, "handwriting", ["printed", "handwritten"])
            restored, _ = load_checkpoint(checkpoint, "handwriting")
            sample = handwriting_tensor(equation_image()).unsqueeze(0)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(sample), restored(sample)))

    def test_missing_checkpoint_explains_training_command(self):
        with self.assertRaisesRegex(FileNotFoundError, "python train.py symbols"):
            load_checkpoint("does-not-exist.pth")


class PredictionTests(unittest.TestCase):
    def test_csv_keeps_filenames_and_existing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("10.png", "2.png"):
                equation_image().save(root / name)
            output = root / "predictions.csv"
            checkpoint = root / "ocr.pth"
            save_checkpoint(checkpoint, OCR_CNN(), "symbols", SYMBOLS)
            result = {"expression": "1+1", "answer": 2, "error": None, "boxes": []}
            with patch("predict.recognize", return_value=result):
                self.assertEqual(predict_folder(root, output, checkpoint), 2)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["path"] for row in rows], ["2.png", "10.png"])
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                predict_folder(root, output, checkpoint)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
