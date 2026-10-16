from pathlib import Path

import torch
from torch import nn

SYMBOLS = ("(", ")", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "divide", "x")


class OCR_CNN(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc2(self.dropout(self.relu(self.fc1(x.flatten(1)))))


class convBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.r1 = nn.ReLU()
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.r2 = nn.ReLU()
        self.p = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        return self.dropout(self.p(self.r2(self.bn2(self.c2(self.r1(self.bn1(self.c1(x))))))))


class ComplexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cb1 = convBlock(4, 64)
        self.cb2 = convBlock(64, 128)
        self.cb3 = convBlock(128, 256)
        self.cb4 = convBlock(256, 256)
        self.cb5 = convBlock(256, 512)
        self.f = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.o = nn.Linear(64, 1)

    def forward(self, x):
        for block in (self.cb1, self.cb2, self.cb3, self.cb4, self.cb5):
            x = block(x)
        x = torch.relu(self.dropout(self.fc1(self.f(x))))
        return self.o(torch.relu(self.fc2(x)))


def save_checkpoint(path, model, task, labels, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "task": task, "labels": list(labels), "state_dict": model.state_dict(),
        "torch_version": str(torch.__version__), **metadata,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path, task="symbols", device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run: python train.py {task}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != task:
        raise ValueError(f"Expected a {task} checkpoint.")
    labels = checkpoint["labels"]
    if task == "symbols" and (len(labels) != len(SYMBOLS) or set(labels) != set(SYMBOLS)):
        raise ValueError("Checkpoint contains an unsupported symbol mapping.")
    model = OCR_CNN(len(labels)) if task == "symbols" else ComplexNet()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), labels
