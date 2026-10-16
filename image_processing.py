from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


def open_image(path):
    with Image.open(path) as image:
        return image.copy()


def white_background(image):
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def grayscale(image):
    gray = white_background(image).convert("L")
    array = np.asarray(gray)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    return ImageOps.invert(gray) if np.median(border) < 128 else gray


def symbol_tensor(image):
    image = grayscale(image)
    scale = min(64 / image.width, 64 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 255)
    canvas.paste(image, ((64 - size[0]) // 2, (64 - size[1]) // 2))
    array = np.array(canvas, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array).unsqueeze(0)


def handwriting_tensor(image):
    image = image.convert("RGBA").resize((128, 128), Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    return torch.from_numpy(array.copy())


def find_chars(image, handwritten=False):
    gray = np.asarray(grayscale(image))
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = sorted(stats[1:].tolist(), key=lambda item: item[0])
    groups = []
    for x, y, w, h, area in components:
        if groups and x < groups[-1][2]:
            previous = groups[-1]
            groups[-1] = [min(x, previous[0]), min(y, previous[1]),
                          max(x + w, previous[2]), max(y + h, previous[3]),
                          area + previous[4]]
        else:
            groups.append([x, y, x + w, y + h, area])
    minimum = max(2, image.width * image.height * (0.0005 if handwritten else 0.001))
    boxes = []
    for left, top, right, bottom, area in groups:
        if area < minimum:
            continue
        left, top = max(0, left - 2), max(0, top - 2)
        right, bottom = min(image.width, right + 2), min(image.height, bottom + 2)
        boxes.append((left, top, right - left, bottom - top))
    return boxes


def save_preview(image, boxes, path):
    preview = white_background(image)
    draw = ImageDraw.Draw(preview)
    for index, (x, y, w, h) in enumerate(boxes, 1):
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline="green", width=1)
        draw.text((x, max(0, y - 12)), str(index), fill="red")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)
