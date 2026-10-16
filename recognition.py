import ast
import math
import operator

import torch

from image_processing import find_chars, symbol_tensor

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
          ast.Mult: operator.mul, ast.Div: operator.truediv}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_expression(expression):
    if not expression or len(expression) > 200:
        raise ValueError("Expression must contain between 1 and 200 characters.")
    try:
        tree = ast.parse(expression, mode="eval")

        def evaluate(node, depth=0):
            if depth > 30:
                raise ValueError("Expression is too deeply nested.")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
                value = BINARY[type(node.op)](evaluate(node.left, depth + 1), evaluate(node.right, depth + 1))
            elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
                value = UNARY[type(node.op)](evaluate(node.operand, depth + 1))
            else:
                raise ValueError("Only numbers, parentheses and + - * / are supported.")
            if not math.isfinite(value) or abs(value) > 1e15:
                raise ValueError("Expression result is outside the supported range.")
            return value

        return round(evaluate(tree.body), 2)
    except (SyntaxError, ZeroDivisionError, OverflowError, RecursionError) as error:
        raise ValueError("Expression is invalid or divides by zero.") from error


def recognize(image, model, labels, device="cpu", handwritten=False):
    boxes = find_chars(image, handwritten)
    if not boxes:
        raise ValueError("No symbols were found in the image.")
    crops = [symbol_tensor(image.crop((x, y, x + w, y + h))) for x, y, w, h in boxes]
    model.eval()
    with torch.inference_mode():
        indices = model(torch.stack(crops).to(device)).argmax(1).tolist()
    expression = "".join({"divide": "/", "x": "*"}.get(labels[i], labels[i]) for i in indices)
    try:
        answer, error = evaluate_expression(expression), None
    except ValueError as exception:
        answer, error = None, str(exception)
    return {"expression": expression, "answer": answer, "error": error, "boxes": boxes}
