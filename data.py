"""Procedural stick-figure generator. No AI involved: just random joint angles + PIL lines."""
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

IMG_SIZE = 32
SUPERSAMPLE = 4


def _pt(origin, length, angle_deg):
    """Point at `length` from `origin` in direction `angle_deg` (0 = down, positive = to the figure's left/screen right)."""
    a = math.radians(angle_deg)
    return (origin[0] + length * math.sin(a), origin[1] + length * math.cos(a))


def draw_stick_figure(rng: np.random.Generator) -> Image.Image:
    S = IMG_SIZE * SUPERSAMPLE
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)

    scale = rng.uniform(0.75, 1.0) * S  # overall figure height ~ scale
    head_r = scale * rng.uniform(0.08, 0.11)
    torso = scale * rng.uniform(0.26, 0.32)
    upper_arm = scale * rng.uniform(0.13, 0.17)
    fore_arm = scale * rng.uniform(0.12, 0.16)
    thigh = scale * rng.uniform(0.15, 0.19)
    shin = scale * rng.uniform(0.14, 0.18)
    width = max(2, int(round(S * rng.uniform(0.035, 0.05))))

    lean = rng.uniform(-12, 12)  # torso tilt
    total_h = 2 * head_r + torso + thigh + shin
    cx = S / 2 + rng.uniform(-0.08, 0.08) * S
    top = (S - total_h) / 2 + rng.uniform(-0.04, 0.04) * S

    head_c = (cx, top + head_r)
    neck = _pt(head_c, head_r, lean)  # bottom of head, in the lean direction
    hip = _pt(neck, torso, lean)
    shoulder = _pt(neck, torso * 0.2, lean)

    # Arms: angle from straight-down; one arm goes left (+), the other right (-)
    for side in (1, -1):
        a1 = side * rng.uniform(10, 170)
        elbow = _pt(shoulder, upper_arm, a1)
        a2 = a1 + side * rng.uniform(-60, 90)
        hand = _pt(elbow, fore_arm, a2)
        d.line([shoulder, elbow, hand], fill=255, width=width, joint="curve")

    # Legs
    for side in (1, -1):
        a1 = side * rng.uniform(5, 50)
        knee = _pt(hip, thigh, a1)
        a2 = a1 + rng.uniform(-35, 35)
        foot = _pt(knee, shin, a2)
        d.line([hip, knee, foot], fill=255, width=width, joint="curve")

    d.line([neck, hip], fill=255, width=width)
    d.ellipse(
        [head_c[0] - head_r, head_c[1] - head_r, head_c[0] + head_r, head_c[1] + head_r],
        outline=255,
        width=width,
    )
    return img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def make_dataset(n: int = 20000, seed: int = 0) -> torch.Tensor:
    """Returns [n, 1, IMG_SIZE, IMG_SIZE] float tensor in [-1, 1]."""
    rng = np.random.default_rng(seed)
    arr = np.stack([np.asarray(draw_stick_figure(rng), dtype=np.uint8) for _ in range(n)])
    x = torch.from_numpy(arr).float().div(127.5).sub(1.0)
    return x.unsqueeze(1)


def save_grid(x: torch.Tensor, path: str, nrow: int = 8, upscale: int = 3) -> None:
    """x: [N,1,H,W] in [-1,1]. Saves a grid PNG (nearest-neighbour upscaled for viewing)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x = ((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    n, _, h, w = x.shape
    ncol = nrow
    nrows = math.ceil(n / ncol)
    pad = 1
    grid = np.full((nrows * (h + pad) + pad, ncol * (w + pad) + pad), 64, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, ncol)
        grid[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = x[i, 0].numpy()
    img = Image.fromarray(grid)
    img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    img.save(path)


if __name__ == "__main__":
    x = make_dataset(64, seed=123)
    save_grid(x, "out/real_grid.png")
    print("wrote out/real_grid.png", tuple(x.shape), float(x.min()), float(x.max()))
