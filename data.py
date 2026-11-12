"""Procedural stick-figure generator with attribute labels. No AI involved: just joint angles + PIL lines."""
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

IMG_SIZE = 32
SUPERSAMPLE = 4
N_TRAIN = 50000  # training-set size, shared by train.py and memorization.py
TRAIN_SEED = 0

# Single source of truth for the conditioning attributes (training, sampling, UI, checkpoint).
ATTRS = {
    "arms": ["down", "out", "up", "waving"],
    "legs": ["standing", "walking", "spread", "sitting"],
    "facing": ["left", "front", "right"],
    "size": ["small", "medium", "large"],
    "position": ["left", "center", "right"],
}
ATTR_NAMES = list(ATTRS)
ATTR_SIZES = [len(v) for v in ATTRS.values()]

# Fraction of the frame: bounding-box height per size class, torso x per position class.
SIZE_RANGES = {"small": (0.48, 0.58), "medium": (0.66, 0.76), "large": (0.84, 0.94)}
POS_RANGES = {"left": (0.27, 0.37), "center": (0.46, 0.54), "right": (0.63, 0.73)}


def _pt(origin, length, angle_deg):
    """Point at `length` from `origin` in direction `angle_deg` (0 = down, positive = towards screen right)."""
    a = math.radians(angle_deg)
    return (origin[0] + length * math.sin(a), origin[1] + length * math.cos(a))


def sample_attrs(rng: np.random.Generator) -> dict:
    return {k: int(rng.integers(len(v))) for k, v in ATTRS.items()}


def _skeleton(rng, a):
    """Figure in local units (height ~1). Returns (polylines, head_center, head_radius, torso_x)."""
    U = rng.uniform
    arms, legs, facing = ATTRS["arms"][a["arms"]], ATTRS["legs"][a["legs"]], ATTRS["facing"][a["facing"]]
    d = {"left": -1, "front": 0, "right": 1}[facing]
    side = d if d else (1 if rng.random() < 0.5 else -1)  # direction asymmetric poses point

    head_r, torso = U(0.08, 0.11), U(0.26, 0.32)
    upper_arm, fore_arm = U(0.13, 0.17), U(0.12, 0.16)
    thigh, shin = U(0.15, 0.19), U(0.14, 0.18)

    # Leaning right = head to the right of the hips = torso direction pointing down-left (negative angle).
    tilt = -d * U(12, 22) if d else U(-3, 3)
    neck = (0.0, 0.0)
    head_c = _pt(neck, head_r, tilt + 180)
    hip = _pt(neck, torso, tilt)
    shoulder = _pt(neck, torso * 0.2, tilt)
    lines = [[neck, hip]]

    def limb(origin, l1, a1, l2, a2):
        mid = _pt(origin, l1, a1)
        lines.append([origin, mid, _pt(mid, l2, a2)])

    for s in (1, -1):
        if arms == "down":
            a1 = s * U(10, 35); a2 = a1 + s * U(-10, 20)
        elif arms == "out":
            a1 = s * U(75, 105); a2 = a1 + s * U(-20, 20)
        elif arms == "up":
            a1 = s * U(125, 150); a2 = a1 + s * U(-15, 15)
        elif s == side:  # waving arm: elbow out, forearm raised
            a1 = s * U(95, 125); a2 = s * U(150, 178)
        else:  # the other arm hangs down
            a1 = s * U(10, 35); a2 = a1 + s * U(-10, 20)
        limb(shoulder, upper_arm, a1, fore_arm, a2)

    if legs == "standing":
        for s in (1, -1):
            a1 = s * U(4, 14); limb(hip, thigh, a1, shin, a1 + s * U(-4, 6))
    elif legs == "spread":
        for s in (1, -1):
            a1 = s * U(32, 50); limb(hip, thigh, a1, shin, a1 + s * U(-5, 8))
    elif legs == "walking":
        a1 = side * U(20, 35); limb(hip, thigh, a1, shin, a1 - side * U(0, 20))     # front leg
        a1 = -side * U(15, 30); limb(hip, thigh, a1, shin, a1 - side * U(20, 45))   # back leg, shin kicked back
    else:  # sitting: thighs horizontal towards `side`, shins hanging down
        for _ in range(2):
            limb(hip, thigh * U(0.9, 1.05), side * U(78, 98), shin, side * U(-12, 5))

    torso_x = (neck[0] + hip[0]) / 2
    return lines, head_c, head_r, torso_x


def draw_stick_figure(rng: np.random.Generator, attrs: dict, max_tries: int = 30) -> Image.Image:
    """Draw a figure with the given attribute indices. Re-draws the continuous parameters until the
    figure fits the frame at its size/position, so labels always match pixels (after max_tries it
    accepts a slightly clipped figure)."""
    S = IMG_SIZE * SUPERSAMPLE
    size_r = SIZE_RANGES[ATTRS["size"][attrs["size"]]]
    pos_r = POS_RANGES[ATTRS["position"][attrs["position"]]]
    for _ in range(max_tries):
        lines, head_c, head_r, torso_x = _skeleton(rng, attrs)
        xs = [p[0] for l in lines for p in l] + [head_c[0] - head_r, head_c[0] + head_r]
        ys = [p[1] for l in lines for p in l] + [head_c[1] - head_r, head_c[1] + head_r]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        k = rng.uniform(*size_r) * S / (y1 - y0)
        width = max(2, int(round(S * rng.uniform(0.035, 0.05))))
        ox = rng.uniform(*pos_r) * S - torso_x * k
        oy = S / 2 + rng.uniform(-0.03, 0.03) * S - (y0 + y1) / 2 * k
        m = width / 2 + 1
        if x0 * k + ox >= m and x1 * k + ox <= S - m and y0 * k + oy >= m and y1 * k + oy <= S - m:
            break
    tr = lambda p: (p[0] * k + ox, p[1] * k + oy)
    img = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(img)
    for l in lines:
        dr.line([tr(p) for p in l], fill=255, width=width, joint="curve")
    cx, cy = tr(head_c)
    r = head_r * k
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=width)
    return img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def make_dataset(n: int, seed: int, labels: torch.Tensor | None = None):
    """Returns (x [n,1,H,W] float in [-1,1], labels [n,len(ATTRS)] long). Labels are drawn uniformly
    unless given."""
    rng = np.random.default_rng(seed)
    if labels is None:
        labels = torch.tensor([[sample_attrs(rng)[k] for k in ATTR_NAMES] for _ in range(n)], dtype=torch.long)
    imgs = [np.asarray(draw_stick_figure(rng, dict(zip(ATTR_NAMES, row.tolist()))), dtype=np.uint8) for row in labels]
    x = torch.from_numpy(np.stack(imgs)).float().div(127.5).sub(1.0).unsqueeze(1)
    return x, labels


def sweep_rows():
    """(attr_index, value_index) for every attribute value, in ATTRS order."""
    return [(i, j) for i, k in enumerate(ATTR_NAMES) for j in range(len(ATTRS[k]))]


def sweep_labels(per_row: int, base: torch.Tensor) -> torch.Tensor:
    """Labels for an attribute sweep: one row per attribute value, `per_row` images each. `base`
    ([rows*per_row, len(ATTRS)]) supplies the other attributes (random labels, or "any")."""
    labels = base.clone()
    for r, (i, j) in enumerate(sweep_rows()):
        labels[r * per_row:(r + 1) * per_row, i] = j
    return labels


def sweep_row_names():
    return [f"{ATTR_NAMES[i]}: {ATTRS[ATTR_NAMES[i]][j]}" for i, j in sweep_rows()]


def save_grid(x: torch.Tensor, path: str, nrow: int = 8, upscale: int = 3, row_labels=None) -> None:
    """x: [N,1,H,W] in [-1,1]. Saves a grid PNG (nearest-neighbour upscaled), optionally with row labels."""
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
    if row_labels:
        margin = 150
        canvas = Image.new("L", (img.width + margin, img.height), 255)
        canvas.paste(img, (margin, 0))
        dr = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=16)
        row_h = (h + pad) * upscale
        for r, text in enumerate(row_labels):
            dr.text((8, r * row_h + row_h / 2), text, fill=0, font=font, anchor="lm")
        img = canvas
    img.save(path)


if __name__ == "__main__":
    import time

    t0 = time.time()
    x, labels = make_dataset(64, seed=123)
    save_grid(x, "out/real_grid.png")
    per = 8
    rows = len(sweep_rows())
    _, base = make_dataset(rows * per, seed=7)
    sl = sweep_labels(per, base)
    xs, _ = make_dataset(len(sl), seed=8, labels=sl)
    save_grid(xs, "out/real_attrs.png", nrow=per, row_labels=sweep_row_names())
    print(f"wrote out/real_grid.png and out/real_attrs.png in {time.time() - t0:.1f}s")
