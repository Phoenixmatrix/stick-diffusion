"""Generate stick figures from a trained checkpoint, optionally with requested attributes.

Example: uv run python sample.py --arms up --legs spread --facing left --size large --position right
"""
import argparse

import torch

from data import ATTRS, IMG_SIZE, save_grid
from diffusion import DEFAULT_GUIDANCE, Diffusion
from model import default_ckpt, load_model, null_cond


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="default: out/ckpt.pt if present, else weights/stick-diffusion.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="classifier-free guidance scale")
    p.add_argument("--out", default="out/generated.png")
    for name, values in ATTRS.items():
        p.add_argument(f"--{name}", choices=["any", *values], default="any")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or default_ckpt()
    model = load_model(ckpt, dev)
    cond = null_cond(args.n)
    for i, (name, values) in enumerate(ATTRS.items()):
        v = getattr(args, name)
        if v != "any":
            cond[:, i] = values.index(v)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    x = Diffusion(dev).sample(model, args.n, IMG_SIZE, cond=cond, guidance=args.guidance, generator=g)
    save_grid(x, args.out)
    print(f"wrote {args.out} (checkpoint: {ckpt})")


if __name__ == "__main__":
    main()
