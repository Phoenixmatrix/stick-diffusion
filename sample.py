"""Generate stick figures from a trained checkpoint."""
import argparse

import torch

from data import IMG_SIZE, save_grid
from diffusion import Diffusion
from model import default_ckpt, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="default: out/ckpt.pt if present, else weights/stick-diffusion.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="out/generated.png")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or default_ckpt()
    model = load_model(ckpt, dev)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    save_grid(Diffusion(dev).sample(model, args.n, IMG_SIZE, generator=g), args.out)
    print(f"wrote {args.out} (checkpoint: {ckpt})")


if __name__ == "__main__":
    main()
