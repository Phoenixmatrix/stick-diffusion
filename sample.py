"""Generate stick figures from a trained checkpoint."""
import argparse

import torch

from data import IMG_SIZE, save_grid
from diffusion import Diffusion
from model import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="out/generated.png")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["ema"])
    g = torch.Generator(device=dev).manual_seed(args.seed)
    save_grid(Diffusion(dev).sample(model, args.n, IMG_SIZE, generator=g), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
