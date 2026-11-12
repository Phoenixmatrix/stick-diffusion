"""Check whether generated samples are copies of training images.

For every image in a group, finds the nearest of the 20k training images by per-pixel RMS
distance (pixels in [-1, 1]). Groups:
  generated  samples from the trained model
  held-out   fresh procedural figures the model never saw (what "new" looks like)
  control    samples from an overfit checkpoint trained on 16 training images
             (what copying looks like through the same sampler; optional)
"""
import argparse
from pathlib import Path

import torch

from data import IMG_SIZE, make_dataset, save_grid
from diffusion import Diffusion
from model import default_ckpt, load_model


def generate(model, diff, n, seed, batch=500):
    out = []
    for i, start in enumerate(range(0, n, batch)):
        g = torch.Generator(device=diff.device).manual_seed(seed + i)
        out.append(diff.sample(model, min(batch, n - start), IMG_SIZE, generator=g))
    return torch.cat(out)


def nearest(x, train, chunk=256):
    """Returns (rms distance, index) of the nearest training image for each image in x."""
    tf = train.flatten(1)
    dists, idxs = [], []
    for xc in x.flatten(1).split(chunk):
        d = torch.cdist(xc, tf)
        m, i = d.min(1)
        dists.append(m / (tf.shape[1] ** 0.5))
        idxs.append(i)
    return torch.cat(dists), torch.cat(idxs)


def summary(name, d):
    q = torch.quantile(d, torch.tensor([0.0, 0.01, 0.05, 0.5, 1.0], device=d.device)).tolist()
    print(f"{name:10s} n={len(d):5d}  min {q[0]:.3f}  p1 {q[1]:.3f}  p5 {q[2]:.3f}  median {q[3]:.3f}  max {q[4]:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--overfit-ckpt", default="out/overfit_ckpt.pt")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", default="out/nearest_pairs.png")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    diff = Diffusion(dev)
    train = make_dataset(20000, seed=0).to(dev)  # exactly the set train.py uses

    gen = generate(load_model(args.ckpt or default_ckpt(), dev), diff, args.n, seed=1000)
    d_gen, i_gen = nearest(gen, train)
    summary("generated", d_gen)
    summary("held-out", nearest(make_dataset(args.n, seed=1).to(dev), train)[0])
    if Path(args.overfit_ckpt).exists():
        ctrl = generate(load_model(args.overfit_ckpt, dev), diff, 256, seed=2000)
        summary("control", nearest(ctrl, train)[0])

    # The 16 generated samples closest to the training set (top rows) and their nearest training image (bottom rows)
    order = d_gen.argsort()[:16]
    rows = [gen[order[:8]], train[i_gen[order[:8]]], gen[order[8:]], train[i_gen[order[8:]]]]
    save_grid(torch.cat(rows), args.out, nrow=8, upscale=4)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
