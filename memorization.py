"""Check whether generated samples are copies of training images.

For every image in a group, finds the nearest training image by per-pixel RMS distance
(pixels in [-1, 1]). Groups:
  generated  samples from the trained model (all attributes "any" unless --arms etc. are given)
  held-out   fresh procedural figures the model never saw, with the same requested attributes
             (what "new" looks like)
  control    samples from an overfit checkpoint trained on 16 training images
             (what copying looks like through the same sampler; optional)
"""
import argparse
from pathlib import Path

import torch

from data import ATTR_SIZES, ATTRS, IMG_SIZE, N_TRAIN, TRAIN_SEED, make_dataset, save_grid
from diffusion import DEFAULT_GUIDANCE, Diffusion
from model import default_ckpt, load_model, null_cond


def generate(model, diff, cond, seed, guidance=DEFAULT_GUIDANCE, batch=500):
    out = []
    for i, c in enumerate(cond.split(batch)):
        g = torch.Generator(device=diff.device).manual_seed(seed + i)
        out.append(diff.sample(model, len(c), IMG_SIZE, cond=c, guidance=guidance, generator=g))
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
    p.add_argument("--out-heldout", default="out/nearest_pairs_heldout.png")
    for name, values in ATTRS.items():
        p.add_argument(f"--{name}", choices=["any", *values], default="any")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    diff = Diffusion(dev)
    train, train_labels = make_dataset(N_TRAIN, seed=TRAIN_SEED)  # exactly the set train.py uses
    train = train.to(dev)

    # Requested attributes; held-out figures get the same ones (others random) for a fair baseline.
    cond = null_cond(args.n)
    g = torch.Generator().manual_seed(0)
    held = torch.stack([torch.randint(0, s, (args.n,), generator=g) for s in ATTR_SIZES], 1)
    for i, (name, values) in enumerate(ATTRS.items()):
        v = getattr(args, name)
        if v != "any":
            cond[:, i] = held[:, i] = values.index(v)

    gen = generate(load_model(args.ckpt or default_ckpt(), dev), diff, cond, seed=1000)
    d_gen, i_gen = nearest(gen, train)
    summary("generated", d_gen)
    held_x = make_dataset(args.n, seed=1, labels=held)[0].to(dev)
    d_held, i_held = nearest(held_x, train)
    summary("held-out", d_held)
    if Path(args.overfit_ckpt).exists():
        ctrl_cond = train_labels[:16].repeat(16, 1)  # labels of the 16 figures train.py --overfit memorises
        ctrl = generate(load_model(args.overfit_ckpt, dev), diff, ctrl_cond, seed=2000, guidance=1.0)
        summary("control", nearest(ctrl, train)[0])

    # How often are samples unusually close, compared with genuinely new figures?
    m = d_held.min()
    print(f"generated closer than the closest held-out figure ({m:.3f}): {(d_gen < m).sum().item()} / {len(d_gen)}")
    for thr in (0.10, 0.12, 0.14):
        print(f"  within {thr:.2f} of a training image: generated {(d_gen < thr).float().mean() * 100:.1f}%"
              f"   held-out {(d_held < thr).float().mean() * 100:.1f}%")

    # The 16 closest pairs (top rows: generated / held-out figure, bottom rows: its nearest training image)
    for x, d, i, path in [(gen, d_gen, i_gen, args.out), (held_x, d_held, i_held, args.out_heldout)]:
        order = d.argsort()[:16]
        rows = [x[order[:8]], train[i[order[:8]]], x[order[8:]], train[i[order[8:]]]]
        save_grid(torch.cat(rows), path, nrow=8, upscale=4)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
