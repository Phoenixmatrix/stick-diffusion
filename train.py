"""Train the tiny attribute-conditioned UNet on procedural stick figures."""
import argparse
import copy
import time

import torch

from data import ATTRS, IMG_SIZE, N_TRAIN, TRAIN_SEED, make_dataset, save_grid, sweep_labels, sweep_row_names, sweep_rows
from diffusion import Diffusion
from model import UNet, null_cond


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--overfit", action="store_true", help="train on a fixed batch of 16 images (sanity check)")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    diff = Diffusion(dev)

    t0 = time.time()
    data, labels = make_dataset(N_TRAIN, seed=TRAIN_SEED)
    if args.overfit:  # the first 16 figures of the real training set (memorization.py's copy control)
        data, labels = data[:16], labels[:16]
    data, labels = data.to(dev), labels.to(dev)
    print(f"dataset {tuple(data.shape)} on {dev} ({time.time() - t0:.1f}s)")

    # Forward-noising sanity image: one figure at t = 0, 250, 500, 750, 999
    ts = torch.tensor([0, 250, 500, 750, 999], device=dev)
    x0 = data[:1].repeat(len(ts), 1, 1, 1)
    save_grid(diff.q_sample(x0, ts, torch.randn_like(x0)), f"{args.out}/noising.png", nrow=len(ts))

    model = UNet().to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema_decay = 0.995 if args.overfit else 0.999
    prefix = "overfit_" if args.overfit else ""

    # Progress grids: the 16 overfit images (with their labels), or an attribute sweep
    # (one row per attribute value, other attributes "any").
    per = 8
    if args.overfit:
        grid_cond, grid_kw = labels.repeat(4, 1), {}
    else:
        grid_cond = sweep_labels(per, null_cond(len(sweep_rows()) * per)).to(dev)
        grid_kw = {"nrow": per, "row_labels": sweep_row_names()}

    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, data.shape[0], (args.batch,), device=dev)
        loss = diff.loss(model, data[idx], labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - ema_decay)

        running += loss.item()
        if step % 100 == 0:
            print(f"step {step:6d}  loss {running / 100:.4f}  {time.time() - t0:.0f}s", flush=True)
            running = 0.0
        if step % args.sample_every == 0 or step == args.steps:
            g = torch.Generator(device=dev).manual_seed(0)
            x = diff.sample(ema, len(grid_cond), IMG_SIZE, cond=grid_cond, generator=g)
            save_grid(x, f"{args.out}/{prefix}samples_{step:06d}.png", **grid_kw)
            torch.save({"ema": ema.state_dict(), "step": step, "attrs": ATTRS}, f"{args.out}/{prefix}ckpt.pt")
            print(f"  saved samples + ckpt at step {step}", flush=True)


if __name__ == "__main__":
    main()
