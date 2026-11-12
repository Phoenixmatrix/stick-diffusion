"""Train the tiny UNet on procedural stick figures."""
import argparse
import copy
import time

import torch

from data import IMG_SIZE, make_dataset, save_grid
from diffusion import Diffusion
from model import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-data", type=int, default=20000)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--overfit", action="store_true", help="train on a fixed batch of 16 images (sanity check)")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    diff = Diffusion(dev)

    data = make_dataset(16 if args.overfit else args.n_data, seed=0).to(dev)
    print(f"dataset {tuple(data.shape)} on {dev}")

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

    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, data.shape[0], (args.batch,), device=dev)
        loss = diff.loss(model, data[idx])
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
            save_grid(diff.sample(ema, 64, IMG_SIZE, generator=g), f"{args.out}/{prefix}samples_{step:06d}.png")
            torch.save({"ema": ema.state_dict(), "step": step}, f"{args.out}/{prefix}ckpt.pt")
            print(f"  saved samples + ckpt at step {step}", flush=True)


if __name__ == "__main__":
    main()
