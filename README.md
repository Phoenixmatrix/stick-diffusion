# stick-diffusion

A tiny image diffusion model, written and trained from scratch, that generates 32×32 stick figures.

It is a proof of concept, not a good model. There are no pretrained weights, no LoRA and no `diffusers`. The noise schedule, sampler and network are about 120 lines of plain PyTorch. The training data isn't produced by AI either: a short script draws stick figures with random poses.

| Training data (procedural) | Model samples |
| --- | --- |
| ![Procedurally drawn training figures](assets/training_data.png) | ![Stick figures sampled from the trained model](assets/generated.png) |

Most samples are recognisable figures. Some come out with fused or missing limbs.

## How it works

- **Data** (`data.py`): each figure is a random skeleton made of a head, a torso, and two arms and two legs with elbow and knee joints. Joint angles, limb lengths, lean, scale, position and line width are all randomised. Figures are drawn at 128×128 with PIL and downsampled to 32×32 grayscale. Training uses 20,000 of them, generated in a few seconds and kept on the GPU.
- **Model** (`model.py`): a small UNet with 1.11M parameters. It has three resolution levels (32 → 16 → 8), ResBlocks with GroupNorm and SiLU, sinusoidal timestep embeddings, and no attention.
- **Diffusion** (`diffusion.py`): [DDPM](https://arxiv.org/abs/2006.11239) with T = 1000 steps and a linear β schedule. The network predicts the added noise and is trained with MSE loss; sampling uses DDPM's step-by-step (ancestral) sampler.
- **Training** (`train.py`): 10k steps of AdamW (lr 3e-4, batch 256), with an exponential moving average (EMA) of the weights used for sampling. This took about 4.5 minutes on an RTX 4080.

The forward process noises a figure at t = 0, 250, 500, 750 and 999:

![Forward noising](assets/noising.png)

## Quickstart

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is strongly recommended; CPU works but training will be slow.

The repo ships trained weights (`weights/stick-diffusion.pt`, 4.5 MB), so you can sample straight away:

```sh
uv sync
uv run python ui.py        # web UI at http://localhost:8000
uv run python sample.py    # writes out/generated.png (--n, --seed)
```

To train your own model:

```sh
uv run python data.py      # optional: writes out/real_grid.png, a preview of the training data
uv run python train.py     # writes out/ckpt.pt plus sample grids every 2k steps
```

`sample.py`, `ui.py` and `memorization.py` use your own `out/ckpt.pt` if it exists, otherwise the shipped weights. Pass `--ckpt` to choose one explicitly.

`uv run python train.py --overfit --steps 2000 --batch 64` is a useful sanity check: it trains on 16 fixed images, and the samples should reproduce them. If they come out as grey mush, the schedule or sampler is broken.

## Does it copy the training data?

No sample is a pixel-level copy of a training image. `memorization.py` generates 1,000 samples and finds each one's nearest neighbour among the 20,000 training images, using per-pixel RMS distance with pixels in [-1, 1]. It compares them with two reference groups:

- **held-out:** 1,000 fresh procedural figures the model never saw. This shows what a "new" figure looks like.
- **control:** 256 samples from a model deliberately overfit on 16 training images (`train.py --overfit`). This shows what copying looks like through the same sampler.

| group | min | median | max |
| --- | --- | --- | --- |
| generated | 0.184 | 0.273 | 0.369 |
| held-out (new figures) | 0.160 | 0.275 | 0.394 |
| control (copies) | 0.014 | 0.022 | 0.290 |

Generated samples are as far from the training set as genuinely new figures are, and the closest one is about 8× further away than a typical copy. The 16 closest pairs are shown below, generated on top and their nearest training image underneath. They share a rough stance but differ in arm angles, leg spread and head shape:

![Closest generated/training pairs](assets/nearest_pairs.png)

This only rules out pixel-level copies. It doesn't measure how varied the poses are, and a copy shifted by a few pixels would pass. The control needs `out/overfit_ckpt.pt` from the overfit run; without it, that row is skipped.

## Web UI

`ui.py` is a small server built on Python's standard library. It loads the checkpoint once and streams the denoising process to the page as server-sent events, so you can watch figures form out of noise. Sampling 16 figures over the full 1000 steps takes about 2 seconds on an RTX 4080. Controls: number of figures, seed, live preview on/off, and a black-on-white "paper" view.

## Files

```
data.py       procedural stick-figure renderer and dataset
model.py      UNet and timestep embedding
diffusion.py  noise schedule, training loss, sampler
train.py      training loop (EMA, periodic samples and checkpoints)
sample.py     generate a grid from a checkpoint
ui.py         local web UI server
ui.html       web UI page
memorization.py  nearest-neighbour check against the training set
weights/      pretrained checkpoint (EMA weights after 10k steps)
```

## License

MIT, see [LICENSE](LICENSE).

This repository contains only original code. No third-party code or assets are vendored, and the web UI loads nothing from CDNs. The dependencies (PyTorch, NumPy, Pillow, plus the NVIDIA CUDA libraries PyTorch pulls in on Linux) are installed separately and come under their own licenses.
