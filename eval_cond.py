"""Does the model draw what was asked for?

Trains a small CNN on fresh procedural figures to read the five attributes back from an image, then:
  ceiling   its accuracy on held-out real figures (the best a perfect generator could score)
  all set   samples with all five attributes requested at random, per guidance scale
  one set   one attribute requested, the rest "any" (the typical UI path)
  control   all-"any" samples scored against random labels (should be at chance)
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ATTR_NAMES, ATTR_SIZES, ATTRS, IMG_SIZE, make_dataset

CLF_PATH = Path("out/attr_classifier.pt")


class AttrClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), nn.Flatten(),
                               nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, sum(ATTR_SIZES)))

    def forward(self, x):
        return self.f(x).split(ATTR_SIZES, dim=1)


def train_classifier(dev, steps=4000, n=60000):
    x, y = make_dataset(n, seed=101)
    x, y = x.to(dev), y.to(dev)
    clf = AttrClassifier().to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for step in range(steps):
        idx = torch.randint(0, n, (256,), device=dev)
        loss = sum(F.cross_entropy(o, y[idx, i]) for i, o in enumerate(clf(x[idx])))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    CLF_PATH.parent.mkdir(exist_ok=True)
    torch.save(clf.state_dict(), CLF_PATH)
    return clf.eval()


@torch.no_grad()
def predict(clf, x):
    """[N,1,H,W] -> predicted attribute indices [N, len(ATTRS)]."""
    return torch.cat([torch.stack([o.argmax(1) for o in clf(b)], 1) for b in x.split(500)])


def accuracy(pred, labels, mask=None):
    """Per-attribute accuracy; mask [N,A] selects which entries count."""
    ok = (pred == labels).float()
    if mask is None:
        return ok.mean(0).tolist()
    return [(ok[:, i][mask[:, i]].mean().item() if mask[:, i].any() else float("nan")) for i in range(ok.shape[1])]


def row(name, accs):
    print(f"{name:28s}" + "".join(f"{a * 100:9.1f}" for a in accs))


def header():
    print(f"{'':28s}" + "".join(f"{k:>9s}" for k in ATTR_NAMES))
    print(f"{'chance':28s}" + "".join(f"{100 / s:9.1f}" for s in ATTR_SIZES))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ceiling-only", action="store_true")
    p.add_argument("--retrain", action="store_true", help="retrain the classifier even if cached")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n", type=int, default=500, help="samples per configuration")
    p.add_argument("--guidance", type=float, nargs="+", default=[1, 2, 3, 5])
    args = p.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if CLF_PATH.exists() and not args.retrain:
        clf = AttrClassifier().to(dev)
        clf.load_state_dict(torch.load(CLF_PATH, map_location=dev, weights_only=True))
        clf.eval()
    else:
        clf = train_classifier(dev)

    header()
    xr, yr = make_dataset(5000, seed=102)
    pr = predict(clf, xr.to(dev)).cpu()
    row("ceiling (real figures)", accuracy(pr, yr))
    for si, s in enumerate(ATTRS["size"]):
        m = (yr[:, ATTR_NAMES.index("size")] == si)
        row(f"  ceiling, size={s}", accuracy(pr[m], yr[m]))
    if args.ceiling_only:
        return

    from diffusion import Diffusion
    from model import default_ckpt, load_model, null_cond

    model = load_model(args.ckpt or default_ckpt(), dev)
    diff = Diffusion(dev)
    g = torch.Generator().manual_seed(1)
    rand_labels = lambda n: torch.stack([torch.randint(0, s, (n,), generator=g) for s in ATTR_SIZES], 1)

    def gen(cond, w, seed):
        gg = torch.Generator(device=dev).manual_seed(seed)
        return torch.cat([diff.sample(model, len(c), IMG_SIZE, cond=c.to(dev), guidance=w, generator=gg)
                          for c in cond.split(500)])

    want = rand_labels(args.n)
    for w in args.guidance:
        pg = predict(clf, gen(want, w, seed=10)).cpu()
        row(f"all set, guidance {w:g}", accuracy(pg, want))

    # One attribute requested, the rest "any": n samples per attribute.
    w = args.guidance[min(1, len(args.guidance) - 1)]
    cond = null_cond(args.n * len(ATTR_NAMES))
    mask = torch.zeros_like(cond, dtype=torch.bool)
    one = rand_labels(args.n * len(ATTR_NAMES))
    for i in range(len(ATTR_NAMES)):
        sl = slice(i * args.n, (i + 1) * args.n)
        cond[sl, i] = one[sl, i]
        mask[sl, i] = True
    pg = predict(clf, gen(cond, w, seed=20)).cpu()
    row(f"one set, guidance {w:g}", accuracy(pg, one, mask))

    pg = predict(clf, gen(null_cond(args.n), 1.0, seed=30)).cpu()
    row("control (all any)", accuracy(pg, rand_labels(args.n)))


if __name__ == "__main__":
    main()
