"""Tiny UNet for 32x32 grayscale diffusion. Written from scratch, randomly initialised."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=(32, 64, 128), tdim=128):
        super().__init__()
        c0, c1, c2 = ch
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.inc = nn.Conv2d(1, c0, 3, padding=1)
        self.d0 = ResBlock(c0, c0, tdim)                 # 32x32
        self.down0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.d1 = ResBlock(c1, c1, tdim)                 # 16x16
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)

        self.mid1 = ResBlock(c2, c2, tdim)               # 8x8
        self.mid2 = ResBlock(c2, c2, tdim)

        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.u1 = ResBlock(c1 * 2, c1, tdim)             # 16x16
        self.up0 = nn.Conv2d(c1, c0, 3, padding=1)
        self.u0 = ResBlock(c0 * 2, c0, tdim)             # 32x32

        self.out = nn.Sequential(nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        h0 = self.d0(self.inc(x), temb)
        h1 = self.d1(self.down0(h0), temb)
        h = self.mid2(self.mid1(self.down1(h1), temb), temb)
        h = self.up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.u1(torch.cat([h, h1], 1), temb)
        h = self.up0(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.u0(torch.cat([h, h0], 1), temb)
        return self.out(h)


if __name__ == "__main__":
    m = UNet()
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params")
    print(m(torch.randn(2, 1, 32, 32), torch.tensor([0, 999])).shape)
