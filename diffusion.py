"""DDPM (Ho et al. 2020) from scratch: linear beta schedule, epsilon prediction, ancestral sampling,
plus classifier-free guidance (Ho & Salimans 2022) for attribute conditioning."""
import torch
import torch.nn.functional as F

T = 1000
DEFAULT_GUIDANCE = 2.0  # chosen with eval_cond.py


class Diffusion:
    def __init__(self, device, T: int = T, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.abar_prev = torch.cat([torch.ones(1, device=device), self.abar[:-1]])
        # variance of q(x_{t-1} | x_t, x_0)
        self.post_var = self.betas * (1 - self.abar_prev) / (1 - self.abar)

    @staticmethod
    def _g(v, t):
        return v[t][:, None, None, None]

    def q_sample(self, x0, t, noise):
        return self._g(self.abar, t).sqrt() * x0 + (1 - self._g(self.abar, t)).sqrt() * noise

    def loss(self, model, x0, c, p_attr=0.2, p_all=0.1):
        """c: [B, n_attrs] labels. Each attribute is independently replaced by "any" with prob p_attr
        (teaches partial specification); the whole label is dropped with prob p_all (the
        unconditional model used by classifier-free guidance)."""
        null = torch.tensor(model.null_index, device=c.device)
        drop = torch.rand(c.shape, device=c.device) < p_attr
        drop |= (torch.rand(c.shape[0], 1, device=c.device) < p_all)
        c = torch.where(drop, null.expand_as(c), c)
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t, c), noise)

    @torch.no_grad()
    def sample(self, model, n, size=32, cond=None, guidance=DEFAULT_GUIDANCE, generator=None, callback=None):
        """cond: [n, n_attrs] labels ("any" = null index; None = all any).
        guidance: classifier-free guidance scale w, eps = eps_null + w * (eps_cond - eps_null); w = 1 is
        plain conditional sampling (one forward pass per step).
        callback(i, x_i), if given, is called after each reverse step (i counts down T-1 .. 0)."""
        null = torch.tensor(model.null_index, device=self.device).repeat(n, 1)
        cond = null if cond is None else cond.to(self.device)
        guided = guidance != 1 and not torch.equal(cond, null)
        x = torch.randn(n, 1, size, size, device=self.device, generator=generator)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            if guided:
                e_c, e_n = model(x.repeat(2, 1, 1, 1), t.repeat(2), torch.cat([cond, null])).chunk(2)
                eps = e_n + guidance * (e_c - e_n)
            else:
                eps = model(x, t, cond)
            a, ab, b = self.alphas[i], self.abar[i], self.betas[i]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if i > 0:
                x = mean + self.post_var[i].sqrt() * torch.randn(x.shape, device=self.device, generator=generator)
            else:
                x = mean
            if callback is not None:
                callback(i, x)
        return x.clamp(-1, 1)
