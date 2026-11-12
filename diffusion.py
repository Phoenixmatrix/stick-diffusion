"""DDPM (Ho et al. 2020) from scratch: linear beta schedule, epsilon prediction, ancestral sampling."""
import torch
import torch.nn.functional as F

T = 1000


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

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model, n, size=32, generator=None, callback=None):
        """callback(i, x_i), if given, is called after each reverse step (i counts down T-1 .. 0)."""
        x = torch.randn(n, 1, size, size, device=self.device, generator=generator)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
            a, ab, b = self.alphas[i], self.abar[i], self.betas[i]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if i > 0:
                x = mean + self.post_var[i].sqrt() * torch.randn(x.shape, device=self.device, generator=generator)
            else:
                x = mean
            if callback is not None:
                callback(i, x)
        return x.clamp(-1, 1)
