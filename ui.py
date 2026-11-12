"""Tiny local web UI. Run `uv run python ui.py`, then open http://localhost:8000"""
import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from data import IMG_SIZE
from diffusion import Diffusion
from model import UNet

HTML = Path(__file__).with_name("ui.html")


class ClientGone(Exception):
    pass


def to_b64(x: torch.Tensor) -> str:
    """[N,1,H,W] in ~[-1,1] -> base64 of uint8 pixels, row-major per image."""
    px = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy().tobytes()
    return base64.b64encode(px).decode()


def make_handler(model, diff, lock, device):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = HTML.read_bytes()  # re-read each request so edits show on refresh
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/generate":
                self.generate(parse_qs(url.query))
            else:
                self.send_error(404)

        def emit(self, payload):
            try:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise ClientGone

        def generate(self, q):
            def arg(name, default, lo, hi):
                try:
                    return max(lo, min(hi, int(q.get(name, [default])[0])))
                except ValueError:
                    return default

            n = arg("n", 16, 1, 64)
            seed = arg("seed", 0, 0, 2**31 - 1)
            every = arg("every", 10, 1, diff.T)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def on_step(i, x):
                if i % every == 0 and i > 0:
                    self.emit({"step": diff.T - i, "total": diff.T, "px": to_b64(x)})

            try:
                with lock:  # one generation on the GPU at a time
                    g = torch.Generator(device=device).manual_seed(seed)
                    x = diff.sample(model, n, IMG_SIZE, generator=g, callback=on_step)
                self.emit({"step": diff.T, "total": diff.T, "px": to_b64(x), "done": True})
            except ClientGone:
                pass

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["ema"])
    handler = make_handler(model, Diffusion(device), threading.Lock(), device)
    print(f"Stick-figure diffusion UI on http://localhost:{args.port}  ({device})")
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
