"""TopK sparse dictionaries (Gao et al., 2024).

As an SAE the dictionary reconstructs an activation x. As a transcoder (Dunefsky et al., 2024) it
predicts an MLP's output from the MLP's input, optionally with a linear skip connection
(Paulo et al., 2025). Training happens in normalised units: inputs and targets are rescaled so
their mean norm is sqrt(d); the scales are stored and folded into the weights on export.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn


def sparse_decode(vals: torch.Tensor, idx: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """sum_j vals[n, j] * W[idx[n, j]] without materialising the dense code."""
    if W.device.type == "mps":
        dense = vals.new_zeros(vals.shape[0], W.shape[0]).scatter_(1, idx, vals)
        return dense @ W
    return F.embedding_bag(idx, W, per_sample_weights=vals, mode="sum")


class TopKDictionary(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int, *, d_out: int | None = None, transcoder: bool = False, skip: bool = False):
        super().__init__()
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        self.d_out = d_out or d_in
        self.transcoder = transcoder
        self.skip = skip and transcoder
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, self.d_out))
        self.b_dec = nn.Parameter(torch.zeros(self.d_out))
        self.W_skip = nn.Parameter(torch.zeros(d_in, self.d_out)) if self.skip else None
        self.register_buffer("in_scale", torch.ones(()))
        self.register_buffer("out_scale", torch.ones(()))
        self.reset_parameters()

    @property
    def meta(self) -> dict:
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "d_out": self.d_out,
            "k": self.k,
            "transcoder": self.transcoder,
            "skip": self.skip,
        }

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.W_dec)
        self.normalize_decoder()
        if self.transcoder:
            nn.init.normal_(self.W_enc, std=self.d_in**-0.5)
        else:
            self.W_enc.copy_(self.W_dec.T)

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor, y: torch.Tensor | None = None) -> None:
        """Set the normalisation scales and start b_dec at the mean target."""
        self.in_scale.fill_(self.d_in**0.5 / x.norm(dim=-1).mean().clamp_min(1e-8))
        if self.transcoder:
            self.out_scale.fill_(self.d_out**0.5 / y.norm(dim=-1).mean().clamp_min(1e-8))
            self.b_dec.copy_((y * self.out_scale).mean(0))
        else:
            self.out_scale.copy_(self.in_scale)
            self.b_dec.copy_((x * self.in_scale).mean(0))

    # ----- forward passes -------------------------------------------------------------------

    def _pre(self, xs: torch.Tensor) -> torch.Tensor:
        if not self.transcoder:
            xs = xs - self.b_dec
        return xs @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (values, indices) of the k active latents for each row of x (original units)."""
        vals, idx = self._pre(x * self.in_scale).topk(self.k, dim=-1, sorted=False)
        return vals.relu(), idx

    def decode(self, vals: torch.Tensor, idx: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        out = sparse_decode(vals, idx, self.W_dec) + self.b_dec
        if self.skip:
            out = out + (x * self.in_scale) @ self.W_skip
        return out / self.out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals, idx = self.encode(x)
        return self.decode(vals, idx, x)

    def dense_codes(self, x: torch.Tensor) -> torch.Tensor:
        vals, idx = self.encode(x)
        return vals.new_zeros(x.shape[0], self.d_sae).scatter_(1, idx, vals)

    def loss(self, x: torch.Tensor, y: torch.Tensor, dead: torch.Tensor | None = None, aux_k: int = 512):
        """Fraction of variance unexplained, plus the AuxK loss that revives dead latents."""
        xs, ys = x * self.in_scale, y * self.out_scale
        pre = self._pre(xs)
        vals, idx = pre.topk(self.k, dim=-1, sorted=False)
        vals = vals.relu()
        recon = sparse_decode(vals, idx, self.W_dec) + self.b_dec
        if self.skip:
            recon = recon + xs @ self.W_skip
        err = ys - recon
        total_var = (ys - ys.mean(0)).pow(2).sum(-1).mean().clamp_min(1e-8)
        fvu = err.pow(2).sum(-1).mean() / total_var

        aux = fvu.new_zeros(())
        if dead is not None and bool(dead.any()):
            k_aux = min(aux_k, int(dead.sum()))
            a_vals, a_idx = pre.masked_fill(~dead, float("-inf")).topk(k_aux, dim=-1, sorted=False)
            err_hat = sparse_decode(a_vals.relu(), a_idx, self.W_dec)
            e = err.detach()
            aux = (e - err_hat).pow(2).sum(-1).mean() / (e - e.mean(0)).pow(2).sum(-1).mean().clamp_min(1e-8)
        return fvu, aux, idx

    # ----- constraints ------------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_parallel_grads(self) -> None:
        """Keep decoder rows on the unit sphere: drop the gradient component along each row."""
        if self.W_dec.grad is not None:
            g, w = self.W_dec.grad, self.W_dec
            g.sub_((g * w).sum(dim=1, keepdim=True) * w)

    # ----- export -----------------------------------------------------------------------------

    @torch.no_grad()
    def folded(self) -> dict[str, torch.Tensor]:
        """Weights with the normalisation folded in, so they act on raw activations.

        SAE:        f = topk(relu((x - b_dec) @ W_enc + b_enc)),  x_hat = f @ W_dec + b_dec
        transcoder: f = topk(relu(x @ W_enc + b_enc)),            y_hat = f @ W_dec + b_dec + x @ W_skip
        """
        s_in, s_out = self.in_scale, self.out_scale
        out = {
            "W_enc": self.W_enc * s_in,
            "b_enc": self.b_enc.clone(),
            "W_dec": self.W_dec / s_out,
            "b_dec": self.b_dec / s_out,
        }
        if self.skip:
            out["W_skip"] = self.W_skip * s_in / s_out
        return {key: value.detach().float().cpu().contiguous() for key, value in out.items()}

    def save(self, directory: str | Path, extra: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, directory / "dictionary.safetensors")
        save_file(self.folded(), directory / "folded.safetensors")
        (directory / "cfg.json").write_text(json.dumps({**self.meta, "architecture": "topk", **(extra or {})}, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device = "cpu") -> "TopKDictionary":
        directory = Path(directory)
        cfg = json.loads((directory / "cfg.json").read_text())
        model = cls(cfg["d_in"], cfg["d_sae"], cfg["k"], d_out=cfg["d_out"], transcoder=cfg["transcoder"], skip=cfg["skip"])
        model.load_state_dict(load_file(directory / "dictionary.safetensors"))
        return model.to(device).eval()


def default_lr(d_sae: int) -> float:
    return 2e-4 / (d_sae / 2**14) ** 0.5
