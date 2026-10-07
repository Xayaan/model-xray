"""What does each latent respond to? Its strongest activations on real text, how often it fires,
and which output tokens its decoder direction promotes (logit lens).

Output layout (served as static files by the browser):
  features/manifest.json             sites and their summary numbers
  features/<site>/index.json         one row per latent: density, max activation, top promoted tokens
  features/<site>/chunk_<n>.json     top examples and logit lists for latents n*chunk .. (n+1)*chunk-1
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import dist
from .config import RunConfig
from .data import token_batches
from .model import input_device, load_model, unembedding
from .sae import TopKDictionary
from .sites import Capture, Site
from .train import load_dictionaries, trained_sites


class TopExamples:
    """Streaming record of each latent's top-m activating windows (one per sequence) and its firing stats."""

    def __init__(self, d_sae: int, m: int, left: int, right: int, device):
        width = left + 1 + right
        self.m, self.left, self.right = m, left, right
        self.vals = torch.full((d_sae, m), -1.0, device=device)
        self.tokens = torch.full((d_sae, m, width), -1, dtype=torch.int32, device=device)
        self.acts = torch.zeros((d_sae, m, width), dtype=torch.float16, device=device)
        self.fire_count = torch.zeros(d_sae, dtype=torch.long, device=device)
        self.act_sum = torch.zeros(d_sae, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, vals: torch.Tensor, idx: torch.Tensor, input_ids: torch.Tensor, chunk: int = 256) -> None:
        """vals, idx: (B, T, k) sparse codes; input_ids: (B, T)."""
        B, T, k = vals.shape
        d_sae, m = self.vals.shape
        active = vals > 0
        self.fire_count += torch.bincount(idx[active], minlength=d_sae)
        self.act_sum += torch.zeros(d_sae, dtype=torch.float64, device=vals.device).scatter_add_(
            0, idx.flatten(), vals.flatten().double()
        )

        row_max = torch.zeros(B, d_sae, device=vals.device).scatter_reduce_(
            1, idx.reshape(B, T * k), vals.reshape(B, T * k), reduce="amax", include_self=True
        )
        new_vals, sel = torch.cat([self.vals, row_max.T], dim=1).topk(m, dim=1)
        old = sel.clamp(max=m - 1)[..., None].expand(-1, -1, self.tokens.shape[-1])
        new_tokens = torch.gather(self.tokens, 1, old)
        new_acts = torch.gather(self.acts, 1, old)

        lat, slot = ((sel >= m) & (new_vals > 0)).nonzero(as_tuple=True)
        rows = sel[lat, slot] - m
        offsets = torch.arange(-self.left, self.right + 1, device=vals.device)
        for start in range(0, len(lat), chunk):
            l, s, r = lat[start : start + chunk], slot[start : start + chunk], rows[start : start + chunk]
            act_l = (vals[r] * (idx[r] == l[:, None, None])).sum(-1)  # (P, T): this latent along the row
            positions = act_l.argmax(-1)[:, None] + offsets[None]
            valid = (positions >= 0) & (positions < T)
            clamped = positions.clamp(0, T - 1)
            toks = input_ids[r[:, None], clamped].masked_fill(~valid, -1)
            acts = act_l.gather(1, clamped).masked_fill(~valid, 0)
            new_tokens[l, s] = toks.to(torch.int32)
            new_acts[l, s] = acts.to(torch.float16)
        self.vals, self.tokens, self.acts = new_vals, new_tokens, new_acts


def vocab_strings(tokenizer, size: int) -> list[str]:
    n = len(tokenizer)
    return [tokenizer.decode([i]) if i < n else "" for i in range(size)]


@torch.no_grad()
def logit_lens(d: TopKDictionary, W_U: torch.Tensor, norm_w: torch.Tensor | None, top: int = 8, chunk: int = 1024):
    """For each decoder direction, the output tokens it pushes up and down most."""
    W = W_U.float()
    dirs = d.W_dec.float().to(W.device)
    if norm_w is not None:
        dirs = dirs * norm_w.float().to(W.device)
    ups, downs = [], []
    for start in range(0, dirs.shape[0], chunk):
        logits = dirs[start : start + chunk] @ W.T
        ups.append(logits.topk(top, dim=-1))
        downs.append((-logits).topk(top, dim=-1))
    up_v = torch.cat([u.values for u in ups]).cpu()
    up_i = torch.cat([u.indices for u in ups]).cpu()
    dn_v = torch.cat([-dn.values for dn in downs]).cpu()
    dn_i = torch.cat([dn.indices for dn in downs]).cpu()
    return up_v, up_i, dn_v, dn_i


def write_site(out_dir: Path, site: Site, d: TopKDictionary, tracker: TopExamples, vocab: list[str],
               n_tokens: int, lens, chunk_size: int, labels: dict | None = None) -> dict:
    site_dir = out_dir / site.name
    site_dir.mkdir(parents=True, exist_ok=True)
    fire = tracker.fire_count.cpu()
    density = (fire.double() / max(n_tokens, 1)).tolist()
    mean_act = (tracker.act_sum.cpu() / fire.clamp_min(1)).tolist()
    max_act = tracker.vals[:, 0].clamp_min(0).cpu().tolist()
    up_v, up_i, dn_v, dn_i = lens
    tokens, acts, vals = tracker.tokens.cpu(), tracker.acts.float().cpu(), tracker.vals.cpu()
    labels = labels or {}

    index = []
    for j in range(d.d_sae):
        index.append({
            "id": j,
            "density": round(density[j], 7),
            "mean_act": round(mean_act[j], 4),
            "max_act": round(max_act[j], 4),
            "up": [vocab[t] for t in up_i[j, :5].tolist()],
            "label": labels.get(str(j)),
        })
    (site_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    for c, start in enumerate(range(0, d.d_sae, chunk_size)):
        feats = {}
        for j in range(start, min(start + chunk_size, d.d_sae)):
            examples = []
            for e in range(vals.shape[1]):
                if vals[j, e] <= 0:
                    break
                row_t, row_a = tokens[j, e].tolist(), acts[j, e].tolist()
                keep = [i for i, t in enumerate(row_t) if t >= 0]
                examples.append({
                    "tokens": [vocab[row_t[i]] for i in keep],
                    "acts": [round(row_a[i], 3) for i in keep],
                })
            feats[j] = {
                "examples": examples,
                "up": [[vocab[t], round(v, 3)] for t, v in zip(up_i[j].tolist(), up_v[j].tolist())],
                "down": [[vocab[t], round(v, 3)] for t, v in zip(dn_i[j].tolist(), dn_v[j].tolist())],
            }
        (site_dir / f"chunk_{c}.json").write_text(json.dumps({"start": start, "features": feats}, separators=(",", ":")))

    alive = sum(1 for x in density if x > 0)
    return {
        "site": site.name,
        "kind": site.kind,
        "layer": site.layer,
        "d_sae": d.d_sae,
        "k": d.k,
        "alive_fraction": alive / d.d_sae,
        "chunk_size": chunk_size,
    }


def analyze(cfg: RunConfig, n_tokens: int | None = None) -> Path:
    """Analyse every trained dictionary. On several GPUs each process takes its own share and the
    main process merges the per-process manifests."""
    rank, world, gpu = dist.setup()
    out_dir = cfg.run_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        if dist.is_main():  # clear leftovers from an interrupted run before anyone writes
            for stale in out_dir.glob("manifest.rank*.json"):
                stale.unlink()
        dist.barrier()
    model, tokenizer = load_model(cfg.model, gpu)
    device = input_device(model)
    names = dist.owned([site.name for site, _ in trained_sites(cfg.run_dir)], rank, world)
    loaded = load_dictionaries(cfg.run_dir, model, names)
    transcoder = cfg.dictionary.kind == "transcoder"
    acfg, skip = cfg.analysis, cfg.data.skip_first
    budget = n_tokens or acfg.n_tokens
    sites = [site for site, _ in loaded.values()]
    trackers = {
        name: TopExamples(d.d_sae, acfg.top_examples, acfg.window_left, acfg.window_right, d.W_enc.device)
        for name, (_, d) in loaded.items()
    }
    manifest = {"model": cfg.model.name, "run": cfg.name, "kind": cfg.dictionary.kind, "tokens": 0, "sites": []}
    if not loaded:  # more GPUs than dictionaries: nothing for this process to do
        return _merge_manifests(out_dir, manifest, rank, world)

    batches = token_batches(
        tokenizer,
        cfg.data.eval_sources or cfg.data.sources,
        cfg.data.ctx_len,
        cfg.data.batch_size,
        seed=cfg.data.seed + 2000,
        shuffle_buffer=cfg.data.shuffle_buffer,
    )
    seen = 0
    with Capture(model, sites, need_input={s.name for s in sites} if transcoder else set()) as cap:
        for input_ids in batches:
            input_ids = input_ids.to(device)
            acts = cap.run(input_ids)
            B, T = input_ids.shape
            for name, (site, d) in loaded.items():
                source = acts[site.name]["in" if transcoder else "out"]
                vals, idx = d.encode(source.reshape(B * T, -1).float())
                vals, idx = vals.reshape(B, T, -1), idx.reshape(B, T, -1)
                vals[:, :skip] = 0
                trackers[name].update(vals, idx, input_ids.to(vals.device))
            seen += B * (T - skip)
            if dist.is_main():
                print(f"\ranalysed {seen:,}/{budget:,} tokens", end="", flush=True)
            if seen >= budget:
                break
    if dist.is_main():
        print()

    W_U, norm_w = unembedding(model)
    vocab = vocab_strings(tokenizer, W_U.shape[0])
    eval_path = cfg.run_dir / "eval.json"
    eval_sites = json.loads(eval_path.read_text())["sites"] if eval_path.exists() else {}
    manifest["tokens"] = seen
    for name, (site, d) in loaded.items():
        labels_path = out_dir / name / "labels.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else None
        lens = logit_lens(d, W_U, norm_w)
        summary = write_site(out_dir, site, d, trackers[name], vocab, seen, lens, acfg.chunk_size, labels)
        summary["eval"] = eval_sites.get(name)
        manifest["sites"].append(summary)
        print(f"{name}: {summary['alive_fraction']:.1%} of {d.d_sae:,} latents fired")
    return _merge_manifests(out_dir, manifest, rank, world)


def _merge_manifests(out_dir: Path, manifest: dict, rank: int, world: int) -> Path:
    """Write manifest.json; on several GPUs the main process merges every process's share."""
    if world > 1:
        (out_dir / f"manifest.rank{rank}.json").write_text(json.dumps(manifest))
        dist.barrier()
        if not dist.is_main():
            return out_dir
        parts = [json.loads(p.read_text()) for p in sorted(out_dir.glob("manifest.rank*.json"))]
        manifest = {**parts[0], "tokens": max(p["tokens"] for p in parts), "sites": [s for p in parts for s in p["sites"]]}
        manifest["sites"].sort(key=lambda s: (s["layer"], s["kind"]))
        for p in out_dir.glob("manifest.rank*.json"):
            p.unlink()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out_dir
