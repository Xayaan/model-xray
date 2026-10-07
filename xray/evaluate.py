"""How good are the dictionaries? Reconstruction quality, and how much loss the model keeps when a
dictionary's output is spliced in place of the real activation.

On several GPUs each process evaluates its own share of the dictionaries on the same text, and the
main process merges the results into eval.json.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from . import dist
from .config import RunConfig
from .data import token_batches
from .model import input_device, load_model
from .sites import Capture, Patch
from .train import load_dictionaries, site_rows, trained_sites


def token_ce(logits: torch.Tensor, input_ids: torch.Tensor, skip_first: int) -> tuple[float, int]:
    """Summed next-token cross-entropy over positions >= skip_first, row by row to bound memory."""
    total, count = 0.0, 0
    input_ids = input_ids.to(logits.device)
    for row_logits, row_ids in zip(logits, input_ids):
        pred, target = row_logits[skip_first:-1].float(), row_ids[skip_first + 1 :]
        total += float(F.cross_entropy(pred, target, reduction="sum"))
        count += target.numel()
    return total, count


def _evaluate_sites(cfg: RunConfig, model, tokenizer, loaded, budget: int) -> dict:
    transcoder = cfg.dictionary.kind == "transcoder"
    skip = cfg.data.skip_first
    device = input_device(model)
    sites = [site for site, _ in loaded.values()]
    batches = token_batches(
        tokenizer,
        cfg.data.eval_sources or cfg.data.sources,
        cfg.data.ctx_len,
        cfg.eval.batch_size,
        seed=cfg.data.seed + 1000,
        shuffle_buffer=cfg.data.shuffle_buffer,
    )
    acc = {name: defaultdict(float) for name in loaded}
    clean_sum, clean_count = 0.0, 0

    for input_ids in batches:
        input_ids = input_ids.to(device)
        with Capture(model, sites, need_input={s.name for s in sites} if transcoder else set(), stop_early=False) as cap:
            acts = cap.run(input_ids)
            ce, count = token_ce(cap.output.logits, input_ids, skip)
        clean_sum += ce
        clean_count += count

        for name, (site, d) in loaded.items():
            with torch.no_grad():
                x, y = (t.float() for t in site_rows(acts, site, transcoder, skip))
                y_hat = d(x)
                acc[name]["err"] += float((y - y_hat).pow(2).sum())
                acc[name]["var"] += float((y - y.mean(0)).pow(2).sum())
                acc[name]["l0"] += float((d.encode(x)[0] > 0).float().sum(-1).mean()) * x.shape[0]
                acc[name]["rows"] += x.shape[0]
                mean_out = y.mean(0)

            def spliced(inp, out, d=d):
                new = out.clone()
                source = inp if transcoder else out
                recon = d(source[:, skip:].reshape(-1, source.shape[-1]).float())
                new[:, skip:] = recon.reshape(out[:, skip:].shape).to(out.dtype)
                return new

            def ablated(inp, out, site=site, mean_out=mean_out):
                new = out.clone()
                new[:, skip:] = mean_out.to(out.dtype) if site.kind == "resid" else 0
                return new

            for label, fn in (("spliced", spliced), ("ablated", ablated)):
                with Patch(model, site, fn), torch.no_grad():
                    logits = model(input_ids=input_ids, use_cache=False).logits
                acc[name][f"ce_{label}"] += token_ce(logits, input_ids, skip)[0]

        if clean_count >= budget:
            break

    ce_clean = clean_sum / clean_count
    results = {"tokens": clean_count, "ce_clean": ce_clean, "sites": {}}
    for name, (site, d) in loaded.items():
        a = acc[name]
        fvu = a["err"] / max(a["var"], 1e-8)
        ce_spliced, ce_ablated = a["ce_spliced"] / clean_count, a["ce_ablated"] / clean_count
        denom = ce_ablated - ce_clean
        meta = json.loads((cfg.run_dir / "dictionaries" / name / "cfg.json").read_text())
        results["sites"][name] = {
            "kind": site.kind,
            "layer": site.layer,
            "d_sae": d.d_sae,
            "k": d.k,
            "fvu": fvu,
            "explained_variance": 1 - fvu,
            "l0": a["l0"] / max(a["rows"], 1),
            "ce_spliced": ce_spliced,
            "ce_ablated": ce_ablated,
            "ce_delta": ce_spliced - ce_clean,
            "loss_recovered": (ce_ablated - ce_spliced) / denom if abs(denom) > 1e-9 else float("nan"),
            "dead_fraction": meta.get("dead_fraction"),
            "train_tokens": meta.get("tokens"),
        }
    return results


def evaluate(cfg: RunConfig, n_tokens: int | None = None) -> dict:
    rank, world, gpu = dist.setup()
    if world > 1:
        if dist.is_main():  # clear leftovers from an interrupted run before anyone writes
            for stale in cfg.run_dir.glob("eval.rank*.json"):
                stale.unlink()
        dist.barrier()
    model, tokenizer = load_model(cfg.model, gpu)
    names = dist.owned([site.name for site, _ in trained_sites(cfg.run_dir)], rank, world)
    loaded = load_dictionaries(cfg.run_dir, model, names)
    partial = _evaluate_sites(cfg, model, tokenizer, loaded, n_tokens or cfg.eval.n_tokens) if loaded else None

    if world > 1:
        part_path = cfg.run_dir / f"eval.rank{rank}.json"
        if partial:
            part_path.write_text(json.dumps(partial))
        dist.barrier()
        if not dist.is_main():
            return partial or {}
        parts = [json.loads(p.read_text()) for p in sorted(cfg.run_dir.glob("eval.rank*.json"))]
        results = {"tokens": parts[0]["tokens"], "ce_clean": parts[0]["ce_clean"], "sites": {}}
        for part in parts:
            results["sites"].update(part["sites"])
        results["sites"] = dict(sorted(results["sites"].items(), key=lambda kv: (kv[1]["layer"], kv[1]["kind"])))
        for p in cfg.run_dir.glob("eval.rank*.json"):
            p.unlink()
    else:
        results = partial

    (cfg.run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"clean CE {results['ce_clean']:.4f} over {results['tokens']:,} tokens")
    print(f"{'site':<12}{'expl.var':>10}{'L0':>8}{'CE +':>10}{'loss rec.':>11}{'dead':>8}")
    for name, r in results["sites"].items():
        print(
            f"{name:<12}{r['explained_variance']:>10.3f}{r['l0']:>8.1f}{r['ce_delta']:>10.4f}"
            f"{r['loss_recovered']:>11.3f}{(r['dead_fraction'] or 0):>8.1%}"
        )
    return results
