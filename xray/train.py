"""Train one dictionary per site, all from the same forward passes.

Activations from several forward passes are pooled in a shuffle buffer before each training step,
so a step mixes tokens from many documents instead of a handful of sequences. On several GPUs
(see dist.py) every process reads its own data, and the activations are swapped after each forward
pass so each process trains its own share of the dictionaries on everyone's tokens.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import dist
from .config import RunConfig
from .data import token_batches
from .model import describe, input_device, load_model, module_device
from .sae import TopKDictionary, default_lr
from .sites import Capture, Site, resolve_sites


@dataclass
class SiteState:
    site: Site
    dictionary: TopKDictionary
    optimizer: torch.optim.Optimizer
    tokens_since_fired: torch.Tensor
    fire_count: torch.Tensor
    log: dict = field(default_factory=lambda: {"fvu": 0.0, "aux": 0.0, "n": 0})


def site_rows(acts: dict, site: Site, transcoder: bool, skip_first: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(input, target) rows for a site in the model's dtype, first positions of each sequence dropped.
    For SAEs the input is the target."""
    out = acts[site.name]["out"][:, skip_first:]
    out = out.reshape(-1, out.shape[-1])
    if not transcoder:
        return out, out
    inp = acts[site.name]["in"][:, skip_first:]
    return inp.reshape(-1, inp.shape[-1]), out


def lr_at(step: int, total_steps: int, base: float, warmup: int, decay_frac: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    decay_start = int(total_steps * (1 - decay_frac))
    if step >= decay_start and total_steps > decay_start:
        return base * max(0.0, (total_steps - step) / (total_steps - decay_start))
    return base


class ShuffleBuffer:
    """Pools (input, target) rows per site. Once `capacity` rows are held, `drain` yields shuffled
    full batches until half the capacity is left, which stays behind to mix with the next passes.
    Sites may live on different devices (a model spread over GPUs)."""

    def __init__(self, capacity: int, batch: int, paired: bool):
        self.capacity, self.batch, self.paired = capacity, batch, paired
        self.rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return next(iter(self.rows.values()))[0].shape[0] if self.rows else 0

    def add(self, new: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        for name, (x, y) in new.items():
            if name in self.rows:
                old_x, old_y = self.rows[name]
                x = torch.cat([old_x, x])
                y = torch.cat([old_y, y]) if self.paired else x
            self.rows[name] = (x, y)

    def drain(self):
        n = len(self)
        if n < self.capacity:
            return
        perm = torch.randperm(n)
        n_train = (n - self.capacity // 2) // self.batch * self.batch
        for start in range(0, n_train, self.batch):
            idx = perm[start : start + self.batch]
            batch = {}
            for name, (x, y) in self.rows.items():
                i = idx.to(x.device)
                batch[name] = (x[i], y[i] if self.paired else None)
            yield batch
        keep = perm[n_train:]
        for name, (x, y) in self.rows.items():
            k = keep.to(x.device)
            kept = x[k]
            self.rows[name] = (kept, y[k] if self.paired else kept)


def save_states(states: dict[str, SiteState], cfg: RunConfig, tokens: int) -> None:
    for name, st in states.items():
        st.dictionary.save(
            cfg.run_dir / "dictionaries" / name,
            extra={
                "site": st.site.to_dict(),
                "model": cfg.model.name,
                "kind": cfg.dictionary.kind,
                "tokens": tokens,
                "dead_fraction": float((st.tokens_since_fired > cfg.dictionary.dead_after_tokens).float().mean()),
                "fire_count": int(st.fire_count.sum()),
            },
        )


def _init_states(sites, rows, cfg: RunConfig, transcoder: bool) -> dict[str, SiteState]:
    dcfg = cfg.dictionary
    states = {}
    for site in sites:
        x, y = (t.float() for t in rows[site.name])
        d = TopKDictionary(
            x.shape[-1], dcfg.expansion * x.shape[-1], dcfg.k,
            d_out=y.shape[-1], transcoder=transcoder, skip=dcfg.skip,
        ).to(x.device)
        d.init_from_data(x, y)
        states[site.name] = SiteState(
            site=site,
            dictionary=d,
            optimizer=torch.optim.Adam(d.parameters(), lr=dcfg.lr or default_lr(d.d_sae), betas=(0.9, 0.999), eps=6.25e-10),
            tokens_since_fired=torch.zeros(d.d_sae, dtype=torch.long, device=x.device),
            fire_count=torch.zeros(d.d_sae, dtype=torch.long, device=x.device),
        )
    return states


def train(cfg: RunConfig, sites_override: list[str] | None = None) -> dict[str, SiteState]:
    rank, world, gpu = dist.setup()
    if world > 1 and cfg.model.device_map:
        raise ValueError("use either --gpus (a model copy per GPU) or model.device_map (one model spread over GPUs), not both")
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(cfg.model, gpu)
    device = input_device(model)
    dcfg, tcfg = cfg.dictionary, cfg.train
    transcoder = dcfg.kind == "transcoder"
    sites = resolve_sites(model, sites_override or tcfg.sites)
    if transcoder and any(s.kind != "mlp" for s in sites):
        raise ValueError("transcoders need MLP sites (use dictionary.kind: sae for resid/mixer sites)")
    if tcfg.buffer_batches < 2:
        raise ValueError("train.buffer_batches must be at least 2")
    names = [s.name for s in sites]
    mine = dist.owned(sites, rank, world)
    if dist.is_main():
        cfg.save(run_dir / "config.yaml")
        (run_dir / "architecture.json").write_text(json.dumps(describe(model), indent=1))
    tag = f"[gpu {rank}] " if world > 1 else ""
    print(f"{tag}training {len(mine)} {dcfg.kind}s: {', '.join(s.name for s in mine) or 'none'}"
          + (f" ({len(sites)} across {world} GPUs)" if world > 1 else ""), flush=True)

    batches = iter(token_batches(
        tokenizer, cfg.data.sources, cfg.data.ctx_len, cfg.data.batch_size, cfg.data.seed,
        cfg.data.shuffle_buffer, rank, world,
    ))
    buffer = ShuffleBuffer(tcfg.buffer_batches * tcfg.train_batch, tcfg.train_batch, paired=transcoder)
    total_steps = max(1, tcfg.total_tokens // tcfg.train_batch)
    states: dict[str, SiteState] = {}
    log_file = (run_dir / (f"train_log.rank{rank}.jsonl" if world > 1 else "train_log.jsonl")).open("a")
    tokens, step, next_save = 0, 0, tcfg.save_every_tokens
    aux_k = None
    started = time.time()
    deadline = started + tcfg.max_minutes * 60 if tcfg.max_minutes else None
    out_of_time = ran_dry = False

    with Capture(model, sites, need_input=set(names) if transcoder else set()) as capture:
        while True:
            input_ids = next(batches, None)
            if dist.any_true(input_ids is None, device):  # every process stops when any shard runs dry
                ran_dry = True
                break
            acts = capture.run(input_ids.to(device))
            rows = {s.name: site_rows(acts, s, transcoder, cfg.data.skip_first) for s in sites}
            rows = dist.exchange(rows, names, transcoder, rank, world)
            if mine and not states:
                states = _init_states(mine, rows, cfg, transcoder)
                aux_k = dcfg.aux_k or min(512, rows[mine[0].name][0].shape[-1] // 2)
            buffer.add(rows)  # a process with more GPUs than sites to own still forwards and sends

            for batch in buffer.drain():
                for name, st in states.items():
                    x, y = batch[name]
                    x = x.float()
                    y = y.float() if y is not None else x
                    d = st.dictionary
                    for group in st.optimizer.param_groups:
                        group["lr"] = lr_at(step, total_steps, dcfg.lr or default_lr(d.d_sae), tcfg.warmup_steps, tcfg.decay_frac)
                    fvu, aux, idx = d.loss(x, y, dead=st.tokens_since_fired > dcfg.dead_after_tokens, aux_k=aux_k)
                    st.optimizer.zero_grad(set_to_none=True)
                    (fvu + dcfg.aux_coef * aux).backward()
                    d.remove_parallel_grads()
                    st.optimizer.step()
                    d.normalize_decoder()

                    fired = torch.bincount(idx.flatten(), minlength=d.d_sae)
                    st.fire_count += fired
                    st.tokens_since_fired += x.shape[0]
                    st.tokens_since_fired[fired > 0] = 0
                    st.log["fvu"] += fvu.item()
                    st.log["aux"] += aux.item()
                    st.log["n"] += 1

                step += 1
                tokens += tcfg.train_batch
                if step % tcfg.log_every == 0 or tokens >= tcfg.total_tokens:
                    _log(states, step, tokens, total_steps, started, dcfg.dead_after_tokens, log_file, tag)
                if tokens >= next_save:
                    save_states(states, cfg, tokens)
                    next_save += tcfg.save_every_tokens
                if tokens >= tcfg.total_tokens:
                    break  # every process counts the same tokens, so all stop at the same step

            out_of_time = bool(deadline) and time.time() > deadline
            if dist.any_true(tokens >= tcfg.total_tokens or out_of_time, device):
                out_of_time = tokens < tcfg.total_tokens
                break

    if out_of_time and states and step % tcfg.log_every:
        _log(states, step, tokens, total_steps, started, dcfg.dead_after_tokens, log_file, tag)
    log_file.close()
    if mine and not states:
        raise RuntimeError("no data: the sources produced no tokens")
    if mine and out_of_time:
        print(f"{tag}time limit of {tcfg.max_minutes} min reached after {tokens:,} of {tcfg.total_tokens:,} tokens")
    elif mine and ran_dry and tokens < tcfg.total_tokens:
        print(f"{tag}warning: data ran out after {tokens:,} of {tcfg.total_tokens:,} tokens")
    save_states(states, cfg, tokens)
    dist.barrier()
    print(f"{tag}done: {tokens:,} tokens in {time.time() - started:.0f}s -> {run_dir / 'dictionaries'}", flush=True)
    return states


def _log(states, step, tokens, total_steps, started, dead_after, log_file, tag="") -> None:
    elapsed = time.time() - started
    record = {"step": step, "tokens": tokens, "elapsed_s": round(elapsed, 1), "sites": {}}
    parts = []
    for name, st in states.items():
        n = max(1, st.log["n"])
        fvu, aux = st.log["fvu"] / n, st.log["aux"] / n
        dead = float((st.tokens_since_fired > dead_after).float().mean())
        record["sites"][name] = {"fvu": round(fvu, 5), "aux": round(aux, 5), "dead": round(dead, 4)}
        parts.append(f"{name} fvu={fvu:.3f} dead={dead:.1%}")
        st.log = {"fvu": 0.0, "aux": 0.0, "n": 0}
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()
    rate = tokens / max(elapsed, 1e-6)
    print(f"{tag}step {step}/{total_steps} | {tokens:,} tok | {rate:,.0f} tok/s | " + " | ".join(parts), flush=True)


def trained_sites(run_dir: str | Path) -> list[tuple[Site, Path]]:
    """Every trained dictionary's site and directory, in network order."""
    found = []
    for path in (Path(run_dir) / "dictionaries").iterdir():
        cfg = json.loads((path / "cfg.json").read_text())
        found.append((Site(**cfg["site"]), path))
    found.sort(key=lambda item: (item[0].layer, item[0].kind))
    return found


def load_dictionaries(run_dir: str | Path, model, names: list[str] | None = None) -> dict[str, tuple[Site, TopKDictionary]]:
    """Trained dictionaries keyed by site name, in network order, each placed on the device that
    holds its site in `model` (one GPU, or the right GPU of a model spread over several)."""
    out = {}
    for site, path in trained_sites(run_dir):
        if names is None or site.name in names:
            out[site.name] = (site, TopKDictionary.load(path, module_device(model.get_submodule(site.path))))
    return out
