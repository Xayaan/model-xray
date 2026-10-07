"""Sites: places in the network we read activations from and can write activations into.

A site is one of
  resid  the residual stream after block L (the block's output)
  mlp    the MLP of block L; transcoders map its input to its output
  mixer  the token mixer of block L (attention or Mamba); SAEs read its output
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import nn

from .model import find_layers, mixer_attr, mlp_attr

KIND_ORDER = {"mixer": 0, "mlp": 1, "resid": 2}  # execution order inside a block


@dataclass(frozen=True)
class Site:
    name: str
    layer: int
    kind: str
    path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_layers(selector: str, candidates: list[int]) -> list[int]:
    if selector == "all":
        return candidates
    if selector.startswith("evenly"):
        n = int(selector[len("evenly"):])
        if n >= len(candidates):
            return candidates
        step = (len(candidates) - 1) / (n - 1) if n > 1 else 0
        return sorted({candidates[round(i * step)] for i in range(n)})
    chosen: list[int] = []
    for part in selector.split(","):
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-"))
            chosen.extend(range(lo, hi + 1))
        else:
            chosen.append(int(part))
    missing = [i for i in chosen if i not in candidates]
    if missing:
        raise ValueError(f"layers {missing} have no such site (valid: {candidates})")
    return chosen


def resolve_sites(model: nn.Module, specs: list[str]) -> list[Site]:
    """Turn specs such as "mlp:all", "mlp:evenly4", "resid:0-5" or "mixer:3,7" into sites."""
    layers_path, layers = find_layers(model)
    sites: dict[str, Site] = {}
    for spec in specs:
        kind, _, selector = spec.partition(":")
        selector = selector or "all"
        if kind == "resid":
            candidates = {i: f"{layers_path}.{i}" for i in range(len(layers))}
        elif kind == "mlp":
            candidates = {
                i: f"{layers_path}.{i}.{attr}"
                for i, block in enumerate(layers)
                if (attr := mlp_attr(block)) is not None
            }
        elif kind == "mixer":
            candidates = {
                i: f"{layers_path}.{i}.{attr}"
                for i, block in enumerate(layers)
                if (attr := mixer_attr(block)) is not None and mlp_attr(block) != attr
            }
        else:
            raise ValueError(f"unknown site kind {kind!r} in {spec!r}")
        for i in _parse_layers(selector, sorted(candidates)):
            site = Site(name=f"L{i}.{kind}", layer=i, kind=kind, path=candidates[i])
            sites[site.name] = site
    return sorted(sites.values(), key=lambda s: (s.layer, KIND_ORDER[s.kind]))


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return value[0]
    return value[0]  # ModelOutput-style containers


def replace_first(value, new: torch.Tensor):
    if isinstance(value, torch.Tensor):
        return new
    if isinstance(value, tuple):
        return (new, *value[1:])
    if isinstance(value, list):
        return [new, *value[1:]]
    value[0] = new
    return value


def _hook_input(args, kwargs) -> torch.Tensor:
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    return next(v for v in kwargs.values() if isinstance(v, torch.Tensor))


class StopForward(Exception):
    """Raised from a hook once every requested activation has been recorded."""


class Capture:
    """Record activations at sites. `run` returns {site_name: {"in": tensor, "out": tensor}}.

    With stop_early=True the forward pass is abandoned after the deepest site, which skips the
    remaining blocks and the (large) unembedding. Otherwise the model output is kept in `output`.
    """

    def __init__(self, model: nn.Module, sites: list[Site], need_input: set[str] = frozenset(), stop_early: bool = True):
        self.model = model
        self.sites = sites
        self.need_input = set(need_input)
        self.stop_early = stop_early
        self.acts: dict[str, dict[str, torch.Tensor]] = {}
        self.output = None
        self._handles: list = []

    def __enter__(self) -> "Capture":
        last = max(self.sites, key=lambda s: (s.layer, KIND_ORDER[s.kind]))
        for site in self.sites:
            module = self.model.get_submodule(site.path)
            if site.name in self.need_input:
                self._handles.append(
                    module.register_forward_pre_hook(self._pre_hook(site.name), with_kwargs=True)
                )
            stop = self.stop_early and site == last
            self._handles.append(module.register_forward_hook(self._post_hook(site.name, stop)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _pre_hook(self, name: str) -> Callable:
        def hook(module, args, kwargs):
            self.acts.setdefault(name, {})["in"] = _hook_input(args, kwargs).detach()

        return hook

    def _post_hook(self, name: str, stop: bool) -> Callable:
        def hook(module, args, output):
            self.acts.setdefault(name, {})["out"] = first_tensor(output).detach()
            if stop:
                raise StopForward

        return hook

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        self.acts, self.output = {}, None
        try:
            self.output = self.model(input_ids=input_ids, use_cache=False, **kwargs)
        except StopForward:
            pass
        return self.acts


class Patch:
    """Replace a site's output during forward passes: fn(site_input, site_output) -> new_output."""

    def __init__(self, model: nn.Module, site: Site, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.module = model.get_submodule(site.path)
        self.fn = fn
        self._handle = None

    def __enter__(self) -> "Patch":
        def hook(module, args, kwargs, output):
            new = self.fn(_hook_input(args, kwargs), first_tensor(output))
            return replace_first(output, new.to(first_tensor(output).dtype))

        self._handle = self.module.register_forward_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc) -> None:
        self._handle.remove()
