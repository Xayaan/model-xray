"""Run configuration. One YAML file describes the model, the data, the dictionaries and the token budget."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str
    dtype: str = "bfloat16"
    device: str = "auto"
    revision: str | None = None
    trust_remote_code: bool = False
    # "auto" spreads a model that doesn't fit on one GPU over all visible GPUs. Each dictionary
    # then trains on the GPU that holds its layer.
    device_map: str | None = None
    # Build a randomly initialised model from the hub config with these overrides instead of
    # downloading weights. Tests use it to exercise an architecture without a multi-GB download.
    random_init: dict[str, Any] | None = None


@dataclass
class DataConfig:
    # Each source is {"hf": repo, "name": subset, "split": ..., "text_field": ...} or
    # {"hf": repo, "chat_field": "messages"} (rendered with the model's chat template) or {"file": path}.
    # "weight" sets the mixing proportion between sources.
    sources: list[dict[str, Any]] = field(default_factory=list)
    eval_sources: list[dict[str, Any]] | None = None
    ctx_len: int = 1024
    batch_size: int = 8
    skip_first: int = 1  # positions dropped from every sequence (the BOS / attention-sink token)
    shuffle_buffer: int = 2000
    seed: int = 0


@dataclass
class DictionaryConfig:
    kind: str = "transcoder"  # "transcoder": MLP input -> MLP output; "sae": reconstruct one activation
    expansion: int = 8
    k: int = 64
    skip: bool = True  # linear skip connection (transcoders only)
    lr: float | None = None  # default 2e-4 / sqrt(d_sae / 2**14), as in Gao et al. (2024)
    aux_k: int | None = None  # default min(512, d_in // 2)
    aux_coef: float = 1 / 32
    dead_after_tokens: int = 10_000_000


@dataclass
class TrainConfig:
    sites: list[str] = field(default_factory=lambda: ["mlp:all"])
    total_tokens: int = 100_000_000
    max_minutes: float | None = None  # stop training early so later stages fit a rented-GPU budget
    train_batch: int = 4096
    buffer_batches: int = 8  # shuffle buffer size, in training batches
    warmup_steps: int = 100
    decay_frac: float = 0.2
    log_every: int = 50
    save_every_tokens: int = 50_000_000


@dataclass
class EvalConfig:
    n_tokens: int = 200_000
    batch_size: int = 4


@dataclass
class AnalysisConfig:
    n_tokens: int = 2_000_000
    top_examples: int = 12
    window_left: int = 24
    window_right: int = 6
    chunk_size: int = 256  # features per file in the published browser data


SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "dictionary": DictionaryConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "analysis": AnalysisConfig,
}


@dataclass
class RunConfig:
    name: str
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    out_root: str = "runs"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / self.name

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in SECTIONS:
                kwargs[key] = _build(SECTIONS[key], value or {})
            elif key in ("name", "out_root"):
                kwargs[key] = value
            else:
                raise ValueError(f"unknown config section: {key}")
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)
