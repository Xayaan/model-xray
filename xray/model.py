"""Load any open Hugging Face language model and describe its blocks.

Two block layouts are handled:
  composite blocks  a token mixer (attention, linear attention, Mamba or convolution) plus an MLP or
                    mixture-of-experts layer in every block: Llama, Qwen, Gemma, Mistral, DeepSeek,
                    gpt-oss, GLM, and hybrids such as Qwen3.5, Granite 4, Jamba and LFM2
  single-mixer      each block holds exactly one mixer, which is Mamba, attention or the MLP itself:
                    NVIDIA Nemotron-H, pure Mamba
Block kinds come from the mixer modules themselves, because implementations reuse the same
`block_type` labels for different things (Nemotron-H's "linear_attention" is Mamba; Qwen3.5's is
Gated DeltaNet).

Vision-language models load through their image-text-to-text class and are read through their
language model. Models too large for one GPU can be spread over several with device_map: auto.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .config import ModelConfig

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

MLP_ATTRS = ("mlp", "feed_forward", "ffn", "block_sparse_moe")
MIXER_ATTRS = ("mixer", "self_attn", "linear_attn", "attention", "mamba", "conv")


def pick_device(pref: str = "auto") -> torch.device:
    # "auto" skips MPS: several ops used here (embedding_bag, scatter_reduce) are incomplete on it.
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_class(config, trust_remote_code: bool = False):
    """The auto class that builds a full language model (with its output head) for this config."""
    if trust_remote_code and "AutoModelForCausalLM" in (getattr(config, "auto_map", None) or {}):
        return AutoModelForCausalLM
    for auto in (AutoModelForCausalLM, AutoModelForImageTextToText):
        if type(config) in auto._model_mapping:
            return auto
    return AutoModelForCausalLM  # raises a clear "unrecognized configuration" error


def build_meta(name: str, trust_remote_code: bool = False) -> nn.Module:
    """The model's architecture without weights (no download, no memory), for inspecting its layout."""
    config = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        return model_class(config, trust_remote_code).from_config(config, trust_remote_code=trust_remote_code)


def load_model(cfg: ModelConfig, device: torch.device | None = None) -> tuple[nn.Module, object]:
    """Load the model and tokenizer. `device` overrides cfg.device (multi-GPU runs pass each
    process its own GPU)."""
    device = device or pick_device(cfg.device)
    dtype = DTYPES[cfg.dtype]
    if device.type == "cuda" or cfg.device_map:
        # Dictionaries train in float32; TF32 matmuls are several times faster and accurate enough.
        torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(cfg.name, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code)
    config = AutoConfig.from_pretrained(cfg.name, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code)
    auto = model_class(config, cfg.trust_remote_code)
    if cfg.random_init is not None:
        for key, value in cfg.random_init.items():
            setattr(config, key, value)
        torch.manual_seed(0)
        model = auto.from_config(config, trust_remote_code=cfg.trust_remote_code)
        model.to(dtype)
    else:
        extra = {"device_map": cfg.device_map} if cfg.device_map else {}
        model = auto.from_pretrained(
            cfg.name, revision=cfg.revision, dtype=dtype, trust_remote_code=cfg.trust_remote_code, **extra
        )
    if not cfg.device_map:
        model.to(device)
    model.eval().requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def input_device(model: nn.Module) -> torch.device:
    """Where token ids go: the embedding's device (models spread over GPUs start there)."""
    return model.get_input_embeddings().weight.device


def module_device(module: nn.Module) -> torch.device:
    """Device of a module's own weights, falling back to its children's."""
    for tensor in module.parameters():
        return tensor.device
    return torch.device("cpu")


def _num_layers(model: nn.Module) -> int | None:
    config = model.config
    n = getattr(config, "num_hidden_layers", None)
    if n is None and hasattr(config, "text_config"):
        n = getattr(config.text_config, "num_hidden_layers", None)
    return n


def find_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Return (dotted path, ModuleList) of the language model's decoder blocks."""
    n = _num_layers(model)
    best: tuple[str, nn.ModuleList] | None = None
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or (n is not None and len(module) != n):
            continue
        # Multimodal models also carry vision and audio blocks; prefer the language model's list.
        if "vision" in name or "visual" in name or "audio" in name:
            continue
        if best is None or name.endswith("layers"):
            best = (name, module)
    if best is None:
        raise ValueError("could not find the decoder layer list")
    return best


def mixer_attr(block: nn.Module) -> str | None:
    for attr in MIXER_ATTRS:
        if isinstance(getattr(block, attr, None), nn.Module):
            return attr
    return None


def _token_mixer_kind(name: str) -> str | None:
    if "mamba" in name:
        return "mamba"
    if "deltanet" in name or "linear" in name:
        return "linear_attention"
    if "conv" in name:
        return "conv"
    if "attention" in name or "attn" in name:
        return "attention"
    return None


def _mixer_kinds(block: nn.Module) -> list[str]:
    """Kinds of every token mixer in a composite block; parallel hybrids (Falcon-H1) have two."""
    kinds = []
    for attr in MIXER_ATTRS:
        module = getattr(block, attr, None)
        if isinstance(module, nn.Module):
            kind = "mamba" if attr == "mamba" else _token_mixer_kind(type(module).__name__.lower())
            if kind and kind not in kinds:
                kinds.append(kind)
    return kinds


def block_kind(block: nn.Module) -> str:
    """mamba / linear_attention / conv / attention for the block's token mixer ("attention+mamba"
    when both run side by side), or mlp / moe for a single-mixer block whose mixer is the
    feed-forward layer (Nemotron-H)."""
    attr = mixer_attr(block)
    name = type(getattr(block, attr)).__name__.lower() if attr else ""
    if any(isinstance(getattr(block, a, None), nn.Module) for a in MLP_ATTRS):
        return "+".join(sorted(_mixer_kinds(block))) or "attention"
    kind = _token_mixer_kind(name)
    if kind:
        return kind
    if "moe" in name or "expert" in name:
        return "moe"
    if "mlp" in name:
        return "mlp"
    return getattr(block, "block_type", None) or "unknown"


def mlp_attr(block: nn.Module) -> str | None:
    """Name of the block's MLP (or MoE) submodule, or None for blocks without one."""
    for attr in MLP_ATTRS:
        if isinstance(getattr(block, attr, None), nn.Module):
            return attr
    if block_kind(block) in ("mlp", "moe"):
        return mixer_attr(block)
    return None


def describe(model: nn.Module) -> list[dict]:
    _, layers = find_layers(model)
    return [
        {"layer": i, "kind": block_kind(block), "mlp": mlp_attr(block), "mixer": mixer_attr(block)}
        for i, block in enumerate(layers)
    ]


def unembedding(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The unembedding matrix (vocab x d_model) and the final norm's scale, for logit-lens views."""
    W_U = model.get_output_embeddings().weight
    layers_path, _ = find_layers(model)
    parent = model.get_submodule(layers_path.rsplit(".", 1)[0]) if "." in layers_path else model
    for attr in ("norm_f", "norm", "final_layernorm", "ln_f"):
        norm = getattr(parent, attr, None)
        if norm is not None and getattr(norm, "weight", None) is not None:
            return W_U, norm.weight
    return W_U, None
