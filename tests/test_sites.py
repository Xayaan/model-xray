import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from xray.model import block_kind, describe
from xray.sites import Capture, Patch, resolve_sites


@pytest.fixture(scope="module")
def llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2)
    return LlamaForCausalLM(cfg).eval()


def test_resolve_sites(llama):
    mlp = resolve_sites(llama, ["mlp:all"])
    assert [s.path for s in mlp] == [f"model.layers.{i}.mlp" for i in range(3)]
    assert [s.name for s in resolve_sites(llama, ["resid:0,2"])] == ["L0.resid", "L2.resid"]
    assert [s.path for s in resolve_sites(llama, ["mixer:1"])] == ["model.layers.1.self_attn"]
    assert [s.layer for s in resolve_sites(llama, ["mlp:evenly2"])] == [0, 2]
    with pytest.raises(ValueError):
        resolve_sites(llama, ["mlp:7"])


def test_capture_shapes_and_early_stop(llama):
    sites = resolve_sites(llama, ["mlp:0,1", "resid:1"])
    ids = torch.randint(0, 100, (2, 10))
    with Capture(llama, sites, need_input={"L0.mlp"}) as cap:
        acts = cap.run(ids)
        assert cap.output is None  # stopped after L1.resid
    assert acts["L0.mlp"]["in"].shape == acts["L0.mlp"]["out"].shape == (2, 10, 32)
    assert "in" not in acts["L1.mlp"]
    assert acts["L1.resid"]["out"].shape == (2, 10, 32)


def test_patch_identity_and_zero(llama):
    site = resolve_sites(llama, ["mlp:1"])[0]
    ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        clean = llama(ids).logits
        with Patch(llama, site, lambda inp, out: out):
            same = llama(ids).logits
        with Patch(llama, site, lambda inp, out: torch.zeros_like(out)):
            zeroed = llama(ids).logits
    assert torch.allclose(clean, same)
    assert not torch.allclose(clean, zeroed)


@pytest.mark.network
@pytest.mark.parametrize("name, kinds, n_mlp", [
    # hybrids
    ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", {"mamba": 21, "mlp": 17, "attention": 4}, 17),  # single-mixer blocks
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", {"mamba": 40, "moe": 40, "attention": 8}, 40),
    ("Qwen/Qwen3.6-35B-A3B", {"linear_attention": 30, "attention": 10}, 40),  # Gated DeltaNet + MoE
    ("ornith-ai/Ornith-1.5-9B", {"linear_attention": 24, "attention": 8}, 32),
    ("ibm-granite/granite-4.0-h-tiny", {"mamba": 36, "attention": 4}, 40),
    ("ai21labs/AI21-Jamba-Reasoning-3B", {"mamba": 26, "attention": 2}, 28),
    ("tiiuae/Falcon-H1-1.5B-Instruct", {"attention+mamba": 24}, 24),  # Mamba and attention side by side
    ("LiquidAI/LFM2-1.2B", {"conv": 10, "attention": 6}, 16),
    ("state-spaces/mamba-130m-hf", {"mamba": 24}, 0),  # pure Mamba has no MLPs
    # mixture of experts and dense transformers
    ("google/gemma-4-26B-A4B-it", {"attention": 30}, 30),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", {"attention": 43}, 43),
    ("openai/gpt-oss-120b", {"attention": 36}, 36),
    ("zai-org/GLM-4.7-Flash", {"attention": 47}, 47),
    ("unsloth/Llama-4-Scout-17B-16E-Instruct", {"attention": 48}, 48),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", {"attention": 40}, 40),  # vision-language model
])
def test_block_detection_on_real_architectures(name, kinds, n_mlp):
    from xray.model import build_meta

    model = build_meta(name)
    counts = {}
    for block in describe(model):
        counts[block["kind"]] = counts.get(block["kind"], 0) + 1
    assert counts == kinds
    assert len(resolve_sites(model, ["mlp:all"])) == n_mlp


@pytest.mark.network
def test_hybrid_nemotron_sites():
    from xray.config import ModelConfig
    from xray.model import load_model

    model, _ = load_model(ModelConfig(
        name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", dtype="float32", device="cpu",
        random_init=dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, mamba_num_heads=8, mamba_head_dim=16, n_groups=2, ssm_state_size=16,
                         time_step_rank=8, chunk_size=32, num_hidden_layers=6,
                         layers_block_type=["linear_attention", "mlp", "linear_attention", "full_attention", "mlp", "linear_attention"]),
    ))
    kinds = [b["kind"] for b in describe(model)]
    assert kinds == ["mamba", "mlp", "mamba", "attention", "mlp", "mamba"]
    mlp = resolve_sites(model, ["mlp:all"])
    assert [s.layer for s in mlp] == [1, 4] and all(s.path.endswith(".mixer") for s in mlp)
    assert [s.layer for s in resolve_sites(model, ["mixer:all"])] == [0, 2, 3, 5]
    ids = torch.randint(0, 1000, (2, 12))
    with Capture(model, mlp, need_input={s.name for s in mlp}) as cap:
        acts = cap.run(ids)
    assert acts["L4.mlp"]["in"].shape == (2, 12, 64)
    assert block_kind(model.model.layers[0]) == "mamba"
