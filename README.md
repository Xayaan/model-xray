# model-xray

Open interpretability tools for **any open-source language model**: the latest releases, such as
Gemma 4, Qwen3.6, DeepSeek V4, gpt-oss, GLM-5, Llama 4, Mistral 3 and Nemotron 3, as well as
established ones. It runs on one GPU, on 2, 4 or 8 GPUs, or with one large model spread over
several GPUs.

**Why this exists:** [the problem, what we solve, and results so far](PROBLEM_AND_SOLUTION.md).

model-xray trains sparse dictionaries (transcoders and sparse autoencoders) on a model's internal
activations. Each dictionary feature is a direction the model uses, and most turn out to stand for
something you can name: a topic, a grammatical role, "this request is harmful", "I don't know this
person". The toolkit then finds the text that triggers each feature, publishes a browsable feature
atlas, and tests what features do by turning them up while the model writes.

## Supported models

Any open decoder-only language model on Hugging Face. `xray info MODEL` shows how a model is laid
out without downloading its weights. These layouts are checked by the test suite:

| Layout | Models checked |
|---|---|
| Dense transformer | Llama 3.2, Qwen3, Gemma 3 and 4, Mistral 7B, Phi-4, OLMo 3, SmolLM3, MiniCPM5 |
| Mixture of experts | gpt-oss-20b and 120b, DeepSeek V3.2 and V4 Flash, Qwen3-30B-A3B, GLM-4.7-Flash and GLM-5.3, MiniMax M2.7, Gemma 4 26B-A4B, Llama 4 Scout |
| Mamba hybrids | Nemotron 3 Nano, Nemotron 3 Super, Nemotron 3.5 Lightning, IBM Granite 4, AI21 Jamba, Falcon-H1 (Mamba and attention side by side) |
| Linear-attention hybrids | Qwen3.5 and Qwen3.6, Qwen3-Next, Kimi Linear |
| Convolution hybrids | Liquid LFM2 |
| State-space only | Mamba |
| Vision-language | Mistral Small 3.2, Ministral 3 (the language model is mapped) |

Models that ship their own modelling code (for example Kimi K2) load with `trust_remote_code: true`.
Gated repositories (for example the official Llama weights) need a Hugging Face token in `HF_TOKEN`.
Encoder-only and encoder-decoder models (BERT, T5) and GGUF files aren't supported; use the
original safetensors weights.

```
$ xray info nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16: 42 blocks {'mamba': 21, 'mlp': 17, 'attention': 4}
mlp sites: [1, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 27, 29, 33, 37, 39, 41]
```

## One GPU, 2/4/8 GPUs, or a model bigger than one GPU

| Situation | How | What happens |
|---|---|---|
| Model fits on one GPU | `xray run CONFIG` | One process does everything |
| Several GPUs, model fits on one | `xray run CONFIG --gpus 8` (or 2, 4) | Each GPU holds a copy of the model and reads different text. The dictionaries are split between the GPUs, and after every forward pass the GPUs swap activations, so each trains its own dictionaries on everyone's tokens. Throughput and dictionary memory grow with the number of GPUs. |
| Model too big for one GPU | `device_map: auto` in the model config | The model is spread over all visible GPUs, and each dictionary trains on the GPU that holds its layer |

`--gpus` works for `train`, `eval`, `features` and `run`, and uses `torchrun` under the hood.

## What's in the box

| Step | Command | What it does |
|---|---|---|
| Inspect | `xray info MODEL` | Block layout and available sites, with no weights downloaded |
| Train | `xray train CONFIG` | TopK transcoders (MLP input to MLP output, with a linear skip) or SAEs, one per site, all trained from the same forward passes |
| Evaluate | `xray eval CONFIG` | Explained variance, L0, dead features, and the loss increase when a dictionary replaces the real activation inside the running model |
| Analyse | `xray features CONFIG` | Each feature's strongest activations on real text, how often it fires, and the output tokens it promotes (logit lens) |
| Browse | `xray browse CONFIG` | A static feature browser (one HTML page plus JSON) you can host on GitHub Pages |
| Findings | `xray findings CONFIG` | Refusal and unknown-person experiments across every trained site: find candidate features, then steer and measure |
| Probe | `xray contrast` / `xray steer` | The same tools for any pair of prompt sets you choose |
| Label | `xray label CONFIG` | Short descriptions for every feature from any OpenAI-compatible LLM endpoint |
| Rent | `lium/run_on_lium.sh CONFIG` | Rents 1 to 8 GPUs on [Lium](https://lium.io), runs the whole pipeline, downloads the results and removes the pod, with a hard dollar budget |

Sites are written as `mlp:all`, `mlp:evenly4`, `resid:0-10` or `mixer:3,7`.

## Quickstart (laptop, about 10 minutes, no GPU)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/xray run configs/smoke-smollm2.yaml --embed
open runs/smoke-smollm2/features/index.html
```

The smoke config trains three transcoders on SmolLM2-135M from Wikipedia text and builds a browser
that opens straight from disk. On an Apple M4 Max CPU it took 11 minutes end to end:

| Site | Explained variance | L0 | CE increase when spliced in | Loss recovered | Dead |
|---|---|---|---|---|---|
| L0.mlp | 0.920 | 31.4 | 0.137 | 0.982 | 0% |
| L14.mlp | 0.683 | 31.9 | 0.016 | 0.831 | 0% |
| L29.mlp | 0.957 | 31.9 | 0.216 | 0.842 | 0% |

`xray browse CONFIG --sample 250` writes a small `preview.html` with 250 random features per site.

## Run on rented GPUs

```bash
lium/run_on_lium.sh configs/nemotron3-nano-4b-pilot.yaml                     # 1x RTX 5090, capped at $3
COUNT=8 GPU=H100 BUDGET=60 TTL=6h lium/run_on_lium.sh configs/nemotron3-nano-4b-full.yaml
```

The script needs the `lium` CLI logged in. Lium removes the pod by itself once `BUDGET` dollars
or `TTL` is reached, so an unattended run can't keep billing. Every run starts with a two-minute
rehearsal on a tiny Nemotron (`configs/preflight-nemotron-tiny.yaml`), so setup problems surface
before the long run starts billing.

## Findings workflow

```bash
# Refusal and unknown-person experiments at every trained site, steering only harmless prompts.
xray findings configs/nemotron3-nano-4b-pilot.yaml

# Or any pair of prompt sets: which features fire more on A than B at the last prompt token?
xray contrast configs/nemotron3-nano-4b-pilot.yaml --site L27.mlp \
  --a hf:walledai/AdvBench --b hf:tatsu-lab/alpaca:instruction --limit 200

# Turn a candidate up (in multiples of its strongest natural activation) and measure the effect.
xray steer configs/nemotron3-nano-4b-pilot.yaml --site L27.mlp --feature <id> --relative \
  --scales 0 1 2 4 --prompts hf:tatsu-lab/alpaca:instruction --limit 50
```

## Output files

```
runs/<name>/
  dictionaries/<site>/folded.safetensors   W_enc, b_enc, W_dec, b_dec (and W_skip) acting on raw activations
  dictionaries/<site>/cfg.json             shapes, k, site, training tokens, dead fraction
  eval.json                                quality numbers per site
  features/                                browser data and index.html
  findings/                                contrast and steering results
```

Folded weights use the usual TopK formulas (see `xray/sae.py`), so they load into other tools
such as SAELens or Neuronpedia with a small adapter.

## Roadmap

1. **Multi-node clusters.** Run one job across several machines, for example Lium's multi-node
   clusters on one InfiniBand/RoCE fabric. This combines the GPU-to-GPU activation exchange across
   nodes with model sharding inside each node, for the largest open models at full precision
   (DeepSeek V4, Kimi K2, GLM-5).
2. **Every mixer as a site.** Dictionaries for Mamba, linear-attention and convolution layers by
   default, and Falcon-H1's parallel Mamba branch as its own site.
3. **Public feature atlases** for the most-used open models that have none, starting with
   Nemotron 3 Nano.
4. **A label for every feature**, generated by an open model.
5. **Releases on Hugging Face and Neuronpedia.**

## Methods

TopK sparse autoencoders, Gao et al. 2024. Transcoders, Dunefsky et al. 2024. Skip transcoders,
Paulo et al. 2025. AuxK dead-feature revival and the decoder-norm constraint follow Gao et al.

MIT license.
