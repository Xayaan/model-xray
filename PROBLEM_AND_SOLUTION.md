# model-xray: the problem, what we solve, and why this is the right solution

## In short

Open language models now run inside products everywhere, and almost none of them can be inspected.
When a model refuses a reasonable request or states something false, the people running it have no
way to see why. Tools that open up a model's internals exist for a handful of model families,
mostly published by the labs that built them. Most other open models have none, including many of
the latest releases. Those releases also use new architectures (mixture of experts, and hybrids
that mix attention with Mamba, linear attention or convolutions) that most interpretability code
doesn't handle.

**model-xray** is an open-source toolkit that maps **any open-source language model**, including
the latest releases, into readable features. It checks how faithful that map is, publishes it as a
browsable atlas, and tests what individual features do by turning them up while the model writes.
It runs on one GPU, on 2, 4 or 8 GPUs, or with one large model spread over several GPUs. It is built
and tested: its layout detection is checked on 35 open models and its full pipeline has run end to
end. Its first worked example is NVIDIA's Nemotron 3 Nano.

---

## The problem

### 1. Open models are everywhere, and their insides are invisible

Open models are downloaded at enormous scale. Monthly Hugging Face downloads in October 2026 include
**Gemma 4 26B-A4B at 12.5M**, **gpt-oss-20b at 6.2M**, **DeepSeek V4 Flash at 4.5M**, **Qwen3.6-35B-A3B
at 3.5M**, **Nemotron 3 Nano 4B at 2.3M** and **GLM-4.7-Flash at 1.6M**. Companies build products on
them. When a model misbehaves, they can change prompts and hope, but they cannot see which internal
concepts drove a decision, such as "this request looks harmful" or "I don't recognise this name".

### 2. The tools exist for only a handful of model families

The standard way to see inside a model is a set of trained sparse dictionaries (sparse autoencoders
or transcoders). Each one breaks a layer's activity into thousands of features, and most features
stand for something nameable. As of October 2026, Neuronpedia, the main public hub for these, lists
releases for Gemma, Qwen, Llama, gpt-oss-20b, OLMo 3 and DeepSeek V4 Flash. Most come from the labs
themselves (Google's Gemma Scope 2, Alibaba's Qwen-Scope). Widely used families with nothing listed
include **NVIDIA Nemotron, Zhipu GLM, MiniMax, Moonshot Kimi, IBM Granite, Mistral and Microsoft
Phi**. Each release also covers a single family, so every new model starts from zero.

### 3. Architectures are changing faster than the tools

Most interpretability code assumes a plain transformer, where every block has attention followed by
an MLP. The latest open models increasingly don't look like that:

| Layout | Examples |
|---|---|
| Mixture of experts | gpt-oss, DeepSeek V3.2 and V4, GLM-4.7 and GLM-5, MiniMax M2.7, Gemma 4 26B-A4B, Qwen3.6-35B-A3B, Llama 4 |
| Mamba hybrids | Nemotron 3 (21 Mamba, 17 MLP and 4 attention blocks in the 4B), IBM Granite 4, AI21 Jamba, Falcon-H1 |
| Linear-attention hybrids | Qwen3.5 and Qwen3.6, Qwen3-Next, Kimi Linear |
| Convolution hybrids | Liquid LFM2 |

We don't yet know how these models divide work between their block types, or whether findings from
plain transformers carry over.

### 4. It takes GPU time, and big models need many GPUs

Training dictionaries means running the model over hundreds of millions of tokens while training
thousands of features per layer. Google reports that Gemma Scope 2 took about 110 petabytes of
stored activations and over a trillion trained parameters. The newest open models are also too big
for one GPU, so both the model and the dictionaries must be spread over several. Without tooling
that does this efficiently, popular models that their labs don't cover stay dark.

---

## What we solve

| Who | Their problem today | What model-xray gives them |
|---|---|---|
| Teams running open models in products | Can't see why a model refuses a legitimate request or makes something up | The features behind those behaviours, shown on real text, and proof by steering that they cause them |
| Safety and evaluation researchers | Tools exist for a few model families, and few cover hybrid or mixture-of-experts models | One toolkit that maps whichever model they use, including the newest architectures |
| Interpretability researchers | Every new model or architecture means new plumbing | A small, tested codebase that reads a model's layout itself and scales from a laptop to 8 GPUs |
| The open-source community | Most popular models have no public feature atlas | Atlases and weights in the standard format for Neuronpedia and SAELens, for any model |

---

## Our solution: model-xray

1. **Load any open model.** Dense, mixture-of-experts, Mamba, linear-attention, convolution and
   vision-language models load through Hugging Face. The toolkit reads each model's own block layout,
   so it doesn't need a hand-written port. Models with custom code load with `trust_remote_code`.
2. **Train.** For each chosen MLP block, a *transcoder* learns to rewrite the block's computation
   as a sum of a few features out of tens of thousands. All blocks train from the same forward
   passes, with a shuffle buffer so each step mixes many documents. Residual-stream and mixer
   SAEs use the same code.
3. **Scale.** `--gpus 2`, `4` or `8` gives each GPU a model copy and its own text, splits the
   dictionaries between the GPUs, and swaps activations after every forward pass. `device_map: auto`
   spreads a model too big for one GPU over all of them.
4. **Check.** Every dictionary reports how much of the real activity it explains, how many features
   died, and the model's loss when the dictionary *replaces* the real block inside the running model.
5. **Map.** For every feature it finds the text that triggers it most, how often it fires, and which
   next tokens it pushes up or down. It then publishes all of this as a static feature browser.
6. **Test.** The findings step compares two prompt sets, for example harmful vs harmless requests or
   made-up vs famous people, to find candidate features. It then adds a candidate to the model while
   it answers held-out prompts and measures whether the behaviour follows. That turns a correlation
   into causal evidence.
7. **Run safely on rented GPUs.** One command rents 1 to 8 GPUs on Lium and runs a 2-minute
   rehearsal, then the whole pipeline with a training time limit. It downloads the results and
   removes the pod, and Lium enforces a hard dollar budget.

### Why this is the right solution

- **It covers the latest models.** Its layout detection is tested on 35 open models, including
  Gemma 4, Qwen3.6, DeepSeek V4 Flash, GLM-5.3, MiniMax M2.7, Llama 4, gpt-oss-120b and Nemotron 3
  Super.
- **It scales.** Multi-GPU training, evaluation and analysis are tested as real multi-process runs,
  including the uneven case where dictionaries don't divide evenly across GPUs.
- **It already works.** It has 35 automated tests, has run end to end on a laptop, and has set itself
  up and run its pipeline on a rented GPU.
- **It holds itself to measurable standards.** Every release ships with its quality numbers, and
  findings are backed by steering experiments, not just examples.
- **It is open and standard.** The code is MIT-licensed and the weights are exported in the usual
  TopK format, so other tools can load them.
- **It is responsible by design.** Steering is only ever applied to *harmless* prompts. We test
  whether a feature makes the model refuse more, and never try to make it comply with harmful
  requests.

---

## Results so far

**Coverage.** `xray info` reads the layout of 35 open models correctly, and the test suite checks 15
of them on every run. Coverage includes every layout in the table above, plus vision-language
models (Mistral Small 3.2, Ministral 3).

**End-to-end run on a laptop (SmolLM2-135M, 1.5M training tokens, 11 minutes on an Apple M4 Max CPU).**
Three MLP transcoders with 4,608 features each:

| Block | Activity explained | Loss increase when the transcoder replaces the block | Loss recovered vs. removing the block | Dead features |
|---|---|---|---|---|
| L0 | 92.0% | +0.137 | 98.2% | 0% |
| L14 | 68.3% | +0.016 | 83.1% | 0% |
| L29 | 95.7% | +0.216 | 84.2% | 0% |

Features are readable even at this small scale. One fires on the number inside "rises to 87 metres
(285"; another fires on the "that" in "so angered Lee that he decided".

**Multi-GPU.** `xray run CONFIG --gpus 2` ran train, evaluate and analyse as two processes using the
same collective operations GPUs use. Each process trained its own share of the dictionaries on
tokens from both, and the results merged into one atlas.

**Validation on a rented GPU (Lium, one H100, 17 minutes, stopped early by choice).** The pod set
itself up, and the full pipeline (train, evaluate, analyse features) ran on a tiny Nemotron-H. We
also measured the real model: **Nemotron 3 Nano 4B runs its forward pass at 12,784 tokens per second
on one H100** with the standard PyTorch Mamba code (batches of 8 x 1,024 tokens).

**Not yet run: the Nemotron pilot.** It covers 4 MLP blocks plus the refusal and unknown-person
findings, fits one RTX 5090, and launches with one command
(`lium/run_on_lium.sh configs/nemotron3-nano-4b-pilot.yaml`).

---

## Roadmap

1. **Multi-node clusters.** Run one job across several machines, for example Lium's multi-node
   clusters on one InfiniBand/RoCE fabric. This combines the GPU-to-GPU activation exchange across
   nodes with model sharding inside each node, for the largest open models at full precision.
2. **Every mixer as a site.** Dictionaries for Mamba, linear-attention and convolution layers by
   default, and Falcon-H1's parallel Mamba branch as its own site.
3. **Public feature atlases** for the most-used open models that have none, starting with
   Nemotron 3 Nano and continuing with GLM, MiniMax and Granite.
4. **A label for every feature**, generated by an open model.
5. **Releases on Hugging Face and Neuronpedia.**

---

## Limitations we are upfront about

- Dictionaries explain most, not all, of a block's activity, and some features mix several concepts.
  We publish the quality numbers alongside every release.
- Layout detection has been checked on 35 models, but full training has so far run on SmolLM2 and
  on small Nemotron-H models. No full-size Nemotron has been mapped yet; the pilot comes first.
- Multi-GPU runs have been tested as multiple processes on one machine's CPU. The first run on real
  multi-GPU hardware will confirm throughput.
- Refusal and "I don't know" answers are detected by phrase matching, which can miss unusual
  wordings. The raw completions are saved, so anyone can check them.
- Encoder-only and encoder-decoder models (BERT, T5) aren't supported, and GGUF files need their
  original weights.
