"""Stream text from Hugging Face datasets or local files and pack it into fixed-length token batches."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

import torch


def _source_texts(src: dict[str, Any], tokenizer, seed: int, shuffle_buffer: int,
                  rank: int = 0, world: int = 1) -> Iterator[str]:
    """Texts from one source. With world > 1 each process reads a disjoint share."""
    while True:
        produced = False
        if "file" in src:
            blocks = [b for b in Path(src["file"]).read_text().split("\n\n") if b.strip()]
            for i, block in enumerate(blocks):
                if i % world == rank:
                    produced = True
                    yield block
        elif "hf" in src:
            from datasets import load_dataset

            ds = load_dataset(src["hf"], src.get("name"), split=src.get("split", "train"), streaming=True)
            if world > 1:
                from datasets.distributed import split_dataset_by_node

                ds = split_dataset_by_node(ds, rank=rank, world_size=world)
            if shuffle_buffer > 0:
                ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
            for row in ds:
                if "chat_field" in src:
                    messages = row[src["chat_field"]]
                    if messages:
                        produced = True
                        yield tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    text = row[src.get("text_field", "text")]
                    if text and text.strip():
                        produced = True
                        yield text
        else:
            raise ValueError(f"data source needs 'hf' or 'file': {src}")
        if not produced or not src.get("repeat", "file" in src):
            return
        seed += 1


def mixed_texts(sources: list[dict[str, Any]], tokenizer, seed: int = 0, shuffle_buffer: int = 2000,
                rank: int = 0, world: int = 1) -> Iterator[str]:
    """Interleave sources at random in proportion to their "weight"."""
    rng = random.Random(seed + rank)
    streams = [_source_texts(src, tokenizer, seed + i, shuffle_buffer, rank, world) for i, src in enumerate(sources)]
    weights = [float(src.get("weight", 1.0)) for src in sources]
    while streams:
        i = rng.choices(range(len(streams)), weights=weights)[0]
        try:
            yield next(streams[i])
        except StopIteration:
            del streams[i], weights[i]


def token_batches(
    tokenizer,
    sources: list[dict[str, Any]],
    ctx_len: int,
    batch_size: int,
    seed: int = 0,
    shuffle_buffer: int = 2000,
    rank: int = 0,
    world: int = 1,
) -> Iterator[torch.Tensor]:
    """Yield (batch_size, ctx_len) LongTensors of packed documents. Each row starts with BOS when the
    tokenizer has one; documents are separated by EOS. With world > 1, process `rank` gets its own
    disjoint share of the documents."""
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    body = ctx_len - (1 if bos is not None else 0)
    buffer: list[int] = []
    for text in mixed_texts(sources, tokenizer, seed, shuffle_buffer, rank, world):
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= body * batch_size:
            rows = []
            for _ in range(batch_size):
                chunk, buffer = buffer[:body], buffer[body:]
                rows.append(([bos] if bos is not None else []) + chunk)
            yield torch.tensor(rows, dtype=torch.long)
